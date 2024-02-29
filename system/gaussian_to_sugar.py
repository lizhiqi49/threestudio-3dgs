import os
import random
from dataclasses import dataclass, field

import numpy as np
import threestudio
import torch
import torch.nn.functional as F
from threestudio.systems.base import BaseLift3DSystem
from threestudio.systems.utils import parse_optimizer, parse_scheduler
from threestudio.utils.loss import tv_loss
from threestudio.utils.ops import get_cam_info_gaussian
from threestudio.utils.typing import *
from torch.cuda.amp import autocast
from torchmetrics import PearsonCorrCoef

from easydict import EasyDict

from ..geometry.gaussian_base import BasicPointCloud, Camera
from ..utils.sugar_utils import SuGaRRegularizer



@threestudio.register("gs-to-sugar-zero123-system")
class Zero123(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        freq: dict = field(default_factory=dict)
        refinement: bool = False
        ambient_ratio_min: float = 0.5
        back_ground_color: Tuple[float, float, float] = (1, 1, 1)

        # ============= SuGaR regularization configs ============= #
        use_sugar_reg: bool = True
        min_opac_prune: Any = 0.5
        knn_to_track: int = 16
        n_samples_for_sugar_sdf_reg: int = 500000

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        self.automatic_optimization = False

    def configure_optimizers(self):
        optim = self.geometry.optimizer
        if hasattr(self, "merged_optimizer"):
            return [optim]
        if hasattr(self.cfg.optimizer, "name"):
            net_optim = parse_optimizer(self.cfg.optimizer, self)
            optim = self.geometry.merge_optimizer(net_optim)
            self.merged_optimizer = True
        else:
            self.merged_optimizer = False
        return [optim]

    def on_load_checkpoint(self, checkpoint):
        num_pts = checkpoint["state_dict"]["geometry._xyz"].shape[0]
        pcd = BasicPointCloud(
            points=np.zeros((num_pts, 3)),
            colors=np.zeros((num_pts, 3)),
            normals=np.zeros((num_pts, 3)),
        )
        self.geometry.create_from_pcd(pcd, 10)
        self.geometry.training_setup()
        return

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        self.geometry.update_learning_rate(self.global_step)
        outputs = self.renderer.batch_forward(batch)
        return outputs

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # no prompt processor
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

        # visualize all training images
        all_images = self.trainer.datamodule.train_dataloader().dataset.get_all_images()
        self.save_image_grid(
            "all_training_images.png",
            [
                {"type": "rgb", "img": image, "kwargs": {"data_format": "HWC"}}
                for image in all_images
            ],
            name="on_fit_start",
            step=self.true_global_step,
        )

        self.sugar_reg = None

        self.pearson = PearsonCorrCoef().to(self.device)

    def training_substep(self, batch, batch_idx, guidance: str):
        """
        Args:
            guidance: one of "ref" (reference image supervision), "zero123"
        """
        if guidance == "ref":
            ambient_ratio = 1.0
            shading = "diffuse"
            batch["shading"] = shading
        elif guidance == "zero123":
            batch = batch["random_camera"]
            ambient_ratio = (
                self.cfg.ambient_ratio_min
                + (1 - self.cfg.ambient_ratio_min) * random.random()
            )

        batch["ambient_ratio"] = ambient_ratio
        batch.update(
            {
                "override_bg_color": torch.ones([1, 3], dtype=torch.float32, device=self.device)
            }
        )

        out = self(batch)
        loss_prefix = f"loss_{guidance}_"

        loss_terms = {}

        def set_loss(name, value):
            loss_terms[f"{loss_prefix}{name}"] = value

        guidance_eval = (
            guidance == "zero123"
            and self.cfg.freq.guidance_eval > 0
            and self.true_global_step % self.cfg.freq.guidance_eval == 0
        )

        if guidance == "ref":
            gt_mask = batch["mask"]
            gt_rgb = batch["rgb"]

            # color loss
            gt_rgb = gt_rgb * gt_mask.float()
            set_loss("rgb", F.mse_loss(gt_rgb, out["comp_rgb"] * gt_mask.float()))

            # mask loss
            set_loss("mask", F.mse_loss(gt_mask.float(), out["comp_mask"]))

            # depth loss
            if self.C(self.cfg.loss.lambda_depth) > 0:
                valid_gt_depth = batch["ref_depth"][gt_mask.squeeze(-1)].unsqueeze(1)
                valid_pred_depth = out["comp_depth"][gt_mask].unsqueeze(1)
                with torch.no_grad():
                    A = torch.cat(
                        [valid_gt_depth, torch.ones_like(valid_gt_depth)], dim=-1
                    )  # [B, 2]
                    X = torch.linalg.lstsq(A, valid_pred_depth).solution  # [2, 1]
                    valid_gt_depth = A @ X  # [B, 1]
                set_loss("depth", F.mse_loss(valid_gt_depth, valid_pred_depth))

            # relative depth loss
            if self.C(self.cfg.loss.lambda_depth_rel) > 0:
                valid_gt_depth = batch["ref_depth"][gt_mask.squeeze(-1)]  # [B,]
                valid_pred_depth = out["comp_depth"][gt_mask]  # [B,]
                set_loss(
                    "depth_rel", 1 - self.pearson(valid_pred_depth, valid_gt_depth)
                )

            # normal loss
            if self.C(self.cfg.loss.lambda_normal) > 0:
                valid_gt_normal = (
                    1 - 2 * batch["ref_normal"][gt_mask.squeeze(-1)]
                )  # [B, 3]
                valid_pred_normal = (
                    2 * out["comp_normal"][gt_mask.squeeze(-1)] - 1
                )  # [B, 3]
                set_loss(
                    "normal",
                    1 - F.cosine_similarity(valid_pred_normal, valid_gt_normal).mean(),
                )
        elif guidance == "zero123":
            # zero123
            guidance_out = self.guidance(
                out["comp_rgb"],
                **batch,
                rgb_as_latents=False,
                guidance_eval=guidance_eval,
            )
            # claforte: TODO: rename the loss_terms keys
            set_loss("sds", guidance_out["loss_sds"])

        # Regularizations
        if self.C(self.cfg.loss.lambda_normal_smooth) > 0:
            if "comp_normal" not in out:
                raise ValueError(
                    "comp_normal is required for 2D normal smooth loss, no comp_normal is found in the output."
                )
            normal = out["comp_normal"]
            set_loss(
                "normal_smooth",
                (normal[:, 1:, :, :] - normal[:, :-1, :, :]).square().mean()
                + (normal[:, :, 1:, :] - normal[:, :, :-1, :]).square().mean(),
            )

        # SuGaR reg
        if self.sugar_reg is not None:
            if self.C(self.cfg.loss.lambda_opacity_binary) > 0:
                # only use in static stage
                visibility_filter = out["visibility_filter"]
                opacity = self.geometry.get_opacity.unsqueeze(0).repeat(len(visibility_filter), 1, 1)
                vis_opacities = opacity[torch.stack(visibility_filter)]
                set_loss(
                    "opacity_binary",
                    -(vis_opacities * torch.log(vis_opacities + 1e-10)
                    + (1 - vis_opacities) * torch.log(1 - vis_opacities + 1e-10)).mean()
                )

            if self.C(self.cfg.loss.lambda_sugar_density_reg) > 0:
                use_sdf_normal_reg = self.C(self.cfg.loss.lambda_sugar_sdf_normal_reg) > 0
                coarse_args = EasyDict(
                    {
                        # "outputs": out,
                        "n_samples_for_sdf_regularization": self.cfg.n_samples_for_sugar_sdf_reg,
                        "use_sdf_better_normal_loss": use_sdf_normal_reg,
                    }
                )
                dloss = self.sugar_reg.coarse_density_regulation(coarse_args)
                set_loss("sugar_density_reg", dloss["density_regulation"])
                if use_sdf_normal_reg:
                    set_loss("sugar_sdf_normal_reg", dloss["normal_regulation"])

        if self.cfg.loss["lambda_rgb_tv"] > 0.0:
            loss_rgb_tv = tv_loss(out["comp_rgb"].permute(0, 3, 1, 2))
            set_loss("rgb_tv", loss_rgb_tv)

        if (
            out.__contains__("comp_depth")
            and self.cfg.loss["lambda_depth_tv"] > 0.0
        ):
            loss_depth_tv = tv_loss(out["comp_depth"].permute(0, 3, 1, 2))
            set_loss("depth_tv", loss_depth_tv)

        if (
            out.__contains__("comp_normal")
            and self.cfg.loss["lambda_normal_tv"] > 0.0
        ):
            loss_normal_tv = tv_loss(out["comp_normal"].permute(0, 3, 1, 2))
            set_loss("normal_tv", loss_normal_tv)


        loss = 0.0
        for name, value in loss_terms.items():
            self.log(f"train/{name}", value)
            if name.startswith(loss_prefix):
                loss_weighted = value * self.C(
                    self.cfg.loss[name.replace(loss_prefix, "lambda_")]
                )
                self.log(f"train/{name}_w", loss_weighted)
                loss += loss_weighted

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        self.log(f"train/loss_{guidance}", loss)

        out.update({"loss": loss})
        return out

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        if self.cfg.freq.get("ref_or_zero123", "accumulate") == "accumulate":
            do_ref = True
            do_zero123 = True
        elif self.cfg.freq.get("ref_or_zero123", "accumulate") == "alternate":
            do_ref = (
                self.true_global_step < self.cfg.freq.ref_only_steps
                or self.true_global_step % self.cfg.freq.n_ref == 0
            )
            do_zero123 = not do_ref

        total_loss = 0.0
        if do_zero123:
            out = self.training_substep(batch, batch_idx, guidance="zero123")
            total_loss += out["loss"]

        if do_ref:
            out = self.training_substep(batch, batch_idx, guidance="ref")
            total_loss += out["loss"]

        self.log("train/loss", total_loss, prog_bar=True)

        visibility_filter = out["visibility_filter"]
        radii = out["radii"]
        guidance_inp = out["comp_rgb"]
        viewspace_point_tensor = out["viewspace_points"]

        total_loss.backward()
        iteration = self.global_step
        self.geometry.update_states(
            iteration,
            visibility_filter,
            radii,
            viewspace_point_tensor,
        )
        opt.step()
        opt.zero_grad(set_to_none=True)

        return {"loss": total_loss}
    
    def on_train_batch_start(self, batch, batch_idx, unused=0):
        super().on_train_batch_start(batch, batch_idx, unused)
        if self.cfg.use_sugar_reg and self.global_step >= self.cfg.freq.start_sugar_reg:
            self.sugar_reg = SuGaRRegularizer(
                self.geometry, keep_track_of_knn=True, knn_to_track=self.cfg.knn_to_track
            )
            self.sugar_reg.reset_neighbors(self.cfg.knn_to_track)

        if self.sugar_reg is not None:
            if (
                self.global_step % self.cfg.freq.reset_neighbors == 0
                or self.geometry.pruned_or_densified
            ):
                self.sugar_reg.reset_neighbors(self.cfg.knn_to_track)

            self.geometry.min_opac_prune = self.C(self.cfg.min_opac_prune)

    def validation_step(self, batch, batch_idx):
        batch.update(
            {
                "override_bg_color": torch.ones([1, 3], dtype=torch.float32, device=self.device)
            }
        )
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-val/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": batch["rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
                if "rgb" in batch
                else []
            )
            + [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + (
                [
                    {
                        "type": "grayscale",
                        "img": out["comp_depth"][0, :, :, 0],
                        "kwargs": {"cmap": "jet", "data_range": (0, 1)},
                    }
                ]
                if "comp_depth" in out
                else []
            )
            + (
                [
                    {
                        "type": "grayscale",
                        "img": out["comp_mask"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    }
                ]
                if "comp_mask" in out
                else []
            ),
            # claforte: TODO: don't hardcode the frame numbers to record... read them from cfg instead.
            name=f"validation_step_batchidx_{batch_idx}"
            if batch_idx in [0, 7, 15, 23, 29]
            else None,
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        filestem = f"it{self.true_global_step}-val"
        self.save_img_sequence(
            filestem,
            filestem,
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="validation_epoch_end",
            step=self.true_global_step,
        )

    def test_step(self, batch, batch_idx):
        batch.update(
            {
                "override_bg_color": torch.ones([1, 3], dtype=torch.float32, device=self.device)
            }
        )
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": batch["rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
                if "rgb" in batch
                else []
            )
            + [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            ),
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
        gs_save_path = os.path.join(
            self.get_save_dir(), f"exported_gs_step{self.global_step}.ply"
        )
        self.geometry.save_ply(gs_save_path)

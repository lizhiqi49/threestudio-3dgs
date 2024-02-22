import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
from easydict import EasyDict

import threestudio
import torch
import torch.nn.functional as F

from threestudio.systems.base import BaseLift3DSystem
from threestudio.systems.utils import parse_optimizer, parse_scheduler
from threestudio.utils.loss import tv_loss
from threestudio.utils.ops import get_cam_info_gaussian, convert_pose
from threestudio.utils.typing import *
from threestudio.utils.misc import C
from torch.cuda.amp import autocast
from torchmetrics import PearsonCorrCoef

from ..geometry.gaussian_base import BasicPointCloud, Camera
from ..sugar.sugar_model import SuGaR


@threestudio.register("gaussian-splatting-4dgen-system")
class Gaussian4DGen(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        stage: str = "static"  # ["static", "motion", "refine"]

        # guidances
        prompt_processor_2d_type: Optional[str] = ""
        prompt_processor_2d: dict = field(default_factory=dict)
        guidance_2d_type: Optional[str] = "stable-diffusion-guidance"
        guidance_2d: dict = field(default_factory=dict)

        guidance_zero123_type: str = "stale-zero123-guidance"
        guidance_zero123: dict = field(default_factory=dict)

        prompt_processor_3d_type: Optional[str] = ""
        prompt_processor_3d: dict = field(default_factory=dict)
        guidance_3d_type: Optional[str] = "image-dream-guidance"
        guidance_3d: dict = field(default_factory=dict)

        prompt_processor_vid_type: Optional[str] = ""
        prompt_processor_vid: dict = field(default_factory=dict)
        guidance_vid_type: Optional[str] = ""
        guidance_vid: dict = field(default_factory=dict)

        freq: dict = field(default_factory=dict)
        refinement: bool = False
        ambient_ratio_min: float = 0.5
        back_ground_color: Tuple[float, float, float] = (1, 1, 1)

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        self.automatic_optimization = False
        self.stage = self.cfg.stage

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
        # if hasattr(self.geometry.cfg, "spatial_lr_scale"):
        self.geometry.create_from_pcd(pcd, self.geometry.cfg.get("spatial_lr_scale", 1))
        self.geometry.training_setup()
        # return
        super().on_load_checkpoint(checkpoint)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        self.geometry.update_learning_rate(self.global_step)
        outputs = self.renderer.batch_forward(batch)
        return outputs

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # no prompt processor
        self.guidance_zero123 = threestudio.find(self.cfg.guidance_zero123_type)(self.cfg.guidance_zero123)

        # Maybe use ImageDream
        self.enable_imagedream = self.cfg.guidance_3d_type is not None and C(self.cfg.loss.lambda_sds_3d, 0, 0) > 0
        if self.enable_imagedream:
            self.guidance_3d = threestudio.find(self.cfg.guidance_3d_type)(self.cfg.guidance_3d)
        else:
            self.guidance_3d = None

        # Maybe use video diffusion models
        self.enable_vid = (
            self.stage == "motion"
            and self.cfg.guidance_vid_type is not None
            and C(self.cfg.loss.lambda_sds_vid, 0, 0) > 0
        )
        if self.enable_vid:
            self.guidance_vid = threestudio.find(self.cfg.guidance_vid_type)(self.cfg.guidance_vid)
        else:
            self.guidance_vid = None

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
            # default store the reference view camera config, switch to random camera for zero123 guidance
            batch = batch["random_camera"]
            ambient_ratio = (
                self.cfg.ambient_ratio_min
                + (1 - self.cfg.ambient_ratio_min) * random.random()
            )

        batch["ambient_ratio"] = ambient_ratio

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
            # if self.stage == "static":
            #     set_loss("rgb", F.mse_loss(gt_rgb, out["comp_rgb"] * gt_mask.float()))
            # else:
            #     set_loss("rgb", F.mse_loss(gt_rgb, out["comp_rgb"]) / gt_rgb.shape[0])
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
            guidance_out = self.guidance_zero123(
                out["comp_rgb"],
                **batch,
                rgb_as_latents=False,
                guidance_eval=guidance_eval,
            )
            # claforte: TODO: rename the loss_terms keys
            set_loss("sds", guidance_out["loss_sds"])

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

        ## cross entropy loss for opacity to make it binary
        if self.C(self.cfg.loss.lambda_opacity_binary, "interval") > 0:
            # only use in static stage
            assert self.cfg.stage == 'static'
            visibility_filter = out["visibility_filter"]
            opacity = self.geometry.get_opacity.unsqueeze(0).repeat(len(visibility_filter), 1, 1)
            vis_opacities = opacity[torch.stack(visibility_filter)]
            set_loss(
                "opacity_binary",
                -(vis_opacities * torch.log(vis_opacities + 1e-10)
                  + (1 - vis_opacities) * torch.log(1 - vis_opacities + 1e-10)).mean()
            )

        ## density regulation loss
        if hasattr(self, 'sugar') and self.C(self.cfg.loss.lambda_density_regulation, "interval") > 0:
            coarse_args = EasyDict(
                {"current_step": self.true_global_step,
                 "outputs": out,
                 "n_samples_for_sdf_regularization": self.cfg.sugar.n_samples_for_sdf_regularization,
                 "use_sdf_better_normal_loss": self.cfg.sugar.use_sdf_better_normal_loss,
                 "start_sdf_better_normal_from": self.cfg.sugar.start_sdf_better_normal_from,
                 "timestamp": batch.get('timestamp')
                 })
            dloss = self.sugar.coarse_density_regulation(coarse_args)
            set_loss("density_regulation", dloss['density_regulation'])
            set_loss("normal_regulation", dloss['normal_regulation'])

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

        if self.cfg.sugar.start_regularization_from is not None:
            if self.global_step == self.cfg.sugar.start_regularization_from:
                self.sugar = SuGaR(self.geometry, keep_track_of_knn=True, knn_to_track=self.cfg.sugar.knn_to_track)
                self.sugar.reset_neighbors(timestamp=batch.get('timestamp'))
            elif self.global_step > self.cfg.sugar.start_regularization_from:
                assert hasattr(self.geometry, "pruned_or_densified")
                # reset neighbors after gaussians densified or settings
                if self.geometry.pruned_or_densified or self.global_step % self.cfg.sugar.reset_neighbors_every == 0:
                    self.geometry.pruned_or_densified = False
                    self.sugar.reset_neighbors(timestamp=batch.get('timestamp'))

        total_loss = 0.0

        self.log(
            "gauss_num",
            int(self.geometry.get_xyz.shape[0]),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        out_zero123 = self.training_substep(batch, batch_idx, guidance="zero123")
        total_loss += out_zero123["loss"]

        out_ref = self.training_substep(batch, batch_idx, guidance="ref")
        total_loss += out_ref["loss"]

        self.log("train/loss", total_loss, prog_bar=True)

        out = out_zero123
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

        ## TODO : add sugar regulation loss
        """
            1. add opacity cross entropy loss between [start step, end step]
            2. add density regulation loss
        """

        return {"loss": total_loss}

    def on_validation_epoch_start(self) -> None:
        if self.stage == 'motion' and self.geometry.cfg.use_spline:
            self.geometry.compute_control_knots()
            self.geometry.spliner.update_end_time()

    def validation_step(self, batch, batch_idx):
        if self.stage != "static" and not batch.__contains__("timestamp"):
            batch.update(
                {
                    "timestamp": torch.as_tensor(
                        [batch["index"] / batch["n_all_views"]], device=self.device
                    )
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
            ),
            # claforte: TODO: don't hardcode the frame numbers to record... read them from cfg instead.
            name=f"validation_step_batchidx_{batch_idx}"
            if batch_idx in [0, 7, 15, 23, 29]
            else None,
            step=self.true_global_step,
        )

        if self.stage != "static":
            if batch["index"] == 0:
                self.batch_ref_eval = batch

            self.batch_ref_eval["timestamp"] = batch["timestamp"]
            out_ref = self(self.batch_ref_eval)
            self.save_image_grid(
                f"it{self.true_global_step}-val-ref/{batch['index'][0]}.png",
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
                        "img": out_ref["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out_ref["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out_ref
                    else []
                ),
                # claforte: TODO: don't hardcode the frame numbers to record... read them from cfg instead.
                name=f"validation_step_batchidx_{batch_idx}-ref"
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

        # debug: save ply at each validation step
        plysavepath = os.path.join(self.get_save_dir(), f"point_cloud_it{self.true_global_step}.ply")
        self.geometry.save_ply(plysavepath)
        # debug

        if self.stage != "static":
            filestem = f"it{self.true_global_step}-val-ref"
            self.save_img_sequence(
                filestem,
                filestem,
                "(\d+)\.png",
                save_format="mp4",
                fps=30,
                name="validation_epoch_end-ref",
                step=self.true_global_step,
            )

    def on_test_epoch_start(self) -> None:
        if self.stage != "static" and self.geometry.cfg.use_spline:
            self.geometry.compute_control_knots()
            self.geometry.spliner.update_end_time()

    def test_step(self, batch, batch_idx):
        if self.stage != "static" and not batch.__contains__("timestamp"):
            batch.update(
                {
                    "timestamp": torch.as_tensor(
                        [batch["index"] / batch["n_all_views"]], device=self.device
                    )
                }
            )

        # debug
        # T_c2w = convert_pose(batch['c2w'][0].cpu()).numpy()
        # idx = batch['index'].item()
        # pos = T_c2w[:3, 3]
        # rot = T_c2w[:3, :3]
        # serializable_array_2d = [x.tolist() for x in rot]
        # if idx == 0:
        #     self.camera_jsn = []
        # self.camera_jsn.append({"index": idx, "rotation": serializable_array_2d, "position": pos.tolist()})
        # if idx == 119:
        #     with open("output.json", 'w') as f:
        #         json.dump(self.camera_jsn, f)

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
        if self.stage != "static":
            if batch["index"] == 0:
                self.batch_ref_eval = batch

            self.batch_ref_eval["timestamp"] = batch["timestamp"]
            out_ref = self(self.batch_ref_eval)
            self.save_image_grid(
                f"it{self.true_global_step}-test-ref/{batch['index'][0]}.png",
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
                        "img": out_ref["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out_ref["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out_ref
                    else []
                ),
                name=f"test-step-ref",
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
        if self.stage != "static":
            self.save_img_sequence(
                f"it{self.true_global_step}-test-ref",
                f"it{self.true_global_step}-test-ref",
                "(\d+)\.png",
                save_format="mp4",
                fps=30,
                name="test-ref",
                step=self.true_global_step,
            )
        plysavepath = os.path.join(self.get_save_dir(), f"point_cloud_it{self.true_global_step}.ply")
        self.geometry.save_ply(plysavepath)

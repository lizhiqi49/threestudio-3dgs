import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
from PIL import Image
from easydict import EasyDict

import threestudio
import torch
import torch.nn.functional as F

from threestudio.systems.base import BaseLift3DSystem
from threestudio.systems.utils import parse_optimizer, parse_scheduler
from threestudio.utils.loss import tv_loss
from threestudio.utils.ops import get_cam_info_gaussian
from threestudio.utils.typing import *
from threestudio.utils.misc import C
from torch.cuda.amp import autocast
from torchmetrics import PearsonCorrCoef

from pytorch3d.loss import mesh_normal_consistency, mesh_laplacian_smoothing

from ..geometry.gaussian_base import BasicPointCloud, Camera
from ..geometry.dynamic_sugar import DynamicSuGaRModel
from ..utils.arap_utils import ARAPCoach

from torchmetrics import PeakSignalNoiseRatio
import torchvision


@threestudio.register("sugar-4dgen-system")
class SuGaR4DGen(BaseLift3DSystem):
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

        # Intermediate frames
        num_inter_frames: int = 10
        length_inter_frames: float = 0.2

        # === Sparse Gaussians (deformation nodes) optimization == #
        optimize_sparse_gs: bool = True
        geometry_sparse_gs: dict = field(default_factory=dict)

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        self.automatic_optimization = False
        self.stage = self.cfg.stage
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)

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

    # def on_load_checkpoint(self, checkpoint):
    #     num_pts = checkpoint["state_dict"]["geometry._xyz"].shape[0]
    #     pcd = BasicPointCloud(
    #         points=np.zeros((num_pts, 3)),
    #         colors=np.zeros((num_pts, 3)),
    #         normals=np.zeros((num_pts, 3)),
    #     )
    #     self.geometry.create_from_pcd(pcd, self.geometry.cfg.spatial_lr_scale)
    #     self.geometry.training_setup()
    #     # return
    #     super().on_load_checkpoint(checkpoint)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        self.geometry.update_learning_rate(self.global_step)
        outputs = self.renderer.batch_forward(batch)
        return outputs

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # no prompt processor
        self.guidance_zero123 = threestudio.find(self.cfg.guidance_zero123_type)(self.cfg.guidance_zero123)

        # Maybe use 2D diffusion prior
        self.enable_2d_sds = self.cfg.guidance_2d_type is not None and C(self.cfg.loss.lambda_sds_2d, 0, 0) > 0
        if self.enable_2d_sds:
            self.prompt_processor_2d = threestudio.find(self.cfg.prompt_processor_2d_type)(self.cfg.prompt_processor_2d)
            self.guidance_2d = threestudio.find(self.cfg.guidance_2d_type)(self.cfg.guidance_2d)
        else:
            self.prompt_processor_2d = None
            self.guidance_2d = None

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

        # ARAP
        self.arap_coach = None

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

        batch["render_sparse_gs"] = False
        out = self(batch)

        if self.geometry.sparse_gs is not None:
            batch["render_sparse_gs"] = True
            out_sg = self.renderer.batch_forward(batch)
        else:
            out_sg = None

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
            set_loss("rgb", F.mse_loss(gt_rgb, out["comp_rgb"]))
            # mask loss
            set_loss("mask", F.mse_loss(gt_mask.float(), out["comp_mask"]))

            if out_sg is not None:
                set_loss("rgb_sg", F.mse_loss(gt_rgb, out_sg["comp_rgb"]))
                set_loss("mask_sg", F.mse_loss(gt_mask.float(), out_sg["comp_mask"]))

            # ref_psnr = self.psnr(gt_rgb, out["comp_rgb"])
            # self.log(f"metric/PSNR", ref_psnr)

            # depth loss
            if self.C(self.cfg.loss.lambda_depth) > 0:
                valid_gt_depth = batch["ref_depth"][gt_mask.squeeze(-1)].unsqueeze(1)
                # problem with mask and gt depth channel
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
                if out_sg is not None:
                    valid_pred_depth_sg = out_sg["comp_depth"][gt_mask]
                    set_loss(
                        "depth_rel_sg", 1 - self.pearson(valid_pred_depth_sg, valid_gt_depth)
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

            if (
                self.C(self.cfg.loss.lambda_normal_consistency) > 0
                or self.C(self.cfg.loss.lambda_laplacian_smoothing) > 0
            ):
                surface_meshes = self.geometry.get_timed_surface_mesh(
                    timestamp=batch["timestamp"], frame_idx=batch["frame_indices"]
                )
                if self.C(self.cfg.loss.lambda_normal_consistency) > 0:
                    set_loss(
                        "normal_consistency",
                        mesh_normal_consistency(surface_meshes)
                    )
                if self.C(self.cfg.loss.lambda_laplacian_smoothing) > 0:
                    set_loss(
                        "laplacian_smoothing",
                        mesh_laplacian_smoothing(surface_meshes, "uniform")
                    )

        elif guidance == "zero123":
            # zero123
            guidance_out = self.guidance_zero123(
                out["comp_rgb"],
                **batch,
                rgb_as_latents=False,
                guidance_eval=guidance_eval,
            )
            set_loss("sds_zero123", guidance_out["loss_sds"])

            if out_sg is not None:
                guidance_out_sg = self.guidance_zero123(
                    out_sg["comp_rgb"],
                    **batch,
                    rgb_as_latents=False,
                    guidance_eval=guidance_eval,
                )
                set_loss("sds_zero123_sg", guidance_out_sg["loss_sds"])

        # Regularization
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

        if self.C(self.cfg.loss.lambda_normal_depth_consistency) > 0:
            if "comp_normal_from_dist" not in out:
                raise ValueError(
                    "comp_normal_from_dist is required for normal-depth consistency loss!"
                )
            raw_normal = out["comp_normal"] * 2 - 1
            raw_normal_from_dist = out["comp_normal_from_dist"] * 2 - 1
            # loss_normal_depth_consistency = F.mse_loss(raw_normal, raw_normal_from_dist)
            loss_normal_depth_consistency = (1 - raw_normal.unsqueeze(-2) @ raw_normal_from_dist.unsqueeze(-1)).mean()
            set_loss("normal_depth_consistency", loss_normal_depth_consistency)

        if self.stage != "static" and guidance == "ref" and self.C(self.cfg.loss.lambda_ref_xyz) > 0:
            xyz_f0 = self.geometry.get_timed_vertex_xyz(torch.as_tensor(0, dtype=torch.float32, device=self.device))
            loss_ref_xyz = torch.abs(xyz_f0 - self.geometry.get_xyz_verts).mean()
            set_loss("ref_xyz", loss_ref_xyz)

        # object centric reg
        if self.C(self.cfg.loss.lambda_obj_centric) > 0:
            vert_timed_xyz = torch.stack(
                [value for value in self.geometry._deformed_vert_positions.values()],
                dim=0
            )
            loss_obj_centric = (
                torch.abs(vert_timed_xyz[..., 0].mean())
                + torch.abs(vert_timed_xyz[..., 1].mean())
            )
            set_loss("obj_centric", loss_obj_centric)

        if self.stage == "motion":
            # ARAP regularization
            if guidance == "ref" and self.C(
                self.cfg.loss.lambda_arap_reg_key_frame) > 0 and self.arap_coach is not None:
                set_loss(
                    "arap_reg_key_frame",
                    self._compute_arap_energy(batch.get("timestamp"), batch.get("frame_indices"))
                )

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

    def training_substep_inter_frames(self, batch, batch_idx):
        loss_terms = {}
        loss_prefix = "loss_interf_"

        def set_loss(name, value):
            loss_terms[f"{loss_prefix}{name}"] = value

        # Densely sample frames in a smaller time range
        rand_range_start = np.random.rand() * (1 - self.cfg.length_inter_frames)
        rand_timestamps = torch.as_tensor(
            np.linspace(
                rand_range_start,
                rand_range_start + self.cfg.length_inter_frames,
                self.cfg.num_inter_frames,
                endpoint=True
            ),
            dtype=torch.float32,
            device=self.device
        )

        if self.guidance_2d is not None and self.C(self.cfg.loss.lambda_sds_2d) > 0:
            batch_for_2d = {
                "c2w": batch["c2w"][:1].repeat(self.cfg.num_inter_frames, 1, 1),
                "fovy": batch["fovy"][:1].repeat(self.cfg.num_inter_frames, ),
                "elevation": batch["elevation"].repeat(self.cfg.num_inter_frames, ),
                "azimuth": batch["azimuth"].repeat(self.cfg.num_inter_frames, ),
                "camera_distances": batch["camera_distances"].repeat(self.cfg.num_inter_frames, ),
                "timestamp": rand_timestamps,
                "width": batch["width"],
                "height": batch["height"],
                "ambient_ratio": 1.0,
            }
            out_2d = self(batch_for_2d)
            prompt_utils_2d = self.prompt_processor_2d()
            guidance_out = self.guidance_2d(
                out_2d["comp_rgb"],
                prompt_utils_2d,
                **batch_for_2d,
                rgb_as_latents=False,
                # guidance_eval=guidance_eval,
            )
            set_loss("sds_2d", guidance_out["loss_sds"])

        # ARAP regularization
        if self.C(self.cfg.loss.lambda_arap_reg_inter_frame) > 0 and self.arap_coach is not None:
            set_loss("arap_reg_inter_frame", self._compute_arap_energy(rand_timestamps))

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

        self.log(f"train/loss_interf", loss)

        return loss

    def _compute_arap_energy(self, tgt_timestamp=None, tgt_frame_idx=None):
        vert_timed_xyz = self.geometry.get_timed_vertex_xyz(tgt_timestamp, tgt_frame_idx)
        vert_timed_rot = self.geometry.get_timed_vertex_rotation(
            tgt_timestamp, tgt_frame_idx, return_matrix=True)
        loss_arap = 0.
        for i in range(vert_timed_xyz.shape[0]):
            loss_arap += self.arap_coach.compute_arap_energy(
                xyz_prime=vert_timed_xyz[i], vert_rotations=vert_timed_rot[i]
            )
        return loss_arap

    def on_train_batch_start(self, batch, batch_idx, unused=0):
        super().on_train_batch_start(batch, batch_idx, unused)
        if self.geometry.cfg.use_spline:
            self.geometry.compute_control_knots()
            self.geometry.spliner.update_end_time()

        if self.global_step == self.cfg.freq.milestone_arap_reg and self.arap_coach is None:
            self.arap_coach = ARAPCoach(
                self.geometry.get_xyz_verts,
                self.geometry.get_faces.cpu().numpy(),
                self.device
            )

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

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

        if self.cfg.freq.inter_frame_reg > 0 and self.global_step % self.cfg.freq.inter_frame_reg == 0:
            total_loss += self.training_substep_inter_frames(batch, batch_idx)

        self.log("train/loss", total_loss, prog_bar=True)

        if not self.automatic_optimization:
            total_loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

        return {"loss": total_loss}

    def on_validation_epoch_start(self) -> None:
        if self.geometry.cfg.use_spline:
            self.geometry.compute_control_knots()
            self.geometry.spliner.update_end_time()

    def validation_step(self, batch, batch_idx):
        def save_out_to_image_grid(filename, out):
            self.save_image_grid(
                filename,
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
                            "type": "rgb",
                            "img": out["comp_normal_from_dist"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal_from_dist" in out
                    else []
                )
                ,
                # claforte: TODO: don't hardcode the frame numbers to record... read them from cfg instead.
                name=None,
                step=self.true_global_step,
            )

        if self.stage != "static" and not batch.__contains__("timestamp"):
            batch.update(
                {
                    "timestamp": torch.as_tensor(
                        [batch["index"] / batch["n_all_views"]], device=self.device
                    ),
                    "frame_indices": (
                        torch.as_tensor([batch_idx], device=self.device)
                        if self.geometry.num_frames > 1 else
                        torch.as_tensor([0], device=self.device)
                    )
                }
            )
        out = self(batch)
        save_out_to_image_grid(f"it{self.true_global_step}-val/{batch['index'][0]}.png", out)

        if self.geometry.sparse_gs is not None:
            batch.update({"render_sparse_gs": True})
            out_sg = self.renderer.batch_forward(batch)
            save_out_to_image_grid(f"it{self.true_global_step}-val/{batch['index'][0]}_sparse_gs.png", out_sg) 

        if self.stage != "static" and self.geometry.num_frames > 1:
            if batch["index"] == 0:
                self.batch_ref_eval = batch

            self.batch_ref_eval["timestamp"] = batch["timestamp"]
            self.batch_ref_eval["render_sparse_gs"] = False
            out_ref = self(self.batch_ref_eval)
            save_out_to_image_grid(f"it{self.true_global_step}-val-ref/{batch['index'][0]}.png", out_ref)
            if self.geometry.sparse_gs is not None:
                self.batch_ref_eval["render_sparse_gs"] = True
                out_ref_sg = self.renderer.batch_forward(self.batch_ref_eval)
                save_out_to_image_grid(f"it{self.true_global_step}-val-ref/{batch['index'][0]}_sparse_gs.png", out_ref_sg)

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
        if self.geometry.sparse_gs is not None:
            self.save_img_sequence(
                filestem+"-sparse-gs",
                filestem,
                "(\d+)_sparse_gs\.png",
                save_format="mp4",
                fps=30,
                name="validation_epoch_end_gs",
                step=self.true_global_step,
            )
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
            if self.geometry.sparse_gs is not None:
                self.save_img_sequence(
                    filestem+"-sparse-gs",
                    filestem,
                    "(\d+)_sparse_gs\.png",
                    save_format="mp4",
                    fps=30,
                    name="validation_epoch_end_sg-ref",
                    step=self.true_global_step,
                )
            

    def on_test_epoch_start(self) -> None:
        if self.geometry.cfg.use_spline:
            self.geometry.compute_control_knots()
            self.geometry.spliner.update_end_time()

    def test_step(self, batch, batch_idx):
        if self.stage != "static" and not batch.__contains__("timestamp"):
            batch.update(
                {
                    "timestamp": torch.as_tensor(
                        [batch["index"] / batch["n_all_views"]], device=self.device
                    ),
                    "frame_indices": (
                        torch.as_tensor([batch_idx], device=self.device)
                        if self.geometry.num_frames > 1 else
                        torch.as_tensor([0], device=self.device)
                    )
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
            )
            + (
                [
                    {
                        "type": "grayscale",
                        "img": batch["depth"][0],
                        "kwargs": {},
                    }
                ]
                if "depth" in batch
                else []
            )
            + (
                [
                    {
                        "type": "grayscale",
                        "img": out["comp_depth"][0],
                        "kwargs": {},
                    }
                ]
                if "comp_depth" in out
                else []
            )
            ,
            name="test_step",
            step=self.true_global_step,
        )
        if self.stage != "static":
            if batch["index"] == 0:
                self.batch_ref_eval = batch

            self.batch_ref_eval["timestamp"] = batch["timestamp"]
            out_ref = self(self.batch_ref_eval)

            # debug
            # depth = out_ref["comp_depth"][0]
            # depth_array = self.convert_data(depth.reshape(*depth.shape[:2]))
            # depth_array = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min())
            # depth_array = (depth_array * 255.0).astype(np.uint8)
            # pil_img = Image.fromarray(depth_array, mode='L')
            # pil_img.show()

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
                )
                + (
                    [
                        {
                            "type": "grayscale",
                            "img": batch["depth"][0],
                            "kwargs": {},
                        }
                    ]
                    if "depth" in batch
                    else []
                )
                + (
                    [
                        {
                            "type": "grayscale",
                            "img": out_ref["comp_depth"][0],
                            "kwargs": {},
                        }
                    ]
                    if "comp_depth" in out_ref
                    else []
                )
                ,
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
        # plysavepath = os.path.join(self.get_save_dir(), f"point_cloud_it{self.true_global_step}.ply")
        # self.geometry.save_ply(plysavepath)

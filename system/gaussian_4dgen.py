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
from threestudio.utils.ops import get_cam_info_gaussian
from threestudio.utils.typing import *
from threestudio.utils.misc import C
from torch.cuda.amp import autocast
from torchmetrics import PearsonCorrCoef

from ..geometry.gaussian_base import BasicPointCloud, Camera
from ..geometry.spacetime_gaussian import SpacetimeGaussianModel
from ..sugar.sugar_model import SuGaR

from mmcv.ops import knn
from pytorch3d.ops import knn_points
import pypose as pp

def prepare_nn_indices(xyz: Float[Tensor, "N_pts 3"], k=2) -> Int[Tensor, "N_pts k"]:
        """ Prepare the indices of k nearest neighbors for each point """
        xyz_input = xyz.float().cuda()
        xyz_input = xyz_input.unsqueeze(0).contiguous()
        nn_indices = knn(k, xyz_input, xyz_input, False)[0]
        nn_indices: Int[Tensor, "N_pts k"] = nn_indices.transpose(0, 1).long()
        return nn_indices
    
def compute_nn_distances(
    xyz: Float[Tensor, "B N_pts 3"], indices: Int[Tensor, "B N_pts k"]
) -> Float[Tensor, "B N_pts k"]:
    if indices.ndim == 2:
        indices = indices.unsqueeze(0)
    if indices.shape[0] == 1:
        indices = indices.expand(xyz.shape[0], *indices.shape[1:])
    bs, N, k = indices.shape
    xyz_nn = torch.zeros(bs, N, k, 3).to(xyz)
    for i in range(bs):
        xyz_nn[i] = xyz[i, indices[i].flatten(), :].reshape(N, k, 3)

    dists = torch.norm(
        xyz[:, :, None, :].repeat(1, 1, k, 1) - xyz_nn, dim=-1
    )
    return dists


### Attempt to import svd batch method. If not provided, use default method
### Sourced from https://github.com/KinglittleQ/torch-batch-svd/blob/master/torch_batch_svd/include/utils.h
try:
	from torch_batch_svd import svd as batch_svd
except ImportError:
	print("torch_batch_svd not installed. Using torch.svd instead")
	batch_svd = torch.svd


def compute_nn_weights(nn_dists: Float[Tensor, "*B N_pts k"]) -> Float[Tensor, "*B N_pts k"]:
    return F.softmax(nn_dists ** 2, dim=-1)

def compute_arap_energy(
    xyz: Float[Tensor, "N_pts 3"], 
    xyz_prime: Float[Tensor, "N_pts 3"],
    nn_indices: Int[Tensor, "N_pts k"],
    nn_dists: Float[Tensor, "N_pts k"] = None,
    nn_weights: Float[Tensor, "N_pts k"] = None,
) -> Float:
    n_pts, n_neighbors = nn_indices.shape 

    if nn_weights is None:
        if nn_dists is None:
            nn_dists = compute_nn_distances(xyz.unsqueeze(0), nn_indices.unsqueeze(0))[0]
        nn_weights = compute_nn_weights(nn_dists)
    w: Float[Tensor, "N_pts k"] = nn_weights   # softmax of negative squared distance

    edge_mtx: Float[Tensor, "N_pts k 3"] = (
        xyz.unsqueeze(1).repeat(1, n_neighbors, 1) 
        - xyz[nn_indices.flatten()].reshape(n_pts, n_neighbors, 3)
    )
    edge_mtx_prime = (
        xyz_prime.unsqueeze(1).repeat(1, n_neighbors, 1)
        - xyz[nn_indices.flatten()].reshape(n_pts, n_neighbors, 3)
    )

    # Calculate covariance matrix in bulk
    D = torch.diag_embed(w, dim1=1, dim2=2)
    S = torch.bmm(edge_mtx.permute(0, 2, 1), torch.bmm(D, edge_mtx_prime))

    # Calculate rotations
    U, sig, W = batch_svd(S)
    R = torch.bmm(W, U.permute(0, 2, 1))

    # Need to flip the column of U corresponding to smallest singular value
	# for any det(Ri) <= 0
    entries_to_flip = torch.nonzero(torch.det(R) <= 0, as_tuple=False).flatten()  # idxs where det(R) <= 0
    if len(entries_to_flip) > 0:
        Umod = U.clone()
        cols_to_flip = torch.argmin(sig[entries_to_flip], dim=1)  # Get minimum singular value for each entry
        Umod[entries_to_flip, :, cols_to_flip] *= -1  # flip cols
        R[entries_to_flip] = torch.bmm(W[entries_to_flip], Umod[entries_to_flip].permute(0, 2, 1))

    # Compute energy
    rot_rigid = torch.bmm(R, edge_mtx.permute(0, 2, 1)).permute(0, 2, 1)
    stretch_vec = edge_mtx_prime - rot_rigid
    stretch_norm = torch.norm(stretch_vec, dim=2) ** 2
    energy = (w * stretch_norm).sum()

    return energy


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

        # SuGaR
        sugar: dict = field(default_factory=dict)

        # KNN configs
        knn_to_track: int = 10

        # Intermediate frames
        num_inter_frames: int = 10
        length_inter_frames: float = 0.2

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        self.automatic_optimization = False
        self.stage = self.cfg.stage

        self.geometry: SpacetimeGaussianModel
        self.gs_original_xyz = self.geometry._xyz.clone()
        self.gs_original_rot = self.geometry._rotation.clone()

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
        self.geometry.create_from_pcd(pcd, self.geometry.cfg.spatial_lr_scale)
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

        # KNN attributes
        self.knn_to_track = self.cfg.knn_to_track

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
            set_loss("sds_zero123", guidance_out["loss_sds"])

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
        if self.stage == "static" and self.C(self.cfg.loss.lambda_opacity_binary) > 0:
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
        
        if self.stage != "static" and guidance == "ref" and self.C(self.cfg.loss.lambda_ref_gs) > 0:
            xyz_0, _, rot_0, _, _ = self.geometry.get_timed_all(torch.as_tensor(0, dtype=torch.float32, device=self.device))
            # loss_ref_gs = F.mse_loss(
            #     pp.SE3(torch.cat([xyz_0, rot_0], dim=-1)).tensor(),
            #     pp.SE3(torch.cat([self.gs_original_xyz, self.gs_original_rot], dim=-1)).tensor()
            # )
            loss_ref_gs = torch.abs(
                pp.SE3(torch.cat([xyz_0, rot_0], dim=-1)).tensor()
                - pp.SE3(torch.cat([self.gs_original_xyz, self.gs_original_rot], dim=-1)).tensor()
            ).mean()
            set_loss("ref_gs", loss_ref_gs)
 
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
        if self.C(self.cfg.loss.lambda_lite_arap_reg) > 0 or self.C(self.cfg.loss.lambda_full_arap_reg):
            xyz_timed = []
            for t in rand_timestamps:
                xyz_t = self.geometry.get_timed_xyz(t)
                xyz_timed.append(xyz_t)
            xyz_timed = torch.stack(xyz_timed, dim=0)

            # Get the indices of nearest reference timestamps
            ref_timestamps = batch.get("timestamp")
            ref_timestamp_idx = torch.argmin(
                (rand_timestamps[..., None] - ref_timestamps[None, ...]).abs(),
                dim=-1
            )
            nn_idx = self.knn_idx[ref_timestamp_idx]
            if self.C(self.cfg.loss.lambda_lite_arap_reg) > 0:
                nn_dists_timed = compute_nn_distances(xyz_timed, nn_idx)
                nn_dists_ref = self.knn_dists[ref_timestamp_idx]
                loss_arap_reg = (
                    torch.abs(nn_dists_timed[:-1] - nn_dists_timed[1:]).mean()
                    # + torch.abs(nn_dists_timed - nn_dists_ref).mean()
                )
                set_loss("lite_arap_reg", loss_arap_reg)
            if self.C(self.cfg.loss.lambda_full_arap_reg) > 0:
                loss_full_arap = 0.
                # Between sampled frames and key frames
                for i in ref_timestamp_idx:
                    loss_full_arap += compute_arap_energy(
                        self.ref_points[i], xyz_timed[i], self.knn_idx[i], 
                        nn_weights=self.knn_arap_weights[i]
                    )

                # Among key frames
                for i in range(self.ref_points.shape[0] - 1):
                    loss_full_arap += compute_arap_energy(
                        self.ref_points[i], self.ref_points[i+1], self.knn_idx[i],
                        nn_weights=self.knn_arap_weights[i]
                    )
                
                # loss_full_arap = loss_full_arap * 1 / len(ref_timestamp_idx)
                set_loss("full_arap_reg", loss_full_arap)

        # SuGaR density regulation loss
        if self.C(self.cfg.loss.lambda_density_regulation) > 0:
            coarse_args = EasyDict(
                {"current_step": self.true_global_step,
                # "outputs": out,
                "n_samples_for_sdf_regularization": self.cfg.sugar.n_samples_for_sdf_regularization,
                "use_sdf_better_normal_loss": self.cfg.sugar.use_sdf_better_normal_loss,
                "start_sdf_better_normal_from": self.cfg.sugar.start_sdf_better_normal_from,
                "timestamp": rand_timestamps
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

        self.log(f"train/loss_interf", loss)

        return loss
    
    @torch.no_grad()
    def reset_neighbors(self, timestamps=None):
        if timestamps is not None:
            assert isinstance(self.geometry, SpacetimeGaussianModel)
            points = []
            for t in timestamps:
                p = self.geometry.get_timed_xyz(t)
                points.append(p)
            points = torch.stack(points, dim=0)
        else:
            points = self.geometry._xyz.unsqueeze(0)
        
        knns = knn_points(points, points, K=self.knn_to_track)
        self.knn_idx: Int[Tensor, "T N_pts k"] = knns.idx
        self.knn_dists: Float[Tensor, "T N_pts k"] = knns.dists
        self.knn_arap_weights: Float[Tensor, "T N_pts k"] = compute_nn_weights(self.knn_dists)
        self.ref_points = points

        if self.sugar is not None:
            if timestamps is not None:
                time_knn_idx = {
                    "{:.{}f}".format(t, 1): self.knn_idx[i] for i, t in enumerate(timestamps)
                }
                time_knn_dists = {
                    "{:.{}f}".format(t, 1): self.knn_dists[i] for i, t in enumerate(timestamps)
                }
                self.sugar.ref_timestamps = timestamps
            else:
                time_knn_idx = None
                time_knn_dists = None
            self.sugar.knn_idx = self.knn_idx[0]
            self.sugar.knn_dists = self.knn_dists[0]
            self.sugar.time_knn_idx = time_knn_idx
            self.sugar.time_knn_dists = time_knn_dists

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        if self.cfg.sugar.start_regularization_from is not None:
            if self.global_step == self.cfg.sugar.start_regularization_from:
                self.sugar = SuGaR(self.geometry, keep_track_of_knn=True, knn_to_track=self.cfg.knn_to_track)
                # self.sugar.reset_neighbors(timestamp=batch.get('timestamp'))
                self.reset_neighbors(timestamps=batch.get('timestamp'))
            elif self.global_step > self.cfg.sugar.start_regularization_from:
                assert hasattr(self.geometry, "pruned_or_densified")
                # reset neighbors after gaussians densified or settings
                if self.geometry.pruned_or_densified or self.global_step % self.cfg.freq.reset_neighbors == 0:
                    self.geometry.pruned_or_densified = False
                    # self.sugar.reset_neighbors(timestamp=batch.get('timestamp'))
                    self.reset_neighbors(timestamps=batch.get('timestamp'))

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

        if self.global_step > self.cfg.freq.milestone_inter_frame_reg and self.global_step % self.cfg.freq.inter_frame_reg == 0:
            total_loss += self.training_substep_inter_frames(batch, batch_idx)

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

        return {"loss": total_loss}

    def on_validation_epoch_start(self) -> None:
        if self.geometry.cfg.use_spline:
            self.geometry.compute_control_knots()
            self.geometry.spliner.update_end_time()

    def validation_step(self, batch, batch_idx):
        if self.stage != "static" and not batch.__contains__("timestamp"):
            batch.update(
                {
                    "timestamp": torch.as_tensor(
                        [batch["index"] / batch["n_all_views"]], device=self.device
                    ),
                    "frame_idx": torch.as_tensor(batch_idx, device=self.device)
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
                    "frame_idx": torch.as_tensor(batch_idx, device=self.device)
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

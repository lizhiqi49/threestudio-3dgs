import math
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import NamedTuple

import numpy as np
import threestudio
import torch
import torch.nn as nn
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from threestudio.models.geometry.base import BaseGeometry
from threestudio.utils.misc import C
from threestudio.utils.typing import *

import pypose as pp

from .gaussian_base import (
    GaussianBaseModel,
    BasicPointCloud,
)
from .gaussian_base import RGB2SH, inverse_sigmoid, build_rotation, SH2RGB
from .deformation import DeformationNetwork, ModelHiddenParams
from .spline_utils import Spline, SplineConfig


@threestudio.register("spacetime-gaussian-splatting")
class SpacetimeGaussianModel(GaussianBaseModel):
    @dataclass
    class Config(GaussianBaseModel.Config):
        num_frames: int = 14
        use_spline: bool = False

        enable_static: bool = False

        enable_dynamic: bool = False  # delta_xyz, delta_rot
        delta_xyz_lr: float = 0.001
        delta_rot_lr: float = 0.0001

        enable_spacetime: bool = True
        omega_lr: float = 0.01
        trbfc_lr: float = 0.01
        trbfs_lr: float = 0.01
        move_lr: float = 0.01

        rank_motion: int = 3
        rank_omega: int = 1

        addsphpointsscale: float = 0.8
        trbfslinit: float = 0.1
        raystart: float = 0.7
        spatial_lr_scale: float = 10.0

        enable_deformation: bool = False
        # deformation_args: dict = field(default_factory=dict)
        deformation_lr: Optional[Any] = 0
        grid_lr: Optional[Any] = 0

    cfg: Config

    def configure(self) -> None:
        self.pruned_or_densified = False

        self.active_sh_degree = 0
        self.max_sh_degree = self.cfg.sh_degree
        self._xyz = torch.empty(0)

        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        if self.cfg.pred_normal:
            self._normal = torch.empty(0)
        self.optimizer = None
        self.spatial_lr_scale = self.cfg.spatial_lr_scale
        self.setup_functions()

        # Add temporal parameters
        self.num_frames = self.cfg.num_frames

        if self.cfg.use_spline:
            self.init_cubic_spliner()

        if self.cfg.enable_dynamic:
            self._delta_xyz = torch.empty(0)
            self._delta_rot = torch.empty(0)

        if self.cfg.enable_spacetime:
            self._motion = torch.empty(0)
            self._omega = torch.empty(0)

            self.delta_t = None
            self.omegamask = None
            self.maskforems = None
            self.distancetocamera = None
            self.trbfslinit = self.cfg.trbfslinit
            self.ts = None
            self.trbfoutput = None
            self.preprocesspoints = False
            self.addsphpointsscale = self.cfg.addsphpointsscale

            self.maxz, self.minz = 0.0, 0.0
            self.maxy, self.miny = 0.0, 0.0
            self.maxx, self.minx = 0.0, 0.0
            self.computedtrbfscale = None
            self.computedopacity = None
            self.raystart = self.cfg.raystart

        # Maybe use deformation
        if self.cfg.enable_deformation:
            deformation_args = ModelHiddenParams(None)
            self._deformation = DeformationNetwork(deformation_args)
            self._deformation_table = torch.empty(0)
        else:
            self._deformation = None
            self._deformation_table = None

        if self.cfg.geometry_convert_from.startswith("shap-e:"):
            shap_e_guidance = threestudio.find("shap-e-guidance")(
                self.cfg.shap_e_guidance_config
            )
            prompt = self.cfg.geometry_convert_from[len("shap-e:"):]
            xyz, color = shap_e_guidance(prompt)

            pcd = BasicPointCloud(
                points=xyz, colors=color, normals=np.zeros((xyz.shape[0], 3))
            )
            self.create_from_pcd(pcd, self.cfg.spatial_lr_scale)
            self.training_setup()

        # Support Initialization from OpenLRM, Please see https://github.com/Adamdad/threestudio-lrm
        elif self.cfg.geometry_convert_from.startswith("lrm:"):
            lrm_guidance = threestudio.find("lrm-guidance")(
                self.cfg.shap_e_guidance_config
            )
            prompt = self.cfg.geometry_convert_from[len("lrm:"):]
            xyz, color = lrm_guidance(prompt)

            pcd = BasicPointCloud(
                points=xyz, colors=color, normals=np.zeros((xyz.shape[0], 3))
            )
            self.create_from_pcd(pcd, self.cfg.spatial_lr_scale)
            self.training_setup()

        elif os.path.exists(self.cfg.geometry_convert_from):
            threestudio.info(
                "Loading point cloud from %s" % self.cfg.geometry_convert_from
            )
            if self.cfg.geometry_convert_from.endswith(".ckpt"):
                ckpt_dict = torch.load(self.cfg.geometry_convert_from)
                num_pts = ckpt_dict["state_dict"]["geometry._xyz"].shape[0]
                pcd = BasicPointCloud(
                    points=np.zeros((num_pts, 3)),
                    colors=np.zeros((num_pts, 3)),
                    normals=np.zeros((num_pts, 3)),
                )
                self.create_from_pcd(pcd, self.cfg.spatial_lr_scale)
                self.training_setup()
                new_ckpt_dict = {}
                for key in self.state_dict():
                    if ckpt_dict["state_dict"].__contains__("geometry." + key):
                        new_ckpt_dict[key] = ckpt_dict["state_dict"]["geometry." + key]
                    else:
                        new_ckpt_dict[key] = self.state_dict()[key]
                self.load_state_dict(new_ckpt_dict)
            elif self.cfg.geometry_convert_from.endswith(".ply"):
                if self.cfg.load_ply_only_vertex:
                    plydata = PlyData.read(self.cfg.geometry_convert_from)
                    vertices = plydata["vertex"]
                    positions = np.vstack(
                        [vertices["x"], vertices["y"], vertices["z"]]
                    ).T
                    if vertices.__contains__("red"):
                        colors = (
                            np.vstack(
                                [vertices["red"], vertices["green"], vertices["blue"]]
                            ).T
                            / 255.0
                        )
                    else:
                        shs = np.random.random((positions.shape[0], 3)) / 255.0
                        C0 = 0.28209479177387814
                        colors = shs * C0 + 0.5
                    normals = np.zeros_like(positions)
                    pcd = BasicPointCloud(
                        points=positions, colors=colors, normals=normals
                    )
                    self.create_from_pcd(pcd, self.cfg.spatial_lr_scale)
                else:
                    self.load_ply(self.cfg.geometry_convert_from)
                self.training_setup()
        else:
            threestudio.info("Geometry not found, initilization with random points")
            num_pts = self.cfg.init_num_pts
            phis = np.random.random((num_pts,)) * 2 * np.pi
            costheta = np.random.random((num_pts,)) * 2 - 1
            thetas = np.arccos(costheta)
            mu = np.random.random((num_pts,))
            radius = self.cfg.pc_init_radius * np.cbrt(mu)
            x = radius * np.sin(thetas) * np.cos(phis)
            y = radius * np.sin(thetas) * np.sin(phis)
            z = radius * np.cos(thetas)
            xyz = np.stack((x, y, z), axis=1)

            shs = np.random.random((num_pts, 3)) / 255.0
            C0 = 0.28209479177387814
            color = shs * C0 + 0.5
            pcd = BasicPointCloud(
                points=xyz, colors=color, normals=np.zeros((num_pts, 3))
            )

            self.create_from_pcd(pcd, 10)
            self.training_setup()

    def get_motion(self, delta_t, frame_idx):
        motion = torch.zeros(self._xyz.shape, dtype=self._xyz.dtype, device=self.device)
        if self.cfg.enable_spacetime:
            motion_st = self._motion.reshape(-1, self.cfg.rank_motion, 3)
            for i in range(self.cfg.rank_motion):
                motion += motion_st[:, i, :] * delta_t ** (i + 1)
        if self.cfg.enable_dynamic:
            motion_dynamic = self._delta_xyz[frame_idx]
            motion += motion_dynamic
        return motion

    def get_omega(self, delta_t, frame_idx):
        omega = torch.zeros(self._rotation.shape, dtype=self._xyz.dtype, device=self.device)
        if self.cfg.enable_spacetime:
            omega_st = self._omega.reshape(-1, self.cfg.rank_omega, 4)
            for i in range(self.cfg.rank_omega):
                omega += omega_st[:, i, :] * delta_t ** (i + 1)
        if self.cfg.enable_dynamic:
            omega_dynamic = self._delta_rot[frame_idx]
            omega += omega_dynamic
        return omega

    @property
    def get_rotations(self):
        return self._rotation

    @property
    def get_trbfcenter(self):
        return self._trbf_center

    @property
    def get_trbfscale(self):
        return self._trbf_scale

    @property
    def get_features(self):
        return SH2RGB(self._features_dc)

    def _get_timed_xyz_and_rot(self, timestamp, frame_idx):
        means3D = self.get_xyz
        rotations = self._rotation
        opacity = self._opacity
        scales = self._scaling
        n_points = means3D.shape[0]
        timestamp = timestamp.expand(n_points, 1)

        if self.cfg.enable_deformation:
            means3D, scales, rotations, opacity = self._deformation(
                means3D, scales, rotations, opacity, timestamp * 2 - 1
            )

        if self.cfg.enable_spacetime:
            trbfcenter = self.get_trbfcenter
            trbfdistanceoffset = timestamp - trbfcenter
            tforpoly = trbfdistanceoffset.detach()
            delta_t = tforpoly
            # trbfdistance =  trbfdistanceoffset / torch.exp(trbfscale)
            # trbfoutput = torch.exp(-1 * trbfdistance.pow(2))
            # # self.trbfoutput = trbfoutput

            # opacity = opacity * trbfoutput
        else:
            delta_t = None

        rotations = rotations + self.get_omega(delta_t, frame_idx)
        means3D = means3D + self.get_motion(delta_t, frame_idx)
        return means3D, rotations

    def get_timed_all(self, timestamp, frame_idx=None):
        means3D = self.get_xyz
        opacity = self._opacity
        scales = self._scaling
        rotations = self._rotation

        n_points = means3D.shape[0]
        timestamp = timestamp.expand(n_points, 1)

        if self.cfg.use_spline:
            means3D, rotations = self._spline_interp(timestamp[:, 0])
        else:
            if self.cfg.enable_deformation:
                means3D, scales, rotations, opacity = self._deformation(
                    means3D, scales, rotations, opacity, timestamp * 2 - 1
                )

            if self.cfg.enable_spacetime:
                trbfcenter = self.get_trbfcenter
                trbfscale = self.get_trbfscale
                trbfdistanceoffset = timestamp - trbfcenter
                tforpoly = trbfdistanceoffset.detach()
                delta_t = tforpoly
                # trbfdistance =  trbfdistanceoffset / torch.exp(trbfscale)
                # trbfoutput = torch.exp(-1 * trbfdistance.pow(2))
                # # self.trbfoutput = trbfoutput

                # opacity = opacity * trbfoutput
            else:
                delta_t = None

            rotations = rotations + self.get_omega(delta_t, frame_idx)
            means3D = means3D + self.get_motion(delta_t, frame_idx)

        colors_precomp = self.get_features.reshape(n_points, 3)
        opacity = self.opacity_activation(opacity)
        scales = self.scaling_activation(scales)
        rotations = self.rotation_activation(rotations)

        return means3D, scales, rotations, opacity, colors_precomp

    def get_timed_xyz(self, timestamp, frame_idx=None):
        means3D = self.get_xyz
        opacity = self._opacity
        scales = self._scaling
        rotations = self._rotation

        n_points = means3D.shape[0]
        timestamp = timestamp.expand(n_points, 1)
        if self.cfg.use_spline:
            means3D, _ = self._spline_interp(timestamp[:, 0])
        else:
            if self.cfg.enable_deformation:
                means3D, _, _, _ = self._deformation(
                    means3D, scales, rotations, opacity, timestamp * 2 - 1
                )
            if self.cfg.enable_spacetime:
                trbfcenter = self.get_trbfcenter
                trbfscale = self.get_trbfscale
                trbfdistanceoffset = timestamp - trbfcenter
                tforpoly = trbfdistanceoffset.detach()
                delta_t = tforpoly
            else:
                delta_t = None
            means3D = means3D + self.get_motion(delta_t, frame_idx)

        return means3D

    def init_cubic_spliner(self):
        n_ctrl_knots = self.num_frames
        t_interv = torch.as_tensor(1 / (n_ctrl_knots - 3)).cuda()   # exclude start and end point
        spline_cfg = SplineConfig(degree=3, sampling_interval=t_interv)
        self.spliner = Spline(spline_cfg)
        self.spliner.data = pp.identity_SE3(1, self.num_frames)
        self.spliner.set_start_time(-t_interv)
        self.spliner.update_end_time()

    def compute_control_knots(self):
        ctrl_knots_xyz = []
        ctrl_knots_rot = []
        ts = torch.as_tensor(
            np.linspace(
                self.spliner.start_time.cpu().numpy(), 
                self.spliner.end_time.cpu().numpy(), 
                self.num_frames, 
                endpoint=True
            ),
            dtype=torch.float32,
            device="cuda"
        )
        for i, t in enumerate(ts):
            xyz, rot = self._get_timed_xyz_and_rot(t, i)
            ctrl_knots_xyz.append(xyz)
            ctrl_knots_rot.append(rot)
        ctrl_knots_xyz = torch.stack(ctrl_knots_xyz, dim=0)
        ctrl_knots_rot = torch.stack(ctrl_knots_rot, dim=0)

        self.spliner.data: Float[pp.LieTensor, "N_pts, N_ctrlknots, 7"]
        self.spliner.data = pp.SE3(
            torch.cat([ctrl_knots_xyz, ctrl_knots_rot], dim=-1).permute(1, 0, 2)
        )

    def _spline_interp(self, timestamp):
        ret = self.spliner(timestamp)
        xyz = ret.translation()
        rot = ret.rotation().tensor()
        return xyz, rot

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        # TODO: whether use RGB2SH here?
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        # fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()   # RGB
        # features = (
        #     torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
        #     .float()
        #     .cuda()
        # )
        # features[:, :3, 0] = fused_color
        # features[:, 3:, 1:] = 0.0

        threestudio.info(
            f"Number of points at initialisation:{fused_point_cloud.shape[0]}"
        )

        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
            0.0000001,
        )
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(
            self.cfg.opacity_init
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(fused_color.unsqueeze(1).contiguous().requires_grad_(True))

        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")

        self.fused_point_cloud = fused_point_cloud.cpu().clone().detach()
        # self.features = features.cpu().clone().detach()
        self.scales = scales.cpu().clone().detach()
        self.rots = rots.cpu().clone().detach()
        self.opacities = opacities.cpu().clone().detach()

        if self.cfg.pred_normal:
            normals = torch.zeros((fused_point_cloud.shape[0], 3), device="cuda")
            self._normal = nn.Parameter(normals.requires_grad_(True))

        if self.cfg.enable_deformation:
            self._deformation = self._deformation.to("cuda")
            self._deformation_table = torch.gt(torch.ones((self.get_xyz.shape[0]), device="cuda"), 0)

        if self.cfg.enable_dynamic:
            delta_xyz = torch.zeros((self.num_frames, *self._xyz.shape), device="cuda")
            self._delta_xyz = nn.Parameter(delta_xyz.requires_grad_(True))
            delta_rot = torch.zeros((self.num_frames, *self._rotation.shape), device="cuda")
            self._delta_rot = nn.Parameter(delta_rot.requires_grad_(True))

        if self.cfg.enable_spacetime:
            omega = torch.zeros(
                (fused_point_cloud.shape[0], 4 * self.cfg.rank_omega),
                device="cuda"
            )
            self._omega = nn.Parameter(omega.requires_grad_(True))

            motion = torch.zeros(
                (fused_point_cloud.shape[0], 3 * self.cfg.rank_motion),
                device="cuda"
            )  # x1, x2, x3,  y1,y2,y3, z1,z2,z3
            self._motion = nn.Parameter(motion.requires_grad_(True))

            times = torch.zeros((fused_point_cloud.shape[0], 1), device="cuda")
            self._trbf_center = nn.Parameter(times.contiguous().requires_grad_(True))
            self._trbf_scale = nn.Parameter(torch.ones((self.get_xyz.shape[0], 1), device="cuda").requires_grad_(True))

            if self.trbfslinit is not None:
                nn.init.constant_(self._trbf_scale, self.trbfslinit)  # too large ?
            else:
                nn.init.constant_(self._trbf_scale, 0)  # too large ?

            nn.init.constant_(self._omega, 0)
            self.rgb_grd = {}

            self.maxz, self.minz = torch.amax(self._xyz[:, 2]), torch.amin(self._xyz[:, 2])
            self.maxy, self.miny = torch.amax(self._xyz[:, 1]), torch.amin(self._xyz[:, 1])
            self.maxx, self.minx = torch.amax(self._xyz[:, 0]), torch.amin(self._xyz[:, 0])
            self.maxz = min((self.maxz, 200.0))  # some outliers in the n4d datasets..

        if self.cfg.use_spline:
            self.init_cubic_spliner()

    def training_setup(self):
        training_args = self.cfg
        # self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        if self.cfg.enable_deformation:
            self._deformation_accum = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")

        position_lr_init = C(training_args.position_lr, 0, 0) * self.spatial_lr_scale
        l = []
        if self.cfg.enable_static:
            l += [
                {
                    "params": [self._xyz],
                    "lr": position_lr_init,
                    "name": "xyz",
                },
                {
                    "params": [self._features_dc],
                    "lr": C(training_args.feature_lr, 0, 0),
                    "name": "f_dc",
                },
                {
                    "params": [self._opacity],
                    "lr": C(training_args.opacity_lr, 0, 0),
                    "name": "opacity",
                },
                {
                    "params": [self._scaling],
                    "lr": C(training_args.scaling_lr, 0, 0),
                    "name": "scaling",
                },
                {
                    "params": [self._rotation],
                    "lr": C(training_args.rotation_lr, 0, 0),
                    "name": "rotation",
                },
            ]

        if self.cfg.enable_dynamic:
            l += [
                {
                    "params": [self._delta_xyz],
                    "lr": C(training_args.delta_xyz_lr, 0, 0),
                    "name": "delta_xyz",
                },
                {
                    "params": [self._delta_rot],
                    "lr": C(training_args.delta_rot_lr, 0, 0),
                    "name": "delta_rot",
                }
            ]

        if self.cfg.enable_spacetime:
            l += [
                {
                    "params": [self._omega],
                    "lr": C(training_args.omega_lr, 0, 0),
                    "name": "omega",
                },
                {
                    "params": [self._trbf_center],
                    "lr": C(training_args.trbfc_lr, 0, 0),
                    "name": "trbf_center",
                },
                {
                    "params": [self._trbf_scale],
                    "lr": C(training_args.trbfs_lr, 0, 0),
                    "name": "trbf_scale",
                },
                {
                    "params": [self._motion],
                    "lr": position_lr_init * 0.5 * training_args.move_lr,
                    "name": "motion"
                }
            ]
        if self.cfg.pred_normal:
            l.append(
                {
                    "params": [self._normal],
                    "lr": C(training_args.normal_lr, 0, 0),
                    "name": "normal",
                },
            )
        if self.cfg.enable_deformation:
            l += [
                {
                    'params': list(self._deformation.get_mlp_parameters()),
                    'lr': C(training_args.deformation_lr, 0, 0) * self.spatial_lr_scale,
                    "name": "deformation"
                },
                {
                    'params': list(self._deformation.get_grid_parameters()),
                    'lr': C(training_args.grid_lr, 0, 0) * self.spatial_lr_scale,
                    "name": "grid"
                },
            ]

        self.optimize_list = l
        self.optimize_params = [d["name"] for d in l]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if not ("name" in param_group):
                continue

            if self.cfg.enable_static:
                if param_group["name"] == "xyz":
                    param_group["lr"] = C(
                        self.cfg.position_lr, 0, iteration, interpolation="exp"
                    ) * self.spatial_lr_scale
                if param_group["name"] == "scaling":
                    param_group["lr"] = C(
                        self.cfg.scaling_lr, 0, iteration, interpolation="exp"
                    )
                if param_group["name"] == "f_dc":
                    param_group["lr"] = C(
                        self.cfg.feature_lr, 0, iteration, interpolation="exp"
                    )
                if param_group["name"] == "opacity":
                    param_group["lr"] = C(
                        self.cfg.opacity_lr, 0, iteration, interpolation="exp"
                    )
                if param_group["name"] == "rotation":
                    param_group["lr"] = C(
                        self.cfg.rotation_lr, 0, iteration, interpolation="exp"
                    )
                if param_group["name"] == "normal":
                    param_group["lr"] = C(
                        self.cfg.normal_lr, 0, iteration, interpolation="exp"
                    )

            if self.cfg.enable_dynamic:
                if param_group["name"] == "delta_xyz":
                    param_group["lr"] = C(
                        self.cfg.delta_xyz_lr, 0, iteration, interpolation="exp"
                    ) * self.spatial_lr_scale
                if param_group["name"] == "delta_rot":
                    param_group["lr"] = C(
                        self.cfg.delta_rot_lr, 0, iteration, interpolation="exp"
                    )

            if self.cfg.enable_spacetime:
                pass  # Following the original paper to use constant lr

            if self.cfg.enable_deformation:
                if "grid" in param_group["name"]:
                    param_group["lr"] = C(
                        self.cfg.grid_lr, 0, iteration, interpolation="exp"
                    ) * self.spatial_lr_scale
                elif param_group["name"] == "deformation":
                    param_group["lr"] = C(
                        self.cfg.deformation_lr, 0, iteration, interpolation="exp"
                    ) * self.spatial_lr_scale

        self.color_clip = C(self.cfg.color_clip, 0, iteration)

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        # self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.cfg.pred_normal:
            self._normal = optimizable_tensors["normal"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        if self.cfg.enable_dynamic:
            self._delta_xyz = optimizable_tensors["delta_xyz"]
            self._detla_rot = optimizable_tensors["delta_rot"]

        if self.cfg.enable_spacetime:
            self._trbf_center = optimizable_tensors["trbf_center"]
            self._trbf_scale = optimizable_tensors["trbf_scale"]
            self._motion = optimizable_tensors["motion"]
            self._omega = optimizable_tensors["omega"]
            if self.omegamask is not None:
                self.omegamask = self.omegamask[valid_points_mask]

        if self.cfg.enable_deformation:
            self._deformation_accum = self._deformation_accum[valid_points_mask]
            self._deformation_table = self._deformation_table[valid_points_mask]

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        # new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_delta_xyz=None,
        new_delta_rot=None,
        new_trbf_center=None,
        new_trbf_scale=None,
        new_motion=None,
        new_omega=None,
        new_deformation_table=None,
        new_normal=None,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            # "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }
        if self.cfg.pred_normal:
            d.update({"normal": new_normal})

        if self.cfg.enable_dynamic:
            d.update({
                "delta_xyz": new_delta_xyz,
                "delta_rot": new_delta_rot,
            })

        if self.cfg.enable_spacetime:
            d.update({
                "trbf_center": new_trbf_center,
                "trbf_scale": new_trbf_scale,
                "motion": new_motion,
                "omega": new_omega,
            })

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        # self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        if self.cfg.enable_dynamic:
            self._delta_xyz = optimizable_tensors["delta_xyz"]
            self._detla_rot = optimizable_tensors["delta_rot"]

        if self.cfg.enable_spacetime:
            self._trbf_center = optimizable_tensors["trbf_center"]
            self._trbf_scale = optimizable_tensors["trbf_scale"]
            self._motion = optimizable_tensors["motion"]
            self._omega = optimizable_tensors["omega"]

        if self.cfg.pred_normal:
            self._normal = optimizable_tensors["normal"]

        if self.cfg.enable_deformation:
            self._deformation_table = torch.cat([self._deformation_table, new_deformation_table], -1)
            self._deformation_accum = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")

        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, N=2):
        n_init_points = self._xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.norm(self.get_scaling, dim=1) > self.cfg.split_thresh,
        )

        # divide N to enhance robustness
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1) / N
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self._xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        # new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        postfix_inputs = (
            new_xyz, new_features_dc, new_opacity, new_scaling, new_rotation,
        )

        if self.cfg.enable_dynamic:
            new_delta_xyz = self._delta_xyz[selected_pts_mask].repeat(1, N, 1)
            new_delta_rot = self._delta_xyz[selected_pts_mask].repeat(1, N, 1)
            postfix_inputs += (new_delta_xyz, new_delta_rot,)

        if self.cfg.enable_spacetime:
            new_trbf_center = self._trbf_center[selected_pts_mask].repeat(N, 1)
            new_trbf_center = torch.rand_like(new_trbf_center)  # * 0.5
            new_trbf_scale = self._trbf_scale[selected_pts_mask].repeat(N, 1)
            new_motion = self._motion[selected_pts_mask].repeat(N, 1)
            new_omega = self._omega[selected_pts_mask].repeat(N, 1)
            postfix_inputs += (
                new_trbf_center, new_trbf_scale, new_motion, new_omega,
            )

        if self.cfg.enable_deformation:
            new_deformation_table = self._deformation_table[selected_pts_mask].repeat(N)
            postfix_inputs += (new_deformation_table,)

        if self.cfg.pred_normal:
            new_normal = self._normal[selected_pts_mask].repeat(N, 1)
            postfix_inputs += (new_normal,)

        self.densification_postfix(*postfix_inputs)

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.norm(self.get_scaling, dim=1) <= self.cfg.split_thresh,
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        # new_features_rest = self._features_rest[selected_pts_mask]
        new_opacity = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        postfix_inputs = (
            new_xyz, new_features_dc, new_opacity, new_scaling, new_rotation,
        )

        if self.cfg.enable_dynamic:
            new_delta_xyz = self._delta_xyz[selected_pts_mask]
            new_delta_rot = self._delta_xyz[selected_pts_mask]
            postfix_inputs += (new_delta_xyz, new_delta_rot,)

        if self.cfg.enable_spacetime:
            new_trbf_center = torch.rand((self._trbf_center[selected_pts_mask].shape[0], 1),
                                         device="cuda")  # self._trbf_center[selected_pts_mask]
            new_trbf_scale = self._trbf_scale[selected_pts_mask]
            new_motion = self._motion[selected_pts_mask]
            new_omega = self._omega[selected_pts_mask]
            postfix_inputs += (
                new_trbf_center, new_trbf_scale, new_motion, new_omega,
            )

        if self.cfg.enable_deformation:
            new_deformation_table = self._deformation_table[selected_pts_mask]
            postfix_inputs += (new_deformation_table,)

        if self.cfg.pred_normal:
            new_normal = self._normal[selected_pts_mask]
            postfix_inputs += (new_normal,)

        self.densification_postfix(*postfix_inputs)

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']

        for i in range(self._features_dc.shape[-1]):
            l.append('f_dc_{}'.format(i))
        # for i in range(self._features_rest.shape[1]):
        #     l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))

        if self.cfg.enable_dynamic:
            # delta_xyz
            for i in range(self.num_frames):
                l += [f"delta_{p}_t{i}" for p in "xyz"]
            # delta_rot
            for i in range(self.num_frames):
                l += [f"delta_rot_{j}_t{i}" for j in range(self._delta_rot.shape[-1])]

        if self.cfg.enable_spacetime:
            l.append('trbf_center')
            l.append('trbf_scale')
            for i in range(self._motion.shape[1]):
                l.append('motion_{}'.format(i))
            for i in range(self._omega.shape[1]):
                l.append('omega_{}'.format(i))

        return l

    def save_ply(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.squeeze(1).detach().cpu().numpy()
        # f_rest = (
        #     self._features_rest.detach()
        #     .transpose(1, 2)
        #     .flatten(start_dim=1)
        #     .contiguous()
        #     .cpu()
        #     .numpy()
        # )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        attributes = (xyz, normals, f_dc, opacities, scale, rotation,)

        if self.cfg.enable_dynamic:
            delta_xyz = self._delta_xyz.detach().cpu().numpy()
            delta_rot = self._delta_rot.detach().cpu().numpy()
            for i in range(self.num_frames):
                attributes += (delta_xyz[i],)
            for i in range(self.num_frames):
                attributes += (delta_rot[i],)

        if self.cfg.enable_spacetime:
            trbf_center = self._trbf_center.detach().cpu().numpy()
            trbf_scale = self._trbf_scale.detach().cpu().numpy()
            motion = self._motion.detach().cpu().numpy()
            omega = self._omega.detach().cpu().numpy()
            attributes += (trbf_center, trbf_scale, motion, omega,)

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(attributes, axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    # Rewrite ply load function to take in temporal parameters
    def load_ply(self, path):
        plydata = PlyData.read(path)
        ply_element = plydata.elements[0]

        def maybe_load_from_ply(plye, key, init_func, shape=None):
            if plye.__contains__(key):
                value = np.asarray(plye[key])
                if shape is not None:
                    value = value.reshape(shape)
            else:
                assert shape is not None
                value = init_func(shape)
            return value

        xyz = np.stack(
            (
                np.asarray(ply_element["x"]),
                np.asarray(ply_element["y"]),
                np.asarray(ply_element["z"]),
            ),
            axis=1,
        )
        n_points = xyz.shape[0]
        opacities = np.asarray(ply_element["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3))
        features_dc[:, 0] = np.asarray(ply_element["f_dc_0"])
        features_dc[:, 1] = np.asarray(ply_element["f_dc_1"])
        features_dc[:, 2] = np.asarray(ply_element["f_dc_2"])
        # features_dc = SH2RGB(features_dc)

        scale_names = [
            p.name
            for p in ply_element.properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(ply_element[attr_name])

        rot_names = [
            p.name for p in ply_element.properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(ply_element[attr_name])

        if self.max_sh_degree > 0:
            extra_f_names = [
                p.name
                for p in ply_element.properties
                if p.name.startswith("f_rest_")
            ]
            extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
            assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(ply_element[attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            features_extra = features_extra.reshape(
                (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
            )

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .unsqueeze(1)
            .contiguous()
            .requires_grad_(True)
        )
        # if self.max_sh_degree > 0:
        #     self._features_rest = nn.Parameter(
        #         torch.tensor(features_extra, dtype=torch.float, device="cuda")
        #         .transpose(1, 2)
        #         .contiguous()
        #         .requires_grad_(True)
        #     )
        # else:
        #     self._features_rest = nn.Parameter(
        #         torch.tensor(features_dc, dtype=torch.float, device="cuda")[:, :, 1:]
        #         .transpose(1, 2)
        #         .contiguous()
        #         .requires_grad_(True)
        #     )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )

        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.active_sh_degree = self.max_sh_degree

        self.computedopacity = self.opacity_activation(self._opacity)
        self.computedscales = torch.exp(self._scaling)  # change not very large

        if self.cfg.enable_dynamic:
            delta_xyz = np.zeros((self.num_frames, n_points, 3))
            delta_rot = np.zeros((self.num_frames, n_points, self._rotation.shape[-1]))
            for i in range(self.num_frames):
                delta_xyz[i, :, 0] = maybe_load_from_ply(
                    ply_element, f"delta_x_t{i}", np.zeros, (n_points,))
                delta_xyz[i, :, 1] = maybe_load_from_ply(
                    ply_element, f"delta_y_t{i}", np.zeros, (n_points,))
                delta_xyz[i, :, 2] = maybe_load_from_ply(
                    ply_element, f"delta_z_t{i}", np.zeros, (n_points,))
                for j in range(delta_rot.shape[-1]):
                    delta_rot[i, :, j] = maybe_load_from_ply(
                        ply_element, f"delta_rot_{j}_t{i}", np.zeros, (n_points,)
                    )
            self._delta_xyz = nn.Parameter(
                torch.tensor(delta_xyz, dtype=torch.float, device="cuda").requires_grad_(True)
            )
            self._delta_rot = nn.Parameter(
                torch.tensor(delta_rot, dtype=torch.float, device="cuda").requires_grad_(True)
            )

        if self.cfg.enable_spacetime:
            trbf_center = maybe_load_from_ply(ply_element, "trbf_center", np.zeros, (n_points, 1))
            trbf_scale = maybe_load_from_ply(ply_element, "trbf_scale", np.ones, (n_points, 1))

            n_motion = 3 * self.cfg.rank_motion
            motion = np.zeros((n_points, n_motion))
            for i in range(n_motion):
                motion[:, i] = maybe_load_from_ply(ply_element, "motion_" + str(i), np.zeros, (n_points,))

            n_omega = 4 * self.cfg.rank_omega
            omegas = np.zeros((n_points, n_omega))
            for i in range(n_omega):
                omegas[:, i] = maybe_load_from_ply(ply_element, "omega_" + str(i), np.zeros, (n_points,))

            self._trbf_center = nn.Parameter(
                torch.tensor(trbf_center, dtype=torch.float, device="cuda").requires_grad_(True)
            )
            self._trbf_scale = nn.Parameter(
                torch.tensor(trbf_scale, dtype=torch.float, device="cuda").requires_grad_(True)
            )
            self._motion = nn.Parameter(
                torch.tensor(motion, dtype=torch.float, device="cuda").requires_grad_(True)
            )
            self._omega = nn.Parameter(
                torch.tensor(omegas, dtype=torch.float, device="cuda").requires_grad_(True)
            )
            self.computedtrbfscale = torch.exp(self._trbf_scale)  # precomputed

        if self.cfg.enable_deformation:
            self._deformation = self._deformation.to("cuda")
            self._deformation_table = torch.gt(torch.ones((self.get_xyz.shape[0]), device="cuda"),
                                               0)  # everything deformed

        if self.cfg.use_spline:
            self.init_cubic_spliner()

    def load_deformation_model(self, path, name):
        print("loading model from exists{}".format(path))
        weight_dict = torch.load(os.path.join(path, name + "_deformation.pth"), map_location="cuda")
        self._deformation.load_state_dict(weight_dict)
        self._deformation = self._deformation.to("cuda")
        self._deformation_table = torch.gt(torch.ones((self.get_xyz.shape[0]), device="cuda"), 0)
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        if os.path.exists(os.path.join(path, name + "_deformation_table.pth")):
            self._deformation_table = torch.load(os.path.join(path, name + "_deformation_table.pth"),
                                                 map_location="cuda")
        if os.path.exists(os.path.join(path, name + "_deformation_accum.pth")):
            self._deformation_accum = torch.load(os.path.join(path, name + "_deformation_accum.pth"),
                                                 map_location="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def save_deformation_model(self, path, name):
        torch.save(self._deformation.state_dict(), os.path.join(path, name + "_deformation.pth"))
        torch.save(self._deformation_table, os.path.join(path, name + "_deformation_table.pth"))
        torch.save(self._deformation_accum, os.path.join(path, name + "_deformation_accum.pth"))

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        super().update_step(epoch, global_step, on_load_weights)
        if self.cfg.use_spline and self.training:
            self.compute_control_knots()
            # self.spliner.update_end_time()

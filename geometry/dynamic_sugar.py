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
import open3d as o3d
from pytorch3d.ops import knn_points
from pytorch3d.transforms import quaternion_apply, quaternion_invert, matrix_to_quaternion
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesUV, TexturesVertex

from .gaussian_base import SH2RGB, RGB2SH
from .sugar import SuGaRModel
from .deformation import DeformationNetwork, ModelHiddenParams
from .spline_utils import Spline, SplineConfig


@threestudio.register("dynamic-sugar")
class DynamicSuGaRModel(SuGaRModel):
    @dataclass
    class Config(SuGaRModel.Config):
        num_frames: int = 14
        static_learnable: bool = False
        use_spline: bool = False

        dynamic_mode: str = "discrete"  # 'discrete', 'deformation'
        delta_xyz_lr: Any = 0.001

        deformation_lr: Any = 0.001
        grid_lr: Any = 0.001

        d_xyz: bool = True
        d_rotation: bool = False
        d_opacity: bool = False
        d_scale: bool = False


    cfg: Config

    def configure(self) -> None:
        super().configure()
        if not self.cfg.static_learnable:
            self._points.requires_grad_(False)
            self._quaternions.requires_grad_(False)
            self.all_densities.requires_grad_(False)
            self._sh_coordinates_dc.requires_grad_(False)
            self._sh_coordinates_rest.requires_grad_(False)
            self._scales.requires_grad_(False)
            self._quaternions.requires_grad_(False)
            self.surface_mesh_thickness.requires_grad_(False)

        self.num_frames = self.cfg.num_frames
        self.dynamic_mode = self.cfg.dynamic_mode

        if self.cfg.use_spline:
            self.init_cubic_spliner()

        if self.dynamic_mode == "discrete":
            self._delta_xyz = nn.Parameter(
                torch.zeros(
                    self.num_frames, *self._points.shape, device=self.device
                ).requires_grad_(True)
            )
            # TODO: add other deformed attributes

        elif self.dynamic_mode == "deformation":
            deformation_args = ModelHiddenParams(None)
            deformation_args.no_ds = not self.cfg.d_scale
            deformation_args.no_dr = not self.cfg.d_rotation
            deformation_args.no_do = not self.cfg.d_opacity
            self._deformation = DeformationNetwork(deformation_args)
            self._deformation_table = torch.empty(0)
        else:
            raise ValueError(f"Unimplemented dynamic mode {self.dynamic_mode}.")
        
        self.training_setup_dynamic()

    def training_setup_dynamic(self):
        training_args = self.cfg

        l = []
        if self.dynamic_mode == "discrete":
            l += [
                {
                    "params": [self._delta_xyz],
                    "lr": C(training_args.delta_xyz_lr, 0, 0) * self.spatial_lr_scale,
                    "name": "delta_xyz",
                }
            ]
        elif self.dynamic_mode == "deformation":
            l += [
                {
                    "params": list(self._deformation.get_mlp_parameters()),
                    "lr": C(training_args.deformation_lr, 0, 0) * self.spatial_lr_scale,
                    "name": "deformation"
                },
                {
                    "params": list(self._deformation.get_grid_parameters()),
                    "lr": C(training_args.grid_lr, 0, 0) * self.spatial_lr_scale,
                    "name": "grid"
                }
            ]

        self.optimize_list = l
        self.optimize_params = [d["name"] for d in l]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if not ("name" in param_group):
                continue

            if self.dynamic_mode == "discrete":
                if param_group["name"] == "delta_xyz":
                    param_group["lr"] = C(
                        self.cfg.delta_xyz_lr, 0, iteration, interpolation="exp"
                    ) * self.spatial_lr_scale
            elif self.dynamic_mode == "deformation":
                if "grid" in param_group["name"]:
                    param_group["lr"] = C(
                        self.cfg.grid_lr, 0, iteration, interpolation="exp"
                    ) * self.spatial_lr_scale
                elif param_group["name"] == "deformation":
                    param_group["lr"] = C(
                        self.cfg.deformation_lr, 0, iteration, interpolation="exp"
                    ) * self.spatial_lr_scale

        self.color_clip = C(self.cfg.color_clip, 0, iteration)
        
    def get_gs_xyz_from_vertices(self, xyz_vert=None):
        if self.binded_to_surface_mesh:
            if xyz_vert is None:
                xyz_vert = self._points
            # First gather vertices of all triangles
            faces_verts = xyz_vert[self._surface_mesh_faces]  # n_faces, 3, n_coords
            
            # Then compute the points using barycenter coordinates in the surface triangles
            points = faces_verts[:, None] * self.surface_triangle_bary_coords[None]  # n_faces, n_gaussians_per_face, 3, n_coords
            points = points.sum(dim=-2)  # n_faces, n_gaussians_per_face, n_coords
            points = points.reshape(self._n_points, 3)
        else:
            raise ValueError("No vertices when with no mesh binded.")
        return points
    
    def get_timed_xyz_vertices(self, timestamp=None, frame_idx=None, no_spline=False):
        xyz_v = self._points
        timestamp = timestamp.expand(xyz_v.shape[0], 1)
        if not no_spline and self.cfg.use_spline:
            xyz_v_timed = self.spline_interp_xyz(timestamp[0])[0]
        else:
            if self.dynamic_mode == "discrete":
                motion = self._delta_xyz[frame_idx]
                xyz_v_timed = xyz_v + motion
            elif self.dynamic_mode == "deformation":
                xyz_v_timed = self._deformation.forward_dynamic_xyz(xyz_v, timestamp*2-1)
        return xyz_v_timed
    
    def get_timed_xyz_gs(self, timestamp=None, frame_idx=None):
        if self.binded_to_surface_mesh:
            xyz_vert_timed = self.get_timed_xyz_vertices(timestamp, frame_idx)
            xyz_gs_timed = self.get_gs_xyz_from_vertices(xyz_vert_timed)
        else:
            raise NotImplementedError
        return xyz_gs_timed
    
    def get_timed_all(self, timestamp=None, frame_idx=None):
        means3D = self.get_timed_xyz_gs(timestamp, frame_idx)
        scales = self.get_scaling
        rotations = self.get_rotation
        opacity = self.get_opacity
        colors_precomp = self.get_points_rgb()
        return means3D, scales, rotations, opacity, colors_precomp
    
    def init_cubic_spliner(self):
        n_ctrl_knots = self.num_frames
        t_interv = torch.as_tensor(1 / (n_ctrl_knots - 3)).cuda()   # exclude start and end point
        spline_cfg = SplineConfig(
            degree=3, 
            sampling_interval=t_interv,
            start_time=-t_interv, 
            n_knots=self.num_frames
        )
        self.spliner = Spline(spline_cfg)

    def compute_control_knots(self):
        ticks = torch.as_tensor(
            np.linspace(
                self.spliner.start_time.cpu().numpy(),
                self.spliner.end_time.cpu().numpy(),
                self.num_frames,
                endpoint=True
            ),
            dtype=torch.float32,
            device=self.device
        )
        ctrl_knots_xyz = []
        for i, t in enumerate(ticks):
            xyz = self.get_timed_xyz_vertices(t, i, no_spline=True)
            ctrl_knots_xyz.append(xyz)
        ctrl_knots_xyz = torch.stack(ctrl_knots_xyz, dim=0)
        self.spliner.set_data("xyz", ctrl_knots_xyz.permute(1, 0, 2))

    def spline_interp_xyz(self, timestamp: Float[Tensor, "N_t"]):
        return self.spliner(timestamp, keys=["xyz"])["xyz"]
    
    # def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
    #     super().update_step(epoch, global_step, on_load_weights)
        

        

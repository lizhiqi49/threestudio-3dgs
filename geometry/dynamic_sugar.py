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
        use_spline: bool = True
        
        use_deform_graph: bool = True
        n_dg_nodes: int = 1000
        dg_node_connectivity: int = 8
        dg_trans_lr: Any = 0.001
        dg_rot_lr: Any = 0.001

        dynamic_mode: str = "discrete"  # 'discrete', 'deformation'
        delta_xyz_lr: Any = 0.001

        deformation_lr: Any = 0.001
        grid_lr: Any = 0.001

        d_xyz: bool = True
        d_rotation: bool = True
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
            
        if self.cfg.use_deform_graph:
            self.build_deformation_graph(self.cfg.n_dg_nodes, self.cfg.dg_node_connectivity)

        if self.dynamic_mode == "discrete":
            if self.cfg.use_deform_graph:
                self._dg_node_trans = nn.Parameter(
                    torch.zeros((self.num_frames, self.cfg.n_dg_nodes, 3), device="cuda"), requires_grad=True
                )
                dg_node_rots = torch.zeros((self.num_frames, self.cfg.n_dg_nodes, 4), device="cuda")
                dg_node_rots[..., -1] = 1
                self._dg_node_rots = nn.Parameter(dg_node_rots.requires_grad_(True))
            else:
                self._delta_xyz = nn.Parameter(
                    torch.zeros(
                        self.num_frames, *self._points.shape, device=self.device
                    ).requires_grad_(True)
                )

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
            if self.cfg.use_deform_graph:
                l += [
                    {
                        "params": [self._dg_node_trans],
                        "lr": C(training_args.dg_trans_lr, 0, 0) * self.spatial_lr_scale,
                        "name": "dg_trans"
                    },
                    {
                        "params": [self._dg_node_rots],
                        "lr": C(training_args.dg_rot_lr, 0, 0),
                        "name": "dg_rotation"
                    },
                ]
            else:
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
                if param_group["name"] == "dg_trans":
                    param_group["lr"] = C(
                        self.cfg.dg_trans_lr, 0, iteration, interpolation="exp"
                    ) * self.spatial_lr_scale
                if param_group["name"] == "dg_rotation":
                    param_group["lr"] = C(
                        self.cfg.dg_rot_lr, 0, iteration, interpolation="exp"
                    )
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
            if self.cfg.use_deform_graph:
                xyz_v_timed = self.deform(timestamp[0])[0]
            else:
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
    
    def get_timed_dg_trans_rotation(
        self, 
        timestamp: Float[Tensor, "N_t"] = None, 
        frame_idx: Int[Tensor, "N_t"] = None
    ):
        if self.dynamic_mode == "discrete":
            assert frame_idx is not None
            trans = self._dg_node_trans[frame_idx]
            rot = self._dg_node_rots[frame_idx]
        elif self.dynamic_mode == "deformation":
            assert timestamp is not None
            pts = self._deform_graph_node_xyz

            num_pts = pts.shape[0]
            num_t = timestamp.shape[0]
            pts = torch.cat([pts] * num_t, dim=0)
            ts = timestamp.unsqueeze(-1).repeat_interleave(num_pts, dim=0)
            trans, rot = self._deformation.forward_dg_trans_and_rotation(pts, ts*2-1)
            trans = trans.reshape(num_t, num_pts, 3)
            rot = rot.reshape(num_t, num_pts, 4)
            idt_quaternion = torch.zeros((1, num_pts, 4)).to(rot)
            idt_quaternion[..., -1] = 1
            rot = rot + idt_quaternion
        return trans, rot
    
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
        if self.cfg.use_deform_graph:
            self.compute_control_knots_dg()
        else:
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
        
    def compute_control_knots_dg(self):
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
        frame_idx = torch.arange(self.cfg.num_frames, dtype=torch.long, device=self.device)
        trans, rot = self.get_timed_dg_trans_rotation(ticks, frame_idx)
        node_ctrl_knots_trans = trans.permute(1, 0, 2)
        node_ctrl_knots_rots = pp.SO3(rot.permute(1, 0, 2))
        self.spliner.set_data("xyz", node_ctrl_knots_trans)
        self.spliner.set_data("rotation", node_ctrl_knots_rots)

    def spline_interp_xyz(self, timestamp: Float[Tensor, "N_t"]):
        return self.spliner(timestamp, keys=["xyz"])["xyz"]
    
    def spline_interp_dg(self, timestamp: Float[Tensor, "N_t"]) -> Tuple[Tensor, pp.LieTensor]:
        outs = self.spliner(timestamp)
        return outs["xyz"], outs["rotation"]
    
    def build_deformation_graph(self, n_nodes, nodes_connectivity=6):
        device = self.device
        xyz_verts = self.get_xyz_verts
        self._xyz_cpu = xyz_verts.cpu().numpy()
        
        mesh = o3d.io.read_triangle_mesh(self.cfg.surface_mesh_to_bind_path)
        downpcd = mesh.sample_points_uniformly(number_of_points=n_nodes)
        # downpcd = mesh.sample_points_poisson_disk(number_of_points=1000, pcl=downpcd)

        # build deformation graph connectivity
        downpcd.paint_uniform_color([0.5, 0.5, 0.5])
        downpcd_tree = o3d.geometry.KDTreeFlann(downpcd)

        self._deform_graph_node_xyz = torch.from_numpy(np.asarray(downpcd.points)).float().to(device)
        downpcd_size, _ = self._deform_graph_node_xyz.size()
        deform_graph_connectivity = [
            torch.from_numpy(
                np.asarray(
                    downpcd_tree.search_knn_vector_3d(downpcd.points[i], nodes_connectivity + 1)[1][1:]
                )
            ).to(device) 
            for i in range(downpcd_size)
        ]

        self._deform_graph_connectivity = torch.stack(deform_graph_connectivity).long().to(device)
        self._deform_graph_tree = downpcd_tree

        # build connections between the original point cloud to deformation graph node
        xyz_neighbor_node_idx = [
            torch.from_numpy(
                np.asarray(downpcd_tree.search_knn_vector_3d(self._xyz_cpu[i], nodes_connectivity)[1])
            ).to(device) 
            for i in range(self._xyz_cpu.shape[0])
        ]
        xyz_neighbor_nodes_weights = [
            torch.from_numpy(
                np.asarray(downpcd_tree.search_knn_vector_3d(self._xyz_cpu[i], nodes_connectivity)[2])
            ).float().to(device) 
            for i in range(self._xyz_cpu.shape[0])
        ]

        self._xyz_neighbor_node_idx = torch.stack(xyz_neighbor_node_idx).long().to(device)
        self._xyz_neighbor_nodes_weights = torch.stack(xyz_neighbor_nodes_weights).to(device)
        self._xyz_neighbor_nodes_weights = torch.sqrt(self._xyz_neighbor_nodes_weights)
        # xyz_neighbor_nodes_weights = torch.sqrt(torch.tensor(xyz_neighbor_nodes_weights))
        self._xyz_neighbor_nodes_weights = (
            self._xyz_neighbor_nodes_weights 
            / self._xyz_neighbor_nodes_weights.sum(dim=1, keepdim=True)
        )
    
    def deform(self, timestamp: Float[Tensor, "N_t"]):
        n_t = len(timestamp)
        neighbor_nodes_xyz: Float[Tensor, "N_p N_n 3"]
        neighbor_nodes_xyz = self._deform_graph_node_xyz[self._xyz_neighbor_node_idx]
        # neighbor_nodes_rots = self._deform_graph_node_rots[self._xyz_neighbor_node_idx]
        # neighbor_nodes_trans = self._deform_graph_node_trans[self._xyz_neighbor_node_idx]
        
        dg_node_trans, dg_node_rots = self.spline_interp_dg(timestamp)
        neighbor_nodes_trans: Float[Tensor, "N_t N_p N_n 3"]
        neighbor_nodes_rots: Float[Tensor, "N_t N_p N_n 3 3"]
        neighbor_nodes_trans = dg_node_trans[:, self._xyz_neighbor_node_idx]
        neighbor_nodes_rots = dg_node_rots[:, self._xyz_neighbor_node_idx].matrix()

        num_pts = self.get_xyz_verts.shape[0]
        dists_vec: Float[Tensor, "N_t N_p N_n 3 1"]
        dists_vec = (self.get_xyz_verts.unsqueeze(1) - neighbor_nodes_xyz).unsqueeze(-1)
        dists_vec = torch.stack([dists_vec] * n_t, dim=0)

        deformed_xyz: Float[Tensor, "N_t N_p 3"]
        deformed_xyz = torch.bmm(
            neighbor_nodes_rots.reshape(-1, 3, 3), dists_vec.reshape(-1, 3, 1)
        ).squeeze(-1).reshape(n_t, num_pts, -1, 3)
        deformed_xyz = deformed_xyz + neighbor_nodes_xyz.unsqueeze(0) + neighbor_nodes_trans

        nn_weights = self._xyz_neighbor_nodes_weights[None, :, :, None]
        deformed_xyz = (nn_weights * deformed_xyz).sum(dim=2)

        return deformed_xyz
    

        
    # def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
    #     super().update_step(epoch, global_step, on_load_weights)

        

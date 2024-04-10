import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import NamedTuple

import numpy
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

from einops import rearrange
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
# import pygeodesic
# import pygeodesic.geodesic as geodesic

import potpourri3d as pp3d


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
        delta_scales_lr: Any = 0.001

        deformation_lr: Any = 0.001
        grid_lr: Any = 0.001

        d_xyz: bool = True
        d_rotation: bool = True
        d_opacity: bool = False
        d_gs_scale: bool = False

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
            # xyz
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

            # scale
            if self.cfg.d_gs_scale:
                self._delta_scales = nn.Parameter(
                    torch.zeros(
                        self.num_frames, *self._scales.shape, device=self.device
                    )
                )

        elif self.dynamic_mode == "deformation":
            deformation_args = ModelHiddenParams(None)
            deformation_args.no_dr = not self.cfg.d_rotation
            deformation_args.no_ds = not self.cfg.d_gs_scale
            deformation_args.no_do = True

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
                    # xyz
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
            # scales
            if self.cfg.d_gs_scale:
                l += [
                    {
                        "params": [self._delta_scales],
                        "lr": C(training_args.delta_scales_lr, 0, 0) * self.spatial_lr_scale,
                        "name": "delta_scales"
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
                if param_group["name"] == "delta_scales":
                    param_group["lr"] = C(
                        self.cfg.delta_scales_lr, 0, iteration, interpolation="exp"
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

    def get_gs_xyz_from_vertices(self, xyz_vert=None) -> Float[Tensor, "N_t N_gs 3"]:
        if self.binded_to_surface_mesh:
            if xyz_vert is None:
                xyz_vert = self._points
            # First gather vertices of all triangles
            if xyz_vert.ndim == 2:
                xyz_vert = xyz_vert[None]  # (n_t, n_v, 3)
            faces_verts = xyz_vert[:, self._surface_mesh_faces]  # (n_t, n_faces, 3, 3)

            # Then compute the points using barycenter coordinates in the surface triangles
            points = faces_verts[:, :, None] * self.surface_triangle_bary_coords[
                None, None]  # n_t, n_faces, n_gaussians_per_face, 3, n_coords
            points = points.sum(dim=-2)  # n_t, n_faces, n_gaussians_per_face, n_coords
            # points = points.reshape(self._n_points, 3)
            points = rearrange(points, "t f n c -> t (f n) c")
        else:
            raise ValueError("No vertices when with no mesh binded.")
        return points

    def get_timed_xyz_vertices(
        self,
        timestamp: Float[Tensor, "N_t"] = None,
        frame_idx: Int[Tensor, "N_t"] = None,
        no_spline: bool = False
    ) -> Float[Tensor, "N_t N_v 3"]:
        if timestamp is not None:
            if timestamp.ndim == 0:
                timestamp = timestamp.unsqueeze(-1)
        if frame_idx is not None:
            if frame_idx.ndim == 0:
                frame_idx = frame_idx.unsqueeze(-1)

        xyz_v = self._points.unsqueeze(0)
        if not no_spline and self.cfg.use_spline:
            if self.cfg.use_deform_graph:
                xyz_v_timed = self.deform(timestamp)
            else:
                xyz_v_timed = self.spline_interp_xyz(timestamp)
        else:
            if self.dynamic_mode == "discrete":
                if self.cfg.use_deform_graph:
                    xyz_v_timed = self.deform(timestamp, frame_idx)
                else:
                    assert frame_idx is not None
                    motion = self._delta_xyz[frame_idx]
                    xyz_v_timed = xyz_v + motion
            elif self.dynamic_mode == "deformation":
                pts_inp = torch.cat([xyz_v] * timestamp.shape[-1], dim=0).reshape(-1, 3)
                time_inp = timestamp.unsqueeze(-1).repeat_interleave(self._points.shape[0], dim=0) * 2 - 1
                xyz_v_timed = self._deformation.forward_dynamic_xyz(pts_inp, time_inp)
                xyz_v_timed = xyz_v_timed.reshape(timestamp.shape[-1], self._points.shape[0], 3)
        return xyz_v_timed

    def get_timed_xyz_gs(self, timestamp=None, frame_idx=None) -> Float[Tensor, "N_t N_gs 3"]:
        if self.binded_to_surface_mesh:
            xyz_vert_timed = self.get_timed_xyz_vertices(timestamp, frame_idx)
            xyz_gs_timed = self.get_gs_xyz_from_vertices(xyz_vert_timed)
        else:
            raise NotImplementedError
        return xyz_gs_timed

    def get_timed_all(self, timestamp=None, frame_idx=None):
        means3D = self.get_timed_xyz_gs(timestamp, frame_idx)[0]

        # common variables
        common_kwargs = self.get_common_kwargs(timestamp, frame_idx)

        scales = self.get_timed_scales(timestamp, frame_idx, **common_kwargs)[0]

        rotations = self.get_rotation
        opacity = self.get_opacity
        colors_precomp = self.get_points_rgb()
        return means3D, scales, rotations, opacity, colors_precomp

    def get_common_kwargs(self, timestamp=None, frame_idx=None):
        no_spline = False
        hidden = None
        if (no_spline or not self.cfg.use_spline) and self.cfg.dynamic_mode == 'deformation':
            hidden = self.get_deformation_hidden_gs(timestamp)
        return {'no_spline': no_spline, 'hidden': hidden}

    def get_deformation_hidden_gs(self, timestamp):
        assert timestamp is not None
        if timestamp.ndim == 0:
            timestamp = timestamp.unsqueeze(-1)
        # get gs postion at timestamp 0
        xyz_v = self._points.unsqueeze(0)
        xyz_gs = self.get_gs_xyz_from_vertices(xyz_v)

        # query hidden for deformation net
        pts_inp = torch.cat([xyz_gs] * timestamp.shape[-1], dim=0).reshape(-1, 3)
        time_inp = timestamp.unsqueeze(-1).repeat_interleave(pts_inp.shape[0], dim=0) * 2 - 1
        hidden = self._deformation.deformation_net.query_time(pts_inp, None, None, time_inp)
        return hidden

    def get_timed_scales(
        self,
        timestamp: Float[Tensor, "N_t"] = None,
        frame_idx: Int[Tensor, "N_t"] = None,
        **common_kwargs
    ):
        if not self.cfg.d_gs_scale:
            return self.get_scaling
        assert timestamp is not None
        if timestamp.ndim == 0:
            timestamp = timestamp.unsqueeze(-1)
        scales = None

        no_spline = common_kwargs.get('no_spline')

        if not no_spline and self.cfg.use_spline:
            scales = self.spline_interp_scales(timestamp)
        else:
            if self.dynamic_mode == "discrete":
                assert frame_idx is not None
                delta_scale = self._delta_scales[frame_idx]
                scales = self._scales + delta_scale
            elif self.dynamic_mode == "deformation":
                hidden = common_kwargs.get('hidden')
                if hidden is None:
                    hidden = self.get_deformation_hidden_gs(timestamp)
                ds = self._deformation.forward_dynamic_scale(None, None, hidden=hidden)
                scales = self._scales + ds

            scales = torch.cat([
                self.surface_mesh_thickness * torch.ones(len(scales), 1, device=scales.device),
                self.scale_activation(scales)
            ], dim=-1)

        return scales

    # def get_timed_no_dg_gs(
    #     self,
    #     timestamp: Float[Tensor, "N_t"] = None,
    #     frame_idx: Int[Tensor, "N_t"] = None,
    #     no_spline: bool = False
    # ):
    #     # get other timed gaussian attributes(no deformation graph)
    #     scales, rotations, opacity, colors_precomp = None, None, None, None
    #
    #     if timestamp is not None:
    #         if timestamp.ndim == 0:
    #             timestamp = timestamp.unsqueeze(-1)
    #     if frame_idx is not None:
    #         if frame_idx.ndim == 0:
    #             frame_idx = frame_idx.unsqueeze(-1)
    #
    #     xyz_v = self._points.unsqueeze(0)
    #     xyz_gs = self.get_gs_xyz_from_vertices(xyz_v)
    #
    #     if not no_spline and self.cfg.use_spline:
    #         raise NotImplementedError
    #     else:
    #         if self.dynamic_mode == "discrete":
    #             raise NotImplementedError
    #         elif self.dynamic_mode == "deformation":
    #             pts_inp = torch.cat([xyz_gs] * timestamp.shape[-1], dim=0).reshape(-1, 3)
    #             time_inp = timestamp.unsqueeze(-1).repeat_interleave(xyz_gs.shape[0], dim=0) * 2 - 1
    #
    #             # add deformation forward function for scaling predict
    #             # xyz_gs_timed = self._deformation.forward_dynamic_xyz(pts_inp, time_inp)

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
            trans, rot = self._deformation.forward_dg_trans_and_rotation(pts, ts * 2 - 1)
            trans = trans.reshape(num_t, num_pts, 3)
            rot = rot.reshape(num_t, num_pts, 4)
            idt_quaternion = torch.zeros((1, num_pts, 4)).to(rot)
            idt_quaternion[..., -1] = 1
            rot = rot + idt_quaternion
        return trans, rot

    def get_timed_surface_mesh(
        self,
        timestamp: Float[Tensor, "N_t"] = None,
        frame_idx: Int[Tensor, "N_t"] = None
    ) -> Meshes:
        n_t = len(timestamp) if timestamp is not None else len(frame_idx)
        surface_mesh = Meshes(
            verts=self.get_timed_xyz_vertices(timestamp, frame_idx),
            faces=torch.stack([self._surface_mesh_faces] * n_t, dim=0),
            textures=TexturesVertex(
                verts_features=torch.stack([self._vertex_colors] * n_t, dim=0).clamp(0, 1).to(self.device)
            )
        )
        return surface_mesh

    def get_timed_face_normals(
        self,
        timestamp: Float[Tensor, "N_t"] = None,
        frame_idx: Int[Tensor, "N_t"] = None
    ) -> Float[Tensor, "N_t N_faces 3"]:
        return F.normalize(
            self.get_timed_surface_mesh(timestamp, frame_idx).faces_normals_padded(),
            dim=-1
        )

    def get_timed_gs_normals(
        self,
        timestamp: Float[Tensor, "N_t"] = None,
        frame_idx: Int[Tensor, "N_t"] = None
    ) -> Float[Tensor, "N_t N_gs 3"]:
        return self.get_timed_face_normals(timestamp, frame_idx).repeat_interleave(
            self.cfg.n_gaussians_per_surface_triangle, dim=1
        )

    def init_cubic_spliner(self):
        n_ctrl_knots = self.num_frames
        # 32 points have 31 intervals, and 29 intervals for 0~1 time, 2 intervals for start_time~0 and 1~endtime(for interpolate first and last keyframes)
        # So the control knots timestamps are different from keyframes
        t_interv = torch.as_tensor(1 / (n_ctrl_knots - 3)).cuda()  # exclude start and end point
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

        if self.cfg.use_deform_graph:
            self.compute_control_knots_dg()
        else:
            ctrl_knots_xyz = []
            for i, t in enumerate(ticks):
                xyz = self.get_timed_xyz_vertices(t, torch.tensor(i), no_spline=True)
                ctrl_knots_xyz.append(xyz)
            ctrl_knots_xyz = torch.concat(ctrl_knots_xyz)
            self.spliner.set_data("xyz", ctrl_knots_xyz.permute(1, 0, 2))

        if self.cfg.d_gs_scale:
            ctrl_knots_scales = []
            for i, t in enumerate(ticks):
                scale = self.get_timed_scales(t, torch.tensor(i), no_spline=True)
                ctrl_knots_scales.append(scale)

            ctrl_knots_scales = torch.stack(ctrl_knots_scales)
            self.spliner.set_data("scales", ctrl_knots_scales.permute(1, 0, 2))

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

    def spline_interp_scales(self, timestamp: Float[Tensor, "N_t"]):
        return self.spliner(timestamp, keys=["scales"])["scales"]

    def spline_interp_dg(self, timestamp: Float[Tensor, "N_t"]) -> Tuple[Tensor, pp.LieTensor]:
        outs = self.spliner(timestamp, keys=["xyz", "rotation"])
        return outs["xyz"], outs["rotation"]

    def build_deformation_graph(self, n_nodes, nodes_connectivity=6, mode="geodisc"):
        device = self.device
        xyz_verts = self.get_xyz_verts
        self._xyz_cpu = xyz_verts.cpu().numpy()
        mesh = o3d.io.read_triangle_mesh(self.cfg.surface_mesh_to_bind_path)
        downpcd = mesh.sample_points_uniformly(number_of_points=n_nodes)
        # downpcd = mesh.sample_points_poisson_disk(number_of_points=1000, pcl=downpcd)

        # build deformation graph connectivity
        downpcd.paint_uniform_color([0.5, 0.5, 0.5])
        self._deform_graph_node_xyz = torch.from_numpy(np.asarray(downpcd.points)).float().to(device)

        downpcd_tree = o3d.geometry.KDTreeFlann(downpcd)

        if mode == "eucdisc":
            downpcd_size, _ = self._deform_graph_node_xyz.size()
            # TODO delete unused connectivity attr
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
        elif mode == "geodisc":
            # vertices = self._xyz_cpu
            # faces = self._surface_mesh_faces.cpu().numpy()
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)

            # init geodisc calculation algorithm (geodesic version)
            # geoalg = geodesic.PyGeodesicAlgorithmExact(vertices, faces)

            # init geodisc calculation algorithm (potpourri3d  version)
            geoalg = pp3d.MeshHeatMethodDistanceSolver(vertices, faces)

            # 1. build a kd tree for all vertices in mesh
            mesh_pcl = o3d.geometry.PointCloud()
            mesh_pcl.points = o3d.utility.Vector3dVector(mesh.vertices)
            mesh_kdtree = o3d.geometry.KDTreeFlann(mesh_pcl)

            # 2. find the nearest vertex of all downsampled points and get their index.
            nearest_vertex = [
                np.asarray(mesh_kdtree.search_knn_vector_3d(downpcd.points[i], 1)[1])[0]
                for i in range(n_nodes)
            ]
            target_index = np.array(nearest_vertex)

            # 3. find k nearest neighbors(geodistance) of mesh vertices and downsample pointcloud
            xyz_neighbor_node_idx = []
            xyz_neighbor_nodes_weights = []
            downpcd_points = np.asarray(downpcd.points)
            for i in range(self._xyz_cpu.shape[0]):
                source_index = np.array([i])
                start_time1 = time.time()
                # geodesic distance calculation
                # distances = geoalg.geodesicDistances(source_index, target_index)[0]

                # potpourri3d distance calculation
                distances = geoalg.compute_distance(source_index)[target_index]

                print(f"i: {i}, geodist calculation: {time.time() - start_time1}")
                sorted_index = np.argsort(distances)

                k_n_neighbor = sorted_index[:nodes_connectivity]

                xyz_neighbor_node_idx.append(torch.from_numpy(k_n_neighbor).to(device))

                xyz_neighbor_nodes_weights.append(
                    torch.from_numpy(np.reciprocal(
                        np.linalg.norm(vertices[i] - downpcd_points[k_n_neighbor], axis=1) + 1e-5)).float().to(device))
        else:
            print("The mode must be eucdisc or geodisc!")
            raise NotImplementedError

        self._xyz_neighbor_node_idx = torch.stack(xyz_neighbor_node_idx).long().to(device)

        print(torch.max(self._xyz_neighbor_node_idx))
        print(torch.min(self._xyz_neighbor_node_idx))

        self._xyz_neighbor_nodes_weights = torch.stack(xyz_neighbor_nodes_weights).to(device)
        # a = torch.sum(self._xyz_neighbor_nodes_weights < 0)
        # self._xyz_neighbor_nodes_weights[self._xyz_neighbor_nodes_weights < 0] = 0
        self._xyz_neighbor_nodes_weights = torch.sqrt(self._xyz_neighbor_nodes_weights)
        # normalize
        self._xyz_neighbor_nodes_weights = (
            self._xyz_neighbor_nodes_weights
            / self._xyz_neighbor_nodes_weights.sum(dim=1, keepdim=True)
        )

    def deform(
        self,
        timestamp: Float[Tensor, "N_t"] = None,
        frame_idx: Int[Tensor, "N_t"] = None
    ) -> Float[Tensor, "N_t N_p 3"]:
        n_t = len(timestamp) if timestamp is not None else len(frame_idx)
        neighbor_nodes_xyz: Float[Tensor, "N_p N_n 3"]
        neighbor_nodes_xyz = self._deform_graph_node_xyz[self._xyz_neighbor_node_idx]
        # neighbor_nodes_rots = self._deform_graph_node_rots[self._xyz_neighbor_node_idx]
        # neighbor_nodes_trans = self._deform_graph_node_trans[self._xyz_neighbor_node_idx]

        if self.cfg.use_spline:
            dg_node_trans, dg_node_rots = self.spline_interp_dg(timestamp)
        else:
            assert frame_idx is not None
            # ! discrete mode is not compatible with no spline config
            dg_node_trans = self._dg_node_trans[frame_idx]
            dg_node_rots = pp.SO3(self._dg_node_rots[frame_idx])

        # debug
        # a = torch.sum(dg_node_trans)
        # b = torch.sum(dg_node_rots)
        # c = torch.sum(self._xyz_neighbor_node_idx)
        # d = torch.sum(neighbor_nodes_xyz)
        # e = torch.sum(self.get_xyz_verts)

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

        # f = torch.sum(nn_weights)
        deformed_xyz = (nn_weights * deformed_xyz).sum(dim=2)

        return deformed_xyz

    # def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
    #     super().update_step(epoch, global_step, on_load_weights)

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

from .gaussian_base import (
    GaussianBaseModel, 
    BasicPointCloud, 
)
from .gaussian_base import RGB2SH, inverse_sigmoid, build_rotation

@threestudio.register("spacetime-gaussian-splatting")
class SpacetimeGaussianModel(GaussianBaseModel):
    @dataclass
    class Config(GaussianBaseModel.Config):
        omega_lr: float = 0.01
        trbfc_lr: float = 0.01
        trbfs_lr: float = 0.01
        move_lr: float = 0.01
        
        addsphpointsscale: float = 0.8
        trbfslinit: float = 0.1
        raystart: float = 0.7
        spatial_lr_scale: float = 10.0
        
    cfg: Config
    
    def configure(self) -> None:
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
        self.setup_functions()
        
        # Add temporal parameters
        self._motion = torch.empty(0)
        self.percent_dense = 0
        self.spatial_lr_scale = 0
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

        self.maxz, self.minz =  0.0 , 0.0 
        self.maxy, self.miny =  0.0 , 0.0 
        self.maxx, self.minx =  0.0 , 0.0  
        self.computedtrbfscale = None
        self.computedopacity = None
        self.raystart = self.cfg.raystart

        if self.cfg.geometry_convert_from.startswith("shap-e:"):
            shap_e_guidance = threestudio.find("shap-e-guidance")(
                self.cfg.shap_e_guidance_config
            )
            prompt = self.cfg.geometry_convert_from[len("shap-e:") :]
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
            prompt = self.cfg.geometry_convert_from[len("lrm:") :]
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
        
        
    def get_rotation(self, delta_t):
        rotation = self._rotation + delta_t * self._omega
        self.delta_t = delta_t
        return self.rotation_activation(rotation)
    
    @property
    def get_trbfcenter(self):
        return self._trbf_center
    
    @property
    def get_trbfscale(self):
        return self._trbf_scale
    
    def get_features(self, delta_t):
        return self._features_dc
    
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            
    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        # TODO: whether use RGB2SH here?
        # fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()   # RGB
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
        self._features_dc = nn.Parameter(fused_color.unqueeze(1).contiguous().requires_grad_(True))

        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        omega = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        self._omega = nn.Parameter(omega.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        if self.cfg.pred_normal:
            normals = torch.zeros((fused_point_cloud.shape[0], 3), device="cuda")
            self._normal = nn.Parameter(normals.requires_grad_(True))
            
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")

        self.fused_point_cloud = fused_point_cloud.cpu().clone().detach()
        # self.features = features.cpu().clone().detach()
        self.scales = scales.cpu().clone().detach()
        self.rots = rots.cpu().clone().detach()
        self.opacities = opacities.cpu().clone().detach()
        
        omega = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        self._omega = nn.Parameter(omega.requires_grad_(True))
        
        motion = torch.zeros((fused_point_cloud.shape[0], 9), device="cuda")# x1, x2, x3,  y1,y2,y3, z1,z2,z3
        self._motion = nn.Parameter(motion.requires_grad_(True))
        
        times = torch.zeros((fused_point_cloud.shape[0], 1), device="cuda")
        self._trbf_center = nn.Parameter(times.contiguous().requires_grad_(True))
        self._trbf_scale = nn.Parameter(torch.ones((self.get_xyz.shape[0], 1), device="cuda").requires_grad_(True)) 
        
        if self.trbfslinit is not None:
            nn.init.constant_(self._trbf_scale, self.trbfslinit) # too large ?
        else:
            nn.init.constant_(self._trbf_scale, 0) # too large ?

        nn.init.constant_(self._omega, 0)
        self.rgb_grd = {}

        self.maxz, self.minz = torch.amax(self._xyz[:,2]), torch.amin(self._xyz[:,2]) 
        self.maxy, self.miny = torch.amax(self._xyz[:,1]), torch.amin(self._xyz[:,1]) 
        self.maxx, self.minx = torch.amax(self._xyz[:,0]), torch.amin(self._xyz[:,0]) 
        self.maxz = min((self.maxz, 200.0)) # some outliers in the n4d datasets.. 
        
    def training_setup(self):
        training_args = self.cfg
        # self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        position_lr_init = C(training_args.position_lr, 0, 0) * self.spatial_lr_scale
        l = [
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

        self.optimize_params = [
            "xyz",
            "f_dc",
            "f_rest",
            "opacity",
            "scaling",
            "rotation",
        ]
        self.optimize_list = l
        self.optimize_params = [d["name"] for d in l]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        
    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if not ("name" in param_group):
                continue
            if param_group["name"] == "xyz":
                param_group["lr"] = C(
                    self.cfg.position_lr, 0, iteration, interpolation="exp"
                ) * self.spatial_lr_scale
            # if param_group["name"] == "scaling":
            #     param_group["lr"] = C(
            #         self.cfg.scaling_lr, 0, iteration, interpolation="exp"
            #     )
            # if param_group["name"] == "f_dc":
            #     param_group["lr"] = C(
            #         self.cfg.feature_lr, 0, iteration, interpolation="exp"
            #     )
            # if param_group["name"] == "opacity":
            #     param_group["lr"] = C(
            #         self.cfg.opacity_lr, 0, iteration, interpolation="exp"
            #     )
            # if param_group["name"] == "rotation":
            #     param_group["lr"] = C(
            #         self.cfg.rotation_lr, 0, iteration, interpolation="exp"
            #     )
            # if param_group["name"] == "normal":
            #     param_group["lr"] = C(
            #         self.cfg.normal_lr, 0, iteration, interpolation="exp"
            #     )
        self.color_clip = C(self.cfg.color_clip, 0, iteration)
        
    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        #self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.cfg.pred_normal:
            self._normal = optimizable_tensors["normal"]
        
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        
        self._trbf_center = optimizable_tensors["trbf_center"]
        self._trbf_scale = optimizable_tensors["trbf_scale"]
        self._motion = optimizable_tensors["motion"]
        self._omega = optimizable_tensors["omega"]
        
        if self.omegamask is not None :
            self.omegamask = self.omegamask[valid_points_mask]
            
    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        # new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_trbf_center,
        new_trbf_scale,
        new_motion,
        new_omega,
        new_normal=None,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            # "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "trbf_center": new_trbf_center,
            "trbf_scale": new_trbf_scale,
            "motion": new_motion,
            "omega": new_omega,
        }
        if self.cfg.pred_normal:
            d.update({"normal": new_normal})

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        # self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._trbf_center = optimizable_tensors["trbf_center"]
        self._trbf_scale = optimizable_tensors["trbf_scale"]
        self._motion = optimizable_tensors["motion"]
        self._omega = optimizable_tensors["omega"]
        if self.cfg.pred_normal:
            self._normal = optimizable_tensors["normal"]

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
        
        new_trbf_center = self._trbf_center[selected_pts_mask].repeat(N,1)
        new_trbf_center = torch.rand_like(new_trbf_center) #* 0.5
        new_trbf_scale = self._trbf_scale[selected_pts_mask].repeat(N,1)
        new_motion = self._motion[selected_pts_mask].repeat(N,1)
        new_omega = self._omega[selected_pts_mask].repeat(N,1)
        
        if self.cfg.pred_normal:
            new_normal = self._normal[selected_pts_mask].repeat(N, 1)
        else:
            new_normal = None

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            # new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_trbf_center,
            new_trbf_scale,
            new_motion,
            new_omega,
            new_normal,
        )

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
        
        new_trbf_center =  torch.rand((self._trbf_center[selected_pts_mask].shape[0], 1), device="cuda")  #self._trbf_center[selected_pts_mask]
        new_trbf_scale = self._trbf_scale[selected_pts_mask]
        new_motion = self._motion[selected_pts_mask]
        new_omega = self._omega[selected_pts_mask]
        
        if self.cfg.pred_normal:
            new_normal = self._normal[selected_pts_mask]
        else:
            new_normal = None

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            # new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_trbf_center,
            new_trbf_scale,
            new_motion,
            new_omega,
            new_normal,
        )
       
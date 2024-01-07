import math
from dataclasses import dataclass

import numpy as np
import threestudio
import torch
import torch.nn.functional as F
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import Rasterizer
from threestudio.utils.typing import *

from .gaussian_batch_renderer import GaussianBatchRenderer
from ..geometry.spacetime_gaussian import TemporalCamera, SpacetimeGaussianModel


def basicfunction(x):
    return torch.exp(-1*x.pow(2))

@threestudio.register("diff-gaussian-rasterizer-spacetime")
class DiffGaussian(Rasterizer, GaussianBatchRenderer):
    @dataclass
    class Config(Rasterizer.Config):
        debug: bool = False
        invert_bg_prob: float = 1.0
        back_ground_color: Tuple[float, float, float] = (1, 1, 1)

    cfg: Config

    def configure(
        self,
        geometry: BaseGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        threestudio.info(
            "[Note] Gaussian Splatting doesn't support material and background now."
        )
        super().configure(geometry, material, background)
        self.background_tensor = torch.tensor(
            self.cfg.back_ground_color, dtype=torch.float32, device="cuda"
        )

    def forward(
        self,
        viewpoint_camera,
        bg_color: torch.Tensor,
        scaling_modifier=1.0,
        override_color=None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """

        if self.training:
            invert_bg_color = np.random.rand() > self.cfg.invert_bg_prob
        else:
            invert_bg_color = True

        bg_color = bg_color if not invert_bg_color else (1.0 - bg_color)

        pc: SpacetimeGaussianModel
        pc = self.geometry
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
            )
            + 0
        )
        pointtimes = (
            torch.ones(
                (pc.get_xyz.shape[0],1), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda"
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = pc.get_xyz
        means2D = screenspace_points
        pointopacity = pc.get_opacity
        
        trbfcenter = pc.get_trbfcenter
        trbfscale = pc.get_trbfscale
        
        trbfdistanceoffset = viewpoint_camera.timestamp * pointtimes - trbfcenter
        trbfdistance =  trbfdistanceoffset / torch.exp(trbfscale) 
        trbfoutput = basicfunction(trbfdistance)
        
        opacity = pointopacity * trbfoutput  # - 0.5
        pc.trbfoutput = trbfoutput

        cov3D_precomp = None

        scales = pc.get_scaling

        shs = None
        tforpoly = trbfdistanceoffset.detach()
        means3D = (
            means3D 
            + pc._motion[:, 0:3] * tforpoly 
            + pc._motion[:, 3:6] * tforpoly * tforpoly 
            + pc._motion[:, 6:9] * tforpoly *tforpoly * tforpoly
        )

        rotations = pc.get_rotation(tforpoly) # to try use 
        colors_precomp = pc.get_features(tforpoly).reshape(pc.get_xyz.shape[0], 3)

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

        # Retain gradients of the 2D (screen-space) means for batch dim
        if self.training:
            screenspace_points.retain_grad()

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "render": rendered_image.clamp(0, 1),
            "depth": rendered_depth,
            "mask": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }

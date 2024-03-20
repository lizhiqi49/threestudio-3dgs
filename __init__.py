import threestudio
from packaging.version import Version

if hasattr(threestudio, "__version__") and Version(threestudio.__version__) >= Version(
    "0.2.1"
):
    pass
else:
    if hasattr(threestudio, "__version__"):
        print(f"[INFO] threestudio version: {threestudio.__version__}")
    raise ValueError(
        "threestudio version must be >= 0.2.0, please update threestudio by pulling the latest version from github"
    )


from .background import gaussian_mvdream_background
from .geometry import (
    exporter,
    gaussian_base,
    gaussian_io,
    spacetime_gaussian,
    sugar,
    dynamic_sugar,
)
from .material import gaussian_material
from .renderer import (
    diff_gaussian_rasterizer,
    diff_gaussian_rasterizer_advanced,
    diff_gaussian_rasterizer_background,
    diff_gaussian_rasterizer_shading,
    diff_gaussian_rasterizer_st,
    diff_gaussian_rasterizer_normal,
    diff_sugar_rasterizer_normal,
    diff_sugar_rasterizer_temporal,
)
from .system import (
    gaussian_mvdream,
    gaussian_splatting,
    gaussian_zero123,
    # gaussian_4dgen,
    # sugar_zero123,
    sugar_4dgen,
    # sugar_imagedream,
    # gaussian_to_sugar,
    sugar_static,
)
from .data import temporal_image, image
from .guidance import (
    temporal_stable_zero123_guidance,
    stable_diffusion_lora_guidance
)

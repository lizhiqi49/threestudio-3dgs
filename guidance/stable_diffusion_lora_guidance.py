from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.ops import perpendicular_component
from threestudio.utils.typing import *

from threestudio.models.guidance.stable_diffusion_guidance import StableDiffusionGuidance
from diffusers import UNet2DConditionModel, StableDiffusionPipeline
from diffusers.models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT

@threestudio.register("stable-diffusion-lora-guidance")
class StableDiffusionGuidanceWithLoRA(StableDiffusionGuidance):
    @dataclass
    class Config(StableDiffusionGuidance.Config):
        pretrained_adapter_name_or_path: str = ...

    def configure(self) -> None:
        super().configure()
        self.load_lora_to_unet()
        self.pipe: StableDiffusionPipeline
        self.pipe.fuse_lora(fuse_text_encoder=False)
        self.unet.enable_lora()
        
    def load_lora_to_unet(self):
        state_dict, network_alphas = self.pipe.lora_state_dict(self.cfg.pretrained_adapter_name_or_path)

        is_correct_format = all("lora" in key for key in state_dict.keys())
        if not is_correct_format:
            raise ValueError("Invalid LoRA checkpoint.")

        low_cpu_mem_usage = _LOW_CPU_MEM_USAGE_DEFAULT

        self.pipe.load_lora_into_unet(
            state_dict,
            network_alphas=network_alphas,
            unet=getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet,
            low_cpu_mem_usage=low_cpu_mem_usage,
            adapter_name=None,
            _pipeline=self.pipe,
        )

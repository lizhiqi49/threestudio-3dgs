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

from .gaussian_base import GaussianBaseModel

@threestudio.register("spacetime-gaussian-splatting")
class SpacetimeGaussianModel(GaussianBaseModel):
    @dataclass
    class Config(GaussianBaseModel.Config):
        
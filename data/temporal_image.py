import bisect
import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

import threestudio
from threestudio import register
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_rank

from threestudio.utils.typing import *

from threestudio.data.image import (
    SingleImageDataModuleConfig, 
    SingleImageIterableDataset,
    SingleImageDataset,
)
from .uncond import (
    RandomCameraDataModuleConfig,
    RandomCameraDataset,
    RandomCameraIterableDataset,
)

@dataclass 
class TemporalRandomImageDataModuleConfig(SingleImageDataModuleConfig):
    num_frames: int = 14
    video_frames_dir: Optional[str] = None 
    norm_timestamp: bool = False

class TemporalRandomImageIterableDataset(IterableDataset, Updateable):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg = parse_structured(
            TemporalRandomImageDataModuleConfig, cfg
        )
        self.num_frames = self.cfg.num_frames

        if self.cfg.use_random_camera:
            self.rand_cam_bs = self.cfg.random_camera.batch_size
            self.cfg.random_camera.update(
                {"batch_size": self.num_frames * self.rand_cam_bs}
            )
        self.setup(cfg, split)
        # self.single_image_dataset = SingleImageIterableDataset(self.cfg, split)

    def setup(self, cfg, split):
        self.split = split
        self.rank = get_rank()
        self.cfg: TemporalRandomImageDataModuleConfig = cfg

        if self.cfg.use_random_camera:
            random_camera_cfg = parse_structured(
                RandomCameraDataModuleConfig, self.cfg.get("random_camera", {})
            )
            if split == "train":
                self.random_pose_generator = RandomCameraIterableDataset(
                    random_camera_cfg
                )
            else:
                self.random_pose_generator = RandomCameraDataset(
                    random_camera_cfg, split
                )

        elevation_deg = torch.FloatTensor([self.cfg.default_elevation_deg])
        azimuth_deg = torch.FloatTensor([self.cfg.default_azimuth_deg])
        camera_distance = torch.FloatTensor([self.cfg.default_camera_distance])

        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180
        camera_position: Float[Tensor, "1 3"] = torch.stack(
            [
                camera_distance * torch.cos(elevation) * torch.cos(azimuth),
                camera_distance * torch.cos(elevation) * torch.sin(azimuth),
                camera_distance * torch.sin(elevation),
            ],
            dim=-1,
        )

        center: Float[Tensor, "1 3"] = torch.zeros_like(camera_position)
        up: Float[Tensor, "1 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None]

        light_position: Float[Tensor, "1 3"] = camera_position
        lookat: Float[Tensor, "1 3"] = F.normalize(center - camera_position, dim=-1)
        right: Float[Tensor, "1 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        self.c2w: Float[Tensor, "1 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_position[:, :, None]],
            dim=-1,
        )
        self.c2w4x4: Float[Tensor, "B 4 4"] = torch.cat(
            [self.c2w, torch.zeros_like(self.c2w[:, :1])], dim=1
        )
        self.c2w4x4[:, 3, 3] = 1.0

        self.camera_position = camera_position
        self.light_position = light_position
        self.elevation_deg, self.azimuth_deg = elevation_deg, azimuth_deg
        self.camera_distance = camera_distance
        self.fovy = torch.deg2rad(torch.FloatTensor([self.cfg.default_fovy_deg]))

        self.heights: List[int] = (
            [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        )
        self.widths: List[int] = (
            [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        )
        assert len(self.heights) == len(self.widths)
        self.resolution_milestones: List[int]
        if len(self.heights) == 1 and len(self.widths) == 1:
            if len(self.cfg.resolution_milestones) > 0:
                threestudio.warn(
                    "Ignoring resolution_milestones since height and width are not changing"
                )
            self.resolution_milestones = [-1]
        else:
            assert len(self.heights) == len(self.cfg.resolution_milestones) + 1
            self.resolution_milestones = [-1] + self.cfg.resolution_milestones

        self.focal_lengths = [
            0.5 * height / torch.tan(0.5 * self.fovy) for height in self.heights
        ]

        self.height: int = self.heights[0]
        self.width: int = self.widths[0]
        self.focal_length = self.focal_lengths[0]
        self.load_video_frames()
        self.prev_height = self.height

        self.timestamps = torch.arange(self.num_frames, dtype=torch.float32)
        if self.cfg.norm_timestamp:
            self.timestamps = self.timestamps / self.num_frames

    # Copied from threestudio.data.image.SingleImageDataBase.load_images
    def load_single_frame(self, frame_path):
        # load image
        assert os.path.exists(frame_path), f"Could not find image {frame_path}!"
        rgba = cv2.cvtColor(
            cv2.imread(frame_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA
        )
        rgba = (
            cv2.resize(
                rgba, (self.width, self.height), interpolation=cv2.INTER_AREA
            ).astype(np.float32)
            / 255.0
        )
        rgb = rgba[..., :3]
        rgb: Float[Tensor, "1 H W 3"] = (
            torch.from_numpy(rgb).unsqueeze(0).contiguous().to(self.rank)
        )
        self.rgbs.append(rgb)
        mask: Float[Tensor, "1 H W 1"] = (
            torch.from_numpy(rgba[..., 3:] > 0.5).unsqueeze(0).to(self.rank)
        )
        self.masks.append(mask)
        print(
            f"[INFO] single image dataset: load image {frame_path} {self.rgb.shape}"
        )

        # load depth
        if self.cfg.requires_depth:
            depth_path = frame_path.replace("_rgba.png", "_depth.png")
            assert os.path.exists(depth_path)
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            depth = cv2.resize(
                depth, (self.width, self.height), interpolation=cv2.INTER_AREA
            )
            depth: Float[Tensor, "1 H W 1"] = (
                torch.from_numpy(depth.astype(np.float32) / 255.0)
                .unsqueeze(0)
                .to(self.rank)
            )
            self.depths.append(depth)
            print(
                f"[INFO] single image dataset: load depth {depth_path} {depth.shape}"
            )
        else:
            depth = None

        # load normal
        if self.cfg.requires_normal:
            normal_path = frame_path.replace("_rgba.png", "_normal.png")
            assert os.path.exists(normal_path)
            normal = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)
            normal = cv2.resize(
                normal, (self.width, self.height), interpolation=cv2.INTER_AREA
            )
            normal: Float[Tensor, "1 H W 3"] = (
                torch.from_numpy(normal.astype(np.float32) / 255.0)
                .unsqueeze(0)
                .to(self.rank)
            )
            self.normals.append(normal)
            print(
                f"[INFO] single image dataset: load normal {normal_path} {normal.shape}"
            )
        else:
            normal = None

    
    def load_video_frames(self):
        assert os.path.exists(self.cfg.video_frames_dir), f"Could not find image {self.cfg.video_frames_dir}!"
        self.rgbs = []
        self.masks = []
        if self.cfg.requires_depth:
            self.depths = []
        if self.cfg.requires_normal:
            self.normals = []

        for idx in range(self.cfg.num_frames):
            frame_path = os.path.join(self.cfg.video_frames_dir, f"{idx:03}_rgba.png")
            self.load_single_frame(frame_path)

        self.rgbs = torch.cat(self.rgbs, dim=0)
        self.masks = torch.cat(self.masks, dim=0)
        if self.cfg.requires_depth:
            self.depths = torch.cat(self.depths, dim=0)
        if self.cfg.requires_normal:
            self.normals = torch.cat(self.normals, dim=0)


    def get_all_images(self):
        return self.rgbs

    def collate(self, batch) -> Dict[str, Any]:

        batch = {
            # "rays_o": self.rays_o,
            # "rays_d": self.rays_d,
            # "mvp_mtx": self.mvp_mtx,
            "camera_positions": self.camera_position,
            "light_positions": self.light_position,
            "elevation": self.elevation_deg,
            "azimuth": self.azimuth_deg,
            "camera_distances": self.camera_distance,
            "rgb": self.rgbs,
            "ref_depth": self.depths,
            "ref_normal": self.normals,
            "mask": self.masks,
            "height": self.height,
            "width": self.width,
            "c2w": self.c2w4x4.repeat(self.num_frames, 1, 1),
            "fovy": self.fovy.repeat(self.num_frames),
            "timestamp": self.timestamps
        }
        if self.cfg.use_random_camera:
            batch_rand_cam = self.random_pose_generator.collate(None)
            batch_rand_cam["timestamp"] = self.timestamps.repeat_interleave(self.rand_cam_bs)
            batch["random_camera"] = batch_rand_cam

        return batch
    
    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        size_ind = bisect.bisect_right(self.resolution_milestones, global_step) - 1
        self.height = self.heights[size_ind]
        if self.height == self.prev_height:
            return

        self.prev_height = self.height
        self.width = self.widths[size_ind]
        self.focal_length = self.focal_lengths[size_ind]
        threestudio.debug(f"Training height: {self.height}, width: {self.width}")
        self.load_video_frames()

        self.random_pose_generator.update_step(epoch, global_step, on_load_weights)

    def __iter__(self):
        while True:
            yield {}


@register("temporal-image-datamodule")
class TemporalRandomImageDataModule(pl.LightningDataModule):
    cfg: TemporalRandomImageDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(TemporalRandomImageDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = TemporalRandomImageIterableDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = RandomCameraDataset(self.cfg.get("random_camera", {}), "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = RandomCameraDataset(self.cfg.get("random_camera", {}), "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset, num_workers=0, batch_size=batch_size, collate_fn=collate_fn
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            collate_fn=self.train_dataset.collate,
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)

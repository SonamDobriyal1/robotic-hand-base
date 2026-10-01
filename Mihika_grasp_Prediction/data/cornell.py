"""Cornell Grasp Dataset loader.

The Cornell dataset stores each object as an RGB image, a depth image, and a
text file containing positive grasp rectangles. This loader converts those
rectangles into dense maps so a fully convolutional network can learn:

- grasp quality at each pixel
- grasp angle at each pixel
- gripper opening width at each pixel
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

InputMode = Literal["rgb", "depth", "rgbd"]
Split = Literal["train", "val", "test", "all"]


@dataclass(frozen=True)
class GraspRectangle:
    """One oriented grasp rectangle using image coordinates as ``(x, y)``."""

    points: np.ndarray

    def __post_init__(self) -> None:
        if self.points.shape != (4, 2):
            raise ValueError("GraspRectangle points must have shape (4, 2).")

    @property
    def center(self) -> np.ndarray:
        return self.points.mean(axis=0)

    @property
    def angle(self) -> float:
        """Return the grasp axis angle in radians, normalized to [-pi/2, pi/2]."""

        edge = self.points[1] - self.points[0]
        angle = float(np.arctan2(edge[1], edge[0]))
        return normalize_grasp_angle(angle)

    @property
    def width(self) -> float:
        """Approximate gripper opening in pixels.

        Cornell rectangles are annotated with four corners. For this simple MVP
        we use the first edge as the gripper opening direction.
        """

        return float(np.linalg.norm(self.points[1] - self.points[0]))

    def scaled(self, sx: float, sy: float) -> "GraspRectangle":
        scale = np.array([sx, sy], dtype=np.float32)
        return GraspRectangle((self.points * scale).astype(np.float32))


@dataclass(frozen=True)
class CornellRecord:
    rgb_path: Path
    depth_path: Path
    grasp_path: Path


def normalize_grasp_angle(angle: float) -> float:
    """Normalize antipodal grasp angles to [-pi/2, pi/2]."""

    while angle > np.pi / 2:
        angle -= np.pi
    while angle < -np.pi / 2:
        angle += np.pi
    return float(angle)


def load_grasp_rectangles(path: Path) -> list[GraspRectangle]:
    """Read positive Cornell grasp rectangles from a ``*cpos.txt`` file."""

    raw_points: list[list[float]] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.replace("\t", " ").split()
        if len(parts) < 2:
            continue
        try:
            x, y = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        if np.isfinite(x) and np.isfinite(y):
            raw_points.append([x, y])

    rectangles: list[GraspRectangle] = []
    for start in range(0, len(raw_points), 4):
        block = raw_points[start : start + 4]
        if len(block) == 4:
            rectangles.append(GraspRectangle(np.asarray(block, dtype=np.float32)))
    return rectangles


class CornellGraspDataset(Dataset[dict[str, torch.Tensor]]):
    """PyTorch dataset for Cornell RGB-D grasp prediction."""

    def __init__(
        self,
        root: str | Path,
        input_mode: InputMode = "rgbd",
        image_size: int = 300,
        split: Split = "train",
        val_split: float = 0.15,
        test_split: float = 0.15,
        seed: int = 42,
    ) -> None:
        self.root = Path(root)
        self.input_mode = input_mode
        self.image_size = int(image_size)
        self.split = split
        self.val_split = float(val_split)
        self.test_split = float(test_split)
        self.seed = int(seed)

        if not 0.0 < self.val_split < 1.0:
            raise ValueError("val_split must be between 0 and 1.")
        if not 0.0 < self.test_split < 1.0:
            raise ValueError("test_split must be between 0 and 1.")
        if self.val_split + self.test_split >= 1.0:
            raise ValueError("val_split + test_split must be less than 1.")

        if self.input_mode not in {"rgb", "depth", "rgbd"}:
            raise ValueError("input_mode must be one of: rgb, depth, rgbd")

        records = self._find_records(self.root)
        if not records:
            raise FileNotFoundError(
                f"No Cornell samples found under {self.root}. Expected files like "
                "pcd0100r.png, pcd0100d.tiff, and pcd0100cpos.txt."
            )

        rng = np.random.default_rng(self.seed)
        indices = np.arange(len(records))
        rng.shuffle(indices)

        if split == "all":
            chosen = indices
        elif split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test, all")
        else:
            val_count = max(1, int(round(len(indices) * self.val_split)))
            test_count = max(1, int(round(len(indices) * self.test_split)))
            train_count = len(indices) - val_count - test_count
            if train_count < 1:
                raise ValueError(
                    "The dataset is too small for non-empty train, validation, and test "
                    "splits with the configured ratios."
                )
            train_indices = indices[:train_count]
            val_indices = indices[train_count : train_count + val_count]
            test_indices = indices[train_count + val_count :]
            chosen = {"train": train_indices, "val": val_indices, "test": test_indices}[split]

        self.records = [records[int(i)] for i in chosen]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rgb, depth, original_size = self._load_images(index)
        image = self._build_input_tensor(rgb, depth)
        rectangles = self.get_rectangles(index, original_size=original_size)
        quality, angle, width = self._build_target_maps(rectangles)

        return {
            "image": image,
            "quality": torch.from_numpy(quality).unsqueeze(0),
            "angle": torch.from_numpy(angle).unsqueeze(0),
            "width": torch.from_numpy(width).unsqueeze(0),
            "index": torch.tensor(index, dtype=torch.long),
        }

    @property
    def in_channels(self) -> int:
        if self.input_mode == "rgb":
            return 3
        if self.input_mode == "depth":
            return 1
        return 4

    def get_rgb_image(self, index: int) -> Image.Image:
        """Return the resized RGB image for visualization."""

        record = self.records[index]
        return Image.open(record.rgb_path).convert("RGB").resize(
            (self.image_size, self.image_size),
            resample=Image.Resampling.BILINEAR,
        )

    def get_rectangles(
        self,
        index: int,
        original_size: tuple[int, int] | None = None,
    ) -> list[GraspRectangle]:
        """Return grasp rectangles scaled to the configured square image size."""

        record = self.records[index]
        rectangles = load_grasp_rectangles(record.grasp_path)
        if original_size is None:
            with Image.open(record.rgb_path) as img:
                original_size = img.size
        original_w, original_h = original_size
        sx = self.image_size / float(original_w)
        sy = self.image_size / float(original_h)
        return [rect.scaled(sx=sx, sy=sy) for rect in rectangles]

    @staticmethod
    def _find_records(root: Path) -> list[CornellRecord]:
        records: list[CornellRecord] = []
        for grasp_path in sorted(root.rglob("*cpos.txt")):
            base = grasp_path.name.replace("cpos.txt", "")
            parent = grasp_path.parent
            rgb_path = _first_existing(
                [
                    parent / f"{base}r.png",
                    parent / f"{base}r.jpg",
                    parent / f"{base}r.jpeg",
                ]
            )
            depth_path = _first_existing(
                [
                    parent / f"{base}d.tiff",
                    parent / f"{base}d.tif",
                    parent / f"{base}d.png",
                    parent / f"{base}d.jpg",
                ]
            )
            if rgb_path is not None and depth_path is not None:
                records.append(
                    CornellRecord(
                        rgb_path=rgb_path,
                        depth_path=depth_path,
                        grasp_path=grasp_path,
                    )
                )
        return records

    def _load_images(self, index: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
        record = self.records[index]
        rgb_img = Image.open(record.rgb_path).convert("RGB")
        original_size = rgb_img.size
        depth_img = Image.open(record.depth_path)

        rgb_img = rgb_img.resize(
            (self.image_size, self.image_size),
            resample=Image.Resampling.BILINEAR,
        )
        depth_img = depth_img.resize(
            (self.image_size, self.image_size),
            resample=Image.Resampling.BILINEAR,
        )

        rgb = np.asarray(rgb_img, dtype=np.float32) / 255.0
        depth = np.asarray(depth_img, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth = _normalize_depth(depth)
        return rgb, depth, original_size

    def _build_input_tensor(self, rgb: np.ndarray, depth: np.ndarray) -> torch.Tensor:
        rgb_chw = np.transpose((rgb - 0.5) / 0.5, (2, 0, 1))
        depth_chw = ((depth - 0.5) / 0.5)[None, ...]

        if self.input_mode == "rgb":
            image = rgb_chw
        elif self.input_mode == "depth":
            image = depth_chw
        else:
            image = np.concatenate([rgb_chw, depth_chw], axis=0)
        return torch.from_numpy(image.astype(np.float32))

    def _build_target_maps(
        self,
        rectangles: list[GraspRectangle],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        quality = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        angle = np.zeros_like(quality)
        width = np.zeros_like(quality)

        for rect in rectangles:
            polygon = [tuple(point) for point in rect.points.tolist()]
            mask_img = Image.new("L", (self.image_size, self.image_size), 0)
            draw = ImageDraw.Draw(mask_img)
            draw.polygon(polygon, fill=1)
            mask = np.asarray(mask_img, dtype=bool)

            quality[mask] = 1.0
            angle[mask] = rect.angle
            width[mask] = np.clip(rect.width / float(self.image_size), 0.0, 1.0)

        return quality, angle, width


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    """Normalize depth to [0, 1] per image using robust percentiles."""

    depth = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    valid = depth[depth > 0]
    if valid.size == 0:
        valid = depth.reshape(-1)
    lo, hi = np.percentile(valid, [1.0, 99.0])
    if hi <= lo:
        return np.zeros_like(depth, dtype=np.float32)
    depth = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    return depth.astype(np.float32)

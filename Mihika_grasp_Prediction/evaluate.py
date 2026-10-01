"""Evaluate grasp rectangle prediction accuracy."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import CornellGraspDataset
from models import GGCNNBaseline
from utils.config import load_config
from utils.geometry import angle_difference, polygon_iou
from utils.prediction import rectangle_from_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate grasp prediction accuracy.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--config", default=None, help="Optional config path. Defaults to checkpoint config.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"], help="Dataset split.")
    return parser.parse_args()


def load_checkpoint_and_config(
    checkpoint_path: str | Path,
    config_path: str | Path | None,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if config_path is not None:
        config = load_config(config_path)
    else:
        config = checkpoint.get("config")
        if config is None:
            raise ValueError("Checkpoint does not contain config. Pass --config explicitly.")
    return checkpoint, config


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint, config = load_checkpoint_and_config(args.checkpoint, args.config, device)

    data_cfg = config["data"]
    dataset = CornellGraspDataset(
        root=data_cfg["root"],
        input_mode=data_cfg.get("input_mode", "rgbd"),
        image_size=int(data_cfg.get("image_size", 300)),
        split=args.split,
        val_split=float(data_cfg.get("val_split", 0.2)),
        test_split=float(data_cfg.get("test_split", 0.15)),
        seed=int(config.get("seed", 42)),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = GGCNNBaseline(in_channels=dataset.in_channels).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    eval_cfg = config.get("evaluation", {})
    iou_threshold = float(eval_cfg.get("iou_threshold", 0.25))
    angle_threshold = math.radians(float(eval_cfg.get("angle_threshold_degrees", 30.0)))

    correct = 0
    total = 0
    for batch in tqdm(loader, desc="evaluate"):
        index = int(batch["index"].item())
        image = batch["image"].to(device)
        with torch.no_grad():
            outputs = model(image)
        pred_rect, pred_angle = rectangle_from_output(outputs, dataset.image_size)

        matched = False
        for gt_rect in dataset.get_rectangles(index):
            iou = polygon_iou(pred_rect, gt_rect.points)
            angle_ok = angle_difference(pred_angle, gt_rect.angle) <= angle_threshold
            if iou >= iou_threshold and angle_ok:
                matched = True
                break

        correct += int(matched)
        total += 1

    accuracy = correct / max(total, 1)
    print(f"samples={total}")
    print(f"correct={correct}")
    print(f"accuracy={accuracy:.4f}")
    print(f"iou_threshold={iou_threshold:.2f}")
    print(f"angle_threshold_degrees={math.degrees(angle_threshold):.1f}")


if __name__ == "__main__":
    main()

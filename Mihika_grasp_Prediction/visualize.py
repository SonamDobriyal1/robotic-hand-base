"""Visualize predicted grasp rectangles over RGB images."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from data import CornellGraspDataset
from models import GGCNNBaseline
from utils.config import load_config
from utils.prediction import grasp_from_output
from utils.geometry import rectangle_from_grasp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize one predicted grasp.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--config", default=None, help="Optional config path. Defaults to checkpoint config.")
    parser.add_argument("--sample_id", type=int, default=0, help="Index within the selected split.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test", "all"], help="Dataset split.")
    parser.add_argument("--output", default=None, help="Output PNG path.")
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


def draw_rectangle(ax: plt.Axes, points: np.ndarray, color: str, label: str) -> None:
    closed = np.vstack([points, points[0]])
    ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=2, label=label)


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
    if args.sample_id < 0 or args.sample_id >= len(dataset):
        raise IndexError(f"sample_id must be in [0, {len(dataset) - 1}]")

    model = GGCNNBaseline(in_channels=dataset.in_channels).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    sample = dataset[args.sample_id]
    image_tensor = sample["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)

    center_x, center_y, angle, width_pixels, score = grasp_from_output(outputs, dataset.image_size)
    pred_rect = rectangle_from_grasp(center_x, center_y, angle, width_pixels)

    rgb = np.asarray(dataset.get_rgb_image(args.sample_id))
    output_path = Path(args.output) if args.output else Path("outputs") / f"prediction_sample_{args.sample_id}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(rgb)
    draw_rectangle(ax, pred_rect, color="red", label=f"prediction score={score:.2f}")

    gt_rects = dataset.get_rectangles(args.sample_id)
    for idx, gt_rect in enumerate(gt_rects):
        draw_rectangle(ax, gt_rect.points, color="lime", label="ground truth" if idx == 0 else "_nolegend_")

    ax.scatter([center_x], [center_y], c="red", s=20)
    ax.axis("off")
    ax.legend(loc="lower right")
    fig.tight_layout(pad=0)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main()

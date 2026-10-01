"""Minimal smoke test for model shapes and loss backpropagation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from models import GGCNNBaseline
from train import compute_loss


def main() -> None:
    torch.manual_seed(0)
    model = GGCNNBaseline(in_channels=4, base_channels=8)
    batch = {
        "image": torch.randn(2, 4, 64, 64),
        "quality": torch.zeros(2, 1, 64, 64),
        "angle": torch.zeros(2, 1, 64, 64),
        "width": torch.zeros(2, 1, 64, 64),
    }
    batch["quality"][:, :, 24:40, 24:40] = 1.0
    batch["angle"][:, :, 24:40, 24:40] = 0.25
    batch["width"][:, :, 24:40, 24:40] = 0.2

    outputs = model(batch["image"])
    assert outputs["quality"].shape == batch["quality"].shape
    assert outputs["angle"].shape == batch["angle"].shape
    assert outputs["width"].shape == batch["width"].shape

    loss, metrics = compute_loss(
        outputs,
        batch,
        {"quality_weight": 1.0, "angle_weight": 1.0, "width_weight": 0.5},
    )
    assert torch.isfinite(loss)
    loss.backward()
    print("smoke test passed")
    print(metrics)


if __name__ == "__main__":
    main()

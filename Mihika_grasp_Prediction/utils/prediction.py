"""Prediction helpers shared by evaluation and visualization."""

from __future__ import annotations

import torch

from utils.geometry import rectangle_from_grasp


@torch.no_grad()
def grasp_from_output(
    outputs: dict[str, torch.Tensor],
    image_size: int,
) -> tuple[float, float, float, float, float]:
    """Convert model maps into one grasp rectangle proposal.

    Returns ``(center_x, center_y, angle, width_pixels, quality_score)``.
    """

    quality = torch.sigmoid(outputs["quality"])[0, 0]
    flat_index = int(torch.argmax(quality).item())
    y = flat_index // quality.shape[1]
    x = flat_index % quality.shape[1]

    angle = float(outputs["angle"][0, 0, y, x].item())
    width_norm = float(torch.sigmoid(outputs["width"])[0, 0, y, x].item())
    width_pixels = max(width_norm * image_size, 1.0)
    score = float(quality[y, x].item())
    return float(x), float(y), angle, width_pixels, score


def rectangle_from_output(outputs: dict[str, torch.Tensor], image_size: int):
    center_x, center_y, angle, width_pixels, _ = grasp_from_output(outputs, image_size)
    return rectangle_from_grasp(center_x, center_y, angle, width_pixels), angle

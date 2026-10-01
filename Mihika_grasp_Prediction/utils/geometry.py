"""Geometry helpers for oriented grasp rectangles."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def angle_difference(a: float, b: float) -> float:
    """Smallest difference between antipodal grasp angles in radians."""

    return abs((a - b + math.pi / 2) % math.pi - math.pi / 2)


def rectangle_from_grasp(
    center_x: float,
    center_y: float,
    angle: float,
    width: float,
    height: float | None = None,
) -> np.ndarray:
    """Create a 4-point oriented rectangle from grasp parameters.

    The model predicts gripper opening width but not rectangle thickness. For
    evaluation and visualization, we use a practical default thickness of half
    the opening width.
    """

    width = max(float(width), 1.0)
    height = max(float(height if height is not None else width * 0.5), 1.0)
    center = np.array([center_x, center_y], dtype=np.float32)
    axis = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
    normal = np.array([-math.sin(angle), math.cos(angle)], dtype=np.float32)

    half_w = width / 2.0
    half_h = height / 2.0
    return np.stack(
        [
            center - axis * half_w - normal * half_h,
            center + axis * half_w - normal * half_h,
            center + axis * half_w + normal * half_h,
            center - axis * half_w + normal * half_h,
        ]
    ).astype(np.float32)


def polygon_iou(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    """Intersection-over-union for two convex polygons."""

    area_a = abs(polygon_area(poly_a))
    area_b = abs(polygon_area(poly_b))
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0

    intersection = polygon_clip(poly_a, poly_b)
    if len(intersection) < 3:
        return 0.0
    area_intersection = abs(polygon_area(np.asarray(intersection, dtype=np.float32)))
    union = area_a + area_b - area_intersection
    if union <= 0.0:
        return 0.0
    return float(area_intersection / union)


def polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def polygon_clip(subject_polygon: np.ndarray, clip_polygon: np.ndarray) -> list[np.ndarray]:
    """Sutherland-Hodgman clipping for convex polygons."""

    output = [np.asarray(p, dtype=np.float32) for p in subject_polygon]
    clip_points = [np.asarray(p, dtype=np.float32) for p in clip_polygon]
    clip_is_ccw = polygon_area(np.asarray(clip_points, dtype=np.float32)) >= 0

    for edge_start, edge_end in _edges(clip_points):
        input_list = output
        output = []
        if not input_list:
            break

        previous = input_list[-1]
        for current in input_list:
            current_inside = _inside(current, edge_start, edge_end, clip_is_ccw)
            previous_inside = _inside(previous, edge_start, edge_end, clip_is_ccw)
            if current_inside:
                if not previous_inside:
                    output.append(_segment_intersection(previous, current, edge_start, edge_end))
                output.append(current)
            elif previous_inside:
                output.append(_segment_intersection(previous, current, edge_start, edge_end))
            previous = current

    return output


def _edges(points: Iterable[np.ndarray]) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    points = list(points)
    for idx, point in enumerate(points):
        yield point, points[(idx + 1) % len(points)]


def _inside(point: np.ndarray, edge_start: np.ndarray, edge_end: np.ndarray, ccw: bool) -> bool:
    cross = _cross2d(edge_end - edge_start, point - edge_start)
    return bool(cross >= -1e-6) if ccw else bool(cross <= 1e-6)


def _cross2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _segment_intersection(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray,
) -> np.ndarray:
    d1 = p2 - p1
    d2 = p4 - p3
    denominator = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(float(denominator)) < 1e-8:
        return p2
    delta = p3 - p1
    t = (delta[0] * d2[1] - delta[1] * d2[0]) / denominator
    return (p1 + t * d1).astype(np.float32)

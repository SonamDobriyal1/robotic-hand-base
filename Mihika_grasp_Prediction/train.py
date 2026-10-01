"""Train a baseline RGB-D grasp prediction model."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import CornellGraspDataset
from models import GGCNNBaseline
from utils.config import load_config, save_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train grasp prediction model.")
    parser.add_argument("--config", default="configs/baseline.yaml", help="Path to YAML config.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    data_cfg = config["data"]
    seed = int(config.get("seed", 42))
    train_dataset = CornellGraspDataset(
        root=data_cfg["root"],
        input_mode=data_cfg.get("input_mode", "rgbd"),
        image_size=int(data_cfg.get("image_size", 300)),
        split="train",
        val_split=float(data_cfg.get("val_split", 0.2)),
        test_split=float(data_cfg.get("test_split", 0.15)),
        seed=seed,
    )
    val_dataset = CornellGraspDataset(
        root=data_cfg["root"],
        input_mode=data_cfg.get("input_mode", "rgbd"),
        image_size=int(data_cfg.get("image_size", 300)),
        split="val",
        val_split=float(data_cfg.get("val_split", 0.2)),
        test_split=float(data_cfg.get("test_split", 0.15)),
        seed=seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"].get("batch_size", 8)),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["training"].get("batch_size", 8)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute quality, angle, and width losses.

    Angle and width are supervised only where a ground-truth grasp rectangle is
    present. This keeps background pixels from dominating the regression terms.
    """

    quality_target = batch["quality"]
    angle_target = batch["angle"]
    width_target = batch["width"]
    grasp_mask = quality_target > 0.5

    quality_loss = F.binary_cross_entropy_with_logits(outputs["quality"], quality_target)
    if grasp_mask.any():
        angle_loss = F.smooth_l1_loss(outputs["angle"][grasp_mask], angle_target[grasp_mask])
        width_loss = F.smooth_l1_loss(torch.sigmoid(outputs["width"])[grasp_mask], width_target[grasp_mask])
    else:
        angle_loss = outputs["angle"].sum() * 0.0
        width_loss = outputs["width"].sum() * 0.0

    loss = (
        float(weights.get("quality_weight", 1.0)) * quality_loss
        + float(weights.get("angle_weight", 1.0)) * angle_loss
        + float(weights.get("width_weight", 0.5)) * width_loss
    )
    metrics = {
        "loss": float(loss.detach().cpu().item()),
        "quality_loss": float(quality_loss.detach().cpu().item()),
        "angle_loss": float(angle_loss.detach().cpu().item()),
        "width_loss": float(width_loss.detach().cpu().item()),
    }
    return loss, metrics


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    weights: dict[str, float],
    desc: str,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals = {"loss": 0.0, "quality_loss": 0.0, "angle_loss": 0.0, "width_loss": 0.0}
    count = 0

    for batch in tqdm(loader, desc=desc, leave=False):
        batch = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        with torch.set_grad_enabled(is_train):
            outputs = model(batch["image"])
            loss, metrics = compute_loss(outputs, batch, weights)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = int(batch["image"].shape[0])
        count += batch_size
        for key in totals:
            totals[key] += metrics[key] * batch_size

    return {key: value / max(count, 1) for key, value in totals.items()}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
    best_val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    requested_device = str(config["training"].get("device", "auto")).lower()
    if requested_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA training was requested, but PyTorch cannot access a CUDA GPU. "
            "Install a CUDA-enabled PyTorch build and check your GPU drivers."
        )
    print(f"Training device: {device}")
    train_loader, val_loader = build_dataloaders(config)
    in_channels = train_loader.dataset.in_channels
    model = GGCNNBaseline(in_channels=in_channels).to(device)

    training_cfg = config["training"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-5)),
    )

    checkpoint_dir = Path(training_cfg.get("checkpoint_dir", "checkpoints"))
    log_path = Path(training_cfg.get("log_path", "outputs/train_log.csv"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    save_config(config, checkpoint_dir / "config.yaml")

    best_val_loss = float("inf")
    epochs = int(training_cfg.get("epochs", 20))
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_quality_loss",
                "train_angle_loss",
                "train_width_loss",
                "val_loss",
                "val_quality_loss",
                "val_angle_loss",
                "val_width_loss",
            ],
        )
        writer.writeheader()

        for epoch in range(1, epochs + 1):
            train_metrics = run_epoch(
                model,
                train_loader,
                optimizer,
                device,
                config.get("loss", {}),
                desc=f"train {epoch}/{epochs}",
            )
            with torch.no_grad():
                val_metrics = run_epoch(
                    model,
                    val_loader,
                    optimizer=None,
                    device=device,
                    weights=config.get("loss", {}),
                    desc=f"val {epoch}/{epochs}",
                )

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_quality_loss": train_metrics["quality_loss"],
                "train_angle_loss": train_metrics["angle_loss"],
                "train_width_loss": train_metrics["width_loss"],
                "val_loss": val_metrics["loss"],
                "val_quality_loss": val_metrics["quality_loss"],
                "val_angle_loss": val_metrics["angle_loss"],
                "val_width_loss": val_metrics["width_loss"],
            }
            writer.writerow(row)
            f.flush()

            print(
                f"epoch={epoch:03d} "
                f"train_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f}"
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                save_checkpoint(
                    checkpoint_dir / "best.pt",
                    model,
                    optimizer,
                    epoch,
                    config,
                    best_val_loss,
                )
            save_checkpoint(
                checkpoint_dir / "last.pt",
                model,
                optimizer,
                epoch,
                config,
                best_val_loss,
            )

    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {checkpoint_dir / 'best.pt'}")


if __name__ == "__main__":
    main()

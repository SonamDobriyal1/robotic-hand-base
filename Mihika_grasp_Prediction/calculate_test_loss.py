"""Calculate test loss for a checkpoint and plot it against training history."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import CornellGraspDataset
from evaluate import load_checkpoint_and_config
from models import GGCNNBaseline
from train import compute_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--config", default=None, help="Defaults to the checkpoint's saved config.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--log", default="outputs/train_log.csv", help="Training history CSV.")
    parser.add_argument("--plot", default=None, help="Output image; defaults to outputs/<split>_loss.png.")
    return parser.parse_args()


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
        val_split=float(data_cfg.get("val_split", 0.15)),
        test_split=float(data_cfg.get("test_split", 0.15)),
        seed=int(config.get("seed", 42)),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = GGCNNBaseline(in_channels=dataset.in_channels).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    totals = {"loss": 0.0, "quality_loss": 0.0, "angle_loss": 0.0, "width_loss": 0.0}
    for batch in tqdm(loader, desc=f"{args.split} loss"):
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        with torch.no_grad():
            _, metrics = compute_loss(model(batch["image"]), batch, config.get("loss", {}))
        for name, value in metrics.items():
            totals[name] += value

    means = {name: value / max(len(dataset), 1) for name, value in totals.items()}
    print(f"split={args.split}")
    print(f"samples={len(dataset)}")
    for name, value in means.items():
        print(f"{args.split}_{name}={value:.6f}")

    epochs: list[int] = []
    train_losses: list[float] = []
    val_losses: list[float] = []
    log_path = Path(args.log)
    if log_path.exists():
        with log_path.open(newline="", encoding="utf-8") as csv_file:
            for row in csv.DictReader(csv_file):
                epochs.append(int(row["epoch"]))
                train_losses.append(float(row["train_loss"]))
                val_losses.append(float(row["val_loss"]))

    plot_path = Path(args.plot) if args.plot else Path("outputs") / f"{args.split}_loss.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    if epochs:
        axes[0].plot(epochs, train_losses, label="Train loss")
        axes[0].plot(epochs, val_losses, label="Validation loss")
        axes[0].axhline(means["loss"], color="tab:red", linestyle="--", label=f"{args.split.title()} loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Weighted loss")
        axes[0].set_title("Loss over training epochs")
        axes[0].legend()
        axes[0].grid(alpha=0.25)
    else:
        axes[0].text(0.5, 0.5, f"Training log not found:\n{log_path}", ha="center", va="center")
        axes[0].set_axis_off()

    component_names = ["quality_loss", "angle_loss", "width_loss"]
    axes[1].bar(["Total", "Quality", "Angle", "Width"], [means[name] for name in ["loss", *component_names]])
    axes[1].set_ylabel("Mean loss")
    axes[1].set_title(f"{args.split.title()} loss breakdown")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle(f"Checkpoint epoch {checkpoint.get('epoch', 'unknown')} | {len(dataset)} {args.split} samples")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"plot={plot_path}")


if __name__ == "__main__":
    main()

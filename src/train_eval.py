# src/train_eval.py

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import yaml

from src.dataset import EgoFallsDataset
from src.model import build_model


# -------------------- Config helpers -------------------- #

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# -------------------- Split & DataLoaders -------------------- #

def build_splits(cfg: dict, n_samples: int) -> Dict[str, np.ndarray]:
    split_cfg = cfg["split"]["random"]
    seed = cfg["reproducibility"]["seed"]

    indices_file = Path(split_cfg["indices_file"])
    indices_file.parent.mkdir(parents=True, exist_ok=True)

    if indices_file.exists():
        arr = np.load(indices_file)
        return {
            "train": arr["train_indices"],
            "val": arr["val_indices"],
            "test": arr["test_indices"],
        }

    rng = np.random.RandomState(seed)
    all_indices = np.arange(n_samples)
    rng.shuffle(all_indices)

    train_frac = split_cfg["train_frac"]
    val_frac = split_cfg["val_frac"]
    test_frac = split_cfg["test_frac"]
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1."

    n_train = int(n_samples * train_frac)
    n_val = int(n_samples * val_frac)
    n_test = n_samples - n_train - n_val

    train_indices = all_indices[:n_train]
    val_indices = all_indices[n_train : n_train + n_val]
    test_indices = all_indices[n_train + n_val :]

    np.savez(
        indices_file,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
    )

    return {"train": train_indices, "val": val_indices, "test": test_indices}


def make_dataloaders(cfg: dict, splits: Dict[str, np.ndarray]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    paths_cfg = cfg["paths"]
    train_cfg = cfg["train"]

    metadata_subset = paths_cfg["metadata_subset"]
    batch_size = int(train_cfg["batch_size"])
    num_workers = int(train_cfg["num_workers"])

    base_ds = EgoFallsDataset(
        metadata_path=metadata_subset,
        config_path="configs/config.yaml",
    )

    train_ds = Subset(base_ds, splits["train"])
    val_ds = Subset(base_ds, splits["val"])
    test_ds = Subset(base_ds, splits["test"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# -------------------- Metrics -------------------- #

def compute_confusion_and_metrics(
    all_labels: np.ndarray,
    all_preds: np.ndarray,
    positive_label: int = 1,
) -> Dict[str, float]:
    """
    Compute accuracy + precision/recall/F1 for the positive (fall) class.
    """
    assert all_labels.shape == all_preds.shape
    num_classes = 2
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for y_true, y_pred in zip(all_labels, all_preds):
        cm[y_true, y_pred] += 1

    total = cm.sum()
    acc = cm.trace() / total if total > 0 else 0.0

    p = positive_label
    tp = cm[p, p]
    fn = cm[p, 1 - p]
    fp = cm[1 - p, p]
    tn = cm[1 - p, 1 - p]

    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return {
        "accuracy": acc,
        "precision_pos": precision,
        "recall_pos": recall,
        "f1_pos": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


# -------------------- Train / Eval loops -------------------- #

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: nn.Module = None,
    train: bool = True,
    log_interval: int = 10,
) -> Dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()

    all_labels = []
    all_preds = []
    running_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="train" if train else "eval", leave=False)
    for batch_idx, batch in enumerate(pbar):
        frames, audio_feats, labels = batch
        frames = frames.to(device, non_blocking=True)
        audio_feats = audio_feats.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(video_frames=frames, audio_feats=audio_feats)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

        running_loss += loss.item()
        n_batches += 1

        preds = torch.argmax(logits, dim=1)
        all_labels.append(labels.detach().cpu().numpy())
        all_preds.append(preds.detach().cpu().numpy())

        if (batch_idx + 1) % log_interval == 0:
            pbar.set_postfix(loss=running_loss / n_batches)

    avg_loss = running_loss / max(n_batches, 1)
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    metrics = compute_confusion_and_metrics(all_labels, all_preds, positive_label=1)
    metrics["loss"] = avg_loss
    return metrics


# -------------------- Main training function -------------------- #

def train_and_eval(config_path: str, mode: str) -> None:
    cfg = load_config(config_path)

    # --- Device ---
    train_cfg = cfg["train"]
    if train_cfg["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(train_cfg["device"])

    # --- Data splits & loaders ---
    subset_path = cfg["paths"]["metadata_subset"]
    n_samples = sum(1 for _ in open(subset_path)) - 1  # minus header
    splits = build_splits(cfg, n_samples)
    train_loader, val_loader, test_loader = make_dataloaders(cfg, splits)

    # --- Model ---
    model, _ = build_model(config_path=config_path, mode=mode, device=device)

    # Only train parameters that require grad (backbone may be frozen)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        params,
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()

    max_epochs = int(train_cfg["max_epochs"])
    log_interval = int(train_cfg.get("log_interval", 10))

    ckpt_dir = Path(cfg["paths"].get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"best_{mode}.pt"

    best_val_f1 = -1.0
    best_state = None

    print(f"\n=== Training mode: {mode} ===")
    print(f"Device: {device}")
    print(f"Train/Val/Test sizes: {len(train_loader.dataset)}, {len(val_loader.dataset)}, {len(test_loader.dataset)}")

    for epoch in range(1, max_epochs + 1):
        print(f"\nEpoch {epoch}/{max_epochs}")

        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            criterion,
            optimizer=optimizer,
            train=True,
            log_interval=log_interval,
        )
        print(
            f"  Train | loss={train_metrics['loss']:.4f} "
            f"acc={train_metrics['accuracy']:.3f} "
            f"F1(fall)={train_metrics['f1_pos']:.3f}"
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            criterion,
            optimizer=None,
            train=False,
            log_interval=log_interval,
        )
        print(
            f"  Val   | loss={val_metrics['loss']:.4f} "
            f"acc={val_metrics['accuracy']:.3f} "
            f"F1(fall)={val_metrics['f1_pos']:.3f}"
        )

        if val_metrics["f1_pos"] > best_val_f1:
            best_val_f1 = val_metrics["f1_pos"]
            best_state = {
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
                "mode": mode,
            }
            torch.save(best_state, ckpt_path)
            print(f"  [*] New best model saved to {ckpt_path} (F1={best_val_f1:.3f})")

    # --- Load best model and evaluate on test set ---
    if best_state is None and ckpt_path.exists():
        best_state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(best_state["model_state"])
        print(f"\nLoaded best checkpoint from {ckpt_path}")
    elif best_state is not None:
        model.load_state_dict(best_state["model_state"])

    print("\n=== Final Test Evaluation ===")
    test_metrics = run_epoch(
        model,
        test_loader,
        device,
        criterion,
        optimizer=None,
        train=False,
        log_interval=log_interval,
    )
    print(
        f"Test | loss={test_metrics['loss']:.4f} "
        f"acc={test_metrics['accuracy']:.3f} "
        f"F1(fall)={test_metrics['f1_pos']:.3f} "
        f"recall(fall)={test_metrics['recall_pos']:.3f} "
        f"precision(fall)={test_metrics['precision_pos']:.3f}"
    )
    print(
        f"Confusion matrix counts (true rows, pred cols): "
        f"TP={test_metrics['tp']}, FP={test_metrics['fp']}, "
        f"FN={test_metrics['fn']}, TN={test_metrics['tn']}"
    )


# -------------------- CLI -------------------- #

def main():
    parser = argparse.ArgumentParser(description="Train & eval multimodal EGOFALLS classifier.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--mode",
        type=str,
        default="fusion",
        choices=["fusion", "video_only", "audio_only"],
        help="Which model variant to train.",
    )
    args = parser.parse_args()

    train_and_eval(args.config, args.mode)


if __name__ == "__main__":
    main()

"""
start:251pm
end

"""

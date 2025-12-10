# src/viz_results.py
"""
Visualize test results for EGOFALLS multimodal fall detection.

This script:
- Loads metrics_test_<mode>.json for fusion, video_only, audio_only
- Builds a metrics summary table and saves it as CSV
- Plots:
  1) Bar chart of F1(fall) by mode
  2) Grouped bar chart of Recall vs Precision (fall class) by mode
  3) ROC curves for fall class (all modes)
  4) PR curves for fall class (all modes)
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Subset

from src.dataset import EgoFallsDataset
from src.model import build_model

# -------------------- Config / helpers -------------------- #

MODES = ["fusion", "video_only", "audio_only"]


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def choose_device(cfg: dict) -> torch.device:
    train_cfg = cfg["train"]
    dev_str = train_cfg.get("device", "auto")

    if dev_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(dev_str)


def load_splits(cfg: dict) -> Dict[str, np.ndarray]:
    """
    Load train/val/test indices that were saved during training.
    """
    split_cfg = cfg["split"]
    strat = split_cfg.get("strategy", "random")
    if strat != "random":
        raise NotImplementedError("viz_results.py currently assumes random split, not LOSO.")

    random_cfg = split_cfg["random"]
    indices_file = Path(random_cfg["indices_file"])
    if not indices_file.exists():
        raise FileNotFoundError(
            f"Split indices file not found: {indices_file}. "
            "Run src.train_eval at least once to create it."
        )

    arr = np.load(indices_file)
    return {
        "train": arr["train_indices"],
        "val": arr["val_indices"],
        "test": arr["test_indices"],
    }


def make_test_loader(cfg: dict, config_path: str, device: torch.device) -> DataLoader:
    """
    Build a DataLoader for the held-out test split, using metadata_subset and saved indices.
    """
    paths_cfg = cfg["paths"]
    train_cfg = cfg["train"]

    metadata_subset = paths_cfg["metadata_subset"]
    batch_size = int(train_cfg["batch_size"])
    num_workers = int(train_cfg["num_workers"])

    base_ds = EgoFallsDataset(
        metadata_path=metadata_subset,
        config_path=config_path,
    )

    splits = load_splits(cfg)
    test_indices = splits["test"]

    test_ds = Subset(base_ds, test_indices)
    pin_memory = bool(device.type == "cuda")

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return test_loader


def find_checkpoint(cfg: dict, mode: str) -> Path:
    """
    Find checkpoint for a given mode.
    Tries:
      1) paths.checkpoints_dir (e.g. "outputs/checkpoints")
      2) top-level "checkpoints/"
    """
    paths_cfg = cfg["paths"]
    ckpt_name_map = cfg.get("logging", {}).get("checkpoint_names", {})
    ckpt_name = ckpt_name_map.get(mode, f"best_{mode}.pt")

    primary_dir = Path(paths_cfg.get("checkpoints_dir", "outputs/checkpoints"))
    alt_dir = Path("checkpoints")

    for ckpt_dir in (primary_dir, alt_dir):
        candidate = ckpt_dir / ckpt_name
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Checkpoint for mode='{mode}' not found in {primary_dir} or {alt_dir}"
    )


# -------------------- Metrics loading -------------------- #

def load_metrics_for_modes(metrics_dir: Path) -> pd.DataFrame:
    """
    Load metrics_test_<mode>.json files and build a summary table.
    """
    rows = []
    for mode in MODES:
        path = metrics_dir / f"metrics_test_{mode}.json"
        if not path.exists():
            raise FileNotFoundError(f"Metrics JSON not found: {path}")
        with path.open("r") as f:
            m = json.load(f)

        rows.append(
            {
                "mode": mode,
                "loss": m.get("loss", np.nan),
                "accuracy": m.get("accuracy", np.nan),
                "f1_fall": m.get("f1_pos", np.nan),
                "recall_fall": m.get("recall_pos", np.nan),
                "precision_fall": m.get("precision_pos", np.nan),
                "roc_auc_fall": m.get("roc_auc_fall", np.nan),
                "tp": m.get("tp", np.nan),
                "fp": m.get("fp", np.nan),
                "fn": m.get("fn", np.nan),
                "tn": m.get("tn", np.nan),
            }
        )

    df = pd.DataFrame(rows)
    return df


# -------------------- Prediction collection -------------------- #

def get_test_predictions(
    cfg: dict,
    config_path: str,
    mode: str,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the trained model for the given mode on the test set and return:
    - labels_np: ground-truth labels (0/1)
    - probs_pos_np: predicted probability of fall (class 1)
    """
    test_loader = make_test_loader(cfg, config_path, device)
    ckpt_path = find_checkpoint(cfg, mode)

    print(f"[viz] Mode '{mode}': loading checkpoint from {ckpt_path}")
    model, _ = build_model(config_path=config_path, mode=mode, device=device)

    # PyTorch 2.6: weights_only=False to allow full checkpoint dict
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()

    all_labels: List[np.ndarray] = []
    all_probs_pos: List[np.ndarray] = []

    with torch.no_grad():
        for frames, audio_feats, labels in test_loader:
            frames = frames.to(device, non_blocking=True)
            audio_feats = audio_feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(video_frames=frames, audio_feats=audio_feats)
            probs = torch.softmax(logits, dim=1)
            probs_pos = probs[:, 1]  # probability of fall (label 1)

            all_labels.append(labels.cpu().numpy())
            all_probs_pos.append(probs_pos.cpu().numpy())

    labels_np = np.concatenate(all_labels, axis=0)
    probs_pos_np = np.concatenate(all_probs_pos, axis=0)
    return labels_np, probs_pos_np


# -------------------- Plotting functions -------------------- #

def plot_f1_bar(df: pd.DataFrame, fig_dir: Path) -> None:
    """
    Bar chart of F1(fall) by mode.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    modes = df["mode"].tolist()
    f1_vals = df["f1_fall"].values

    ax.bar(modes, f1_vals)
    ax.set_ylabel("F1 (fall class)")
    ax.set_title("F1 for Fall Detection by Mode")
    ax.set_ylim(0.0, 1.0)
    for i, v in enumerate(f1_vals):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "bar_f1_fall_by_mode.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[viz] Saved F1 bar chart to {out_path}")


def plot_recall_precision_bar(df: pd.DataFrame, fig_dir: Path) -> None:
    """
    Grouped bar chart: Recall vs Precision (fall class) by mode.
    """
    modes = df["mode"].tolist()
    recall_vals = df["recall_fall"].values
    precision_vals = df["precision_fall"].values

    x = np.arange(len(modes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, recall_vals, width, label="Recall (fall)")
    ax.bar(x + width / 2, precision_vals, width, label="Precision (fall)")

    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel("Score")
    ax.set_title("Recall vs Precision for Fall Class by Mode")
    ax.set_ylim(0.0, 1.0)
    ax.legend()

    for i, v in enumerate(recall_vals):
        ax.text(x[i] - width / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(precision_vals):
        ax.text(x[i] + width / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "bar_recall_precision_fall_by_mode.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[viz] Saved Recall vs Precision bar chart to {out_path}")


def plot_roc_curves(
    labels_by_mode: Dict[str, np.ndarray],
    probs_by_mode: Dict[str, np.ndarray],
    fig_dir: Path,
) -> None:
    """
    ROC curves for fall class across modes.
    """
    fig, ax = plt.subplots(figsize=(5, 5))

    for mode in MODES:
        y_true = labels_by_mode[mode]
        y_score = probs_by_mode[mode]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{mode} (AUC={roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--", label="Random")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curves – Fall vs Non-fall")
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "roc_curves_fall_all_modes.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[viz] Saved ROC curves to {out_path}")


def plot_pr_curves(
    labels_by_mode: Dict[str, np.ndarray],
    probs_by_mode: Dict[str, np.ndarray],
    fig_dir: Path,
) -> None:
    """
    PR curves for fall class across modes.
    """
    fig, ax = plt.subplots(figsize=(5, 5))

    for mode in MODES:
        y_true = labels_by_mode[mode]
        y_score = probs_by_mode[mode]
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, label=f"{mode} (AUC={pr_auc:.2f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curves – Fall Class")
    ax.legend(loc="lower left")

    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "pr_curves_fall_all_modes.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[viz] Saved PR curves to {out_path}")


# -------------------- Main -------------------- #

def main():
    config_path = "configs/config.yaml"
    cfg = load_config(config_path)
    device = choose_device(cfg)
    print(f"[viz] Using device: {device}")

    metrics_dir = Path(cfg["paths"].get("logs_dir", "outputs/logs"))
    fig_dir = Path(cfg["paths"].get("figures_dir", "outputs/figures"))

    # 1) Load metrics JSONs and build summary table
    df = load_metrics_for_modes(metrics_dir)
    print("\n=== Test Metrics Summary (per mode) ===")
    print(df.to_string(index=False))

    # Save table as CSV
    summary_path = metrics_dir / "metrics_test_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"[viz] Saved metrics summary CSV to {summary_path}")

    # 2) Plots based on summary metrics
    plot_f1_bar(df, fig_dir)
    plot_recall_precision_bar(df, fig_dir)

    # 3) ROC & PR curves – need per-example probabilities
    labels_by_mode: Dict[str, np.ndarray] = {}
    probs_by_mode: Dict[str, np.ndarray] = {}

    for mode in MODES:
        labels_np, probs_pos_np = get_test_predictions(cfg, config_path, mode, device)
        labels_by_mode[mode] = labels_np
        probs_by_mode[mode] = probs_pos_np

    plot_roc_curves(labels_by_mode, probs_by_mode, fig_dir)
    plot_pr_curves(labels_by_mode, probs_by_mode, fig_dir)

    print("\n[viz] Done. Plots saved to:", fig_dir)


if __name__ == "__main__":
    main()

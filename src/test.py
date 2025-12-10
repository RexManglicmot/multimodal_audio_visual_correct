# src/test.py
"""
Evaluate a trained EGOFALLS model (fusion / video_only / audio_only)
on the held-out test set.

- Reuses the same metadata_subset and saved split indices
- Loads the best_* checkpoint for the chosen mode
- Computes accuracy, precision/recall/F1 for the fall class
- Optionally computes ROC–AUC and saves a confusion matrix figure
- Saves a metrics JSON per mode under outputs/logs/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Subset

from sklearn.metrics import roc_auc_score, precision_recall_curve

import matplotlib.pyplot as plt
from tqdm.auto import tqdm  # progress bar

from src.dataset import EgoFallsDataset
from src.model import build_model


# -------------------- Config / helpers -------------------- #

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


def load_splits(cfg: dict, n_samples: int) -> Dict[str, np.ndarray]:
    """
    Load train/val/test indices that were saved during training.
    """
    split_cfg = cfg["split"]
    strat = split_cfg.get("strategy", "random")
    if strat != "random":
        raise NotImplementedError("test.py currently assumes random split, not LOSO.")

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


def make_test_loader(cfg: dict, config_path: str, device: torch.device) -> Tuple[DataLoader, int]:
    """
    Build a DataLoader for the held-out test split, using metadata_subset and saved indices.
    """
    paths_cfg = cfg["paths"]
    train_cfg = cfg["train"]

    metadata_subset = paths_cfg["metadata_subset"]
    batch_size = int(train_cfg["batch_size"])
    num_workers = int(train_cfg["num_workers"])

    # Full subset dataset
    base_ds = EgoFallsDataset(
        metadata_path=metadata_subset,
        config_path=config_path,
    )

    n_samples = len(base_ds)
    splits = load_splits(cfg, n_samples)
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
    return test_loader, len(test_indices)


# -------------------- Metrics / plotting -------------------- #

def compute_confusion_and_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    positive_label: int = 1,
) -> Dict[str, float]:
    """
    Compute confusion matrix + accuracy + precision/recall/F1 for the positive class.
    """
    assert labels.shape == preds.shape
    num_classes = 2
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for y_true, y_pred in zip(labels, preds):
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


def plot_confusion_matrix(
    cm: np.ndarray,
    mode: str,
    out_path: Path,
    class_names: Tuple[str, str] = ("non-fall", "fall"),
) -> None:
    """
    Save a simple 2x2 confusion matrix heatmap.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(f"Confusion Matrix (test) – {mode}")
    plt.colorbar(im, ax=ax)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    # Print counts inside the cells
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_pr_curve(
    labels: np.ndarray,
    probs_pos: np.ndarray,
    mode: str,
    out_path: Path,
) -> None:
    """
    Save a precision–recall curve for the fall class (label=1).
    """
    precision, recall, _ = precision_recall_curve(labels, probs_pos)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(recall, precision)
    ax.set_title(f"PR curve (fall class) – {mode}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# -------------------- Core evaluation -------------------- #

def evaluate_mode_on_test(
    cfg: dict,
    config_path: str,
    mode: str,
    device: torch.device,
) -> Dict[str, float]:
    """
    Load the best checkpoint for the given mode and evaluate on the test split.
    """
    eval_cfg = cfg.get("eval", {})
    paths_cfg = cfg["paths"]

    # Build test loader
    test_loader, n_test = make_test_loader(cfg, config_path, device)
    print(f"[test] Mode '{mode}': evaluating on {n_test} test clips.")

    # Build model
    model, _ = build_model(config_path=config_path, mode=mode, device=device)

    # We support two possible locations:
    # 1) paths.checkpoints_dir from config (e.g. "outputs/checkpoints")
    # 2) top-level "checkpoints/" (which you already have on Vast)
    ckpt_name_map = cfg.get("logging", {}).get("checkpoint_names", {})
    ckpt_name = ckpt_name_map.get(mode, f"best_{mode}.pt")

    primary_dir = Path(paths_cfg.get("checkpoints_dir", "outputs/checkpoints"))
    alt_dir = Path("checkpoints")

    ckpt_path = None
    for ckpt_dir in (primary_dir, alt_dir):
        candidate = ckpt_dir / ckpt_name
        if candidate.exists():
            ckpt_path = candidate
            break

    if ckpt_path is None:
        raise FileNotFoundError(
            f"Checkpoint for mode='{mode}' not found in {primary_dir} or {alt_dir}"
        )

    print(f"[test] Loading checkpoint for mode='{mode}' from {ckpt_path}")
    # PyTorch 2.6: explicitly allow full checkpoint dict
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()

    criterion = nn.CrossEntropyLoss()

    all_labels: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []
    all_probs_pos: List[np.ndarray] = []

    running_loss = 0.0
    n_batches = 0

    # tqdm progress bar over test batches
    with torch.no_grad():
        for frames, audio_feats, labels in tqdm(
            test_loader,
            desc=f"Test ({mode})",
            unit="batch",
            leave=False,
        ):
            frames = frames.to(device, non_blocking=True)
            audio_feats = audio_feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(video_frames=frames, audio_feats=audio_feats)
            loss = criterion(logits, labels)
            running_loss += loss.item()
            n_batches += 1

            probs = torch.softmax(logits, dim=1)
            probs_pos = probs[:, 1]  # probability of fall (label 1)
            preds = torch.argmax(probs, dim=1)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs_pos.append(probs_pos.cpu().numpy())

    avg_loss = running_loss / max(n_batches, 1)
    labels_np = np.concatenate(all_labels, axis=0)
    preds_np = np.concatenate(all_preds, axis=0)
    probs_pos_np = np.concatenate(all_probs_pos, axis=0)

    metrics = compute_confusion_and_metrics(
        labels_np,
        preds_np,
        positive_label=int(eval_cfg.get("positive_label", 1)),
    )
    metrics["loss"] = avg_loss

    # Optional ROC–AUC
    if eval_cfg.get("compute_roc_auc", False):
        try:
            metrics["roc_auc_fall"] = roc_auc_score(labels_np, probs_pos_np)
        except Exception:
            metrics["roc_auc_fall"] = float("nan")

    # Confusion matrix / figures
    if eval_cfg.get("save_confusion_matrix", False):
        num_classes = 2
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for y_true, y_pred in zip(labels_np, preds_np):
            cm[y_true, y_pred] += 1

        base_cm_path = Path(eval_cfg["confusion_matrix_file"])
        cm_path = base_cm_path.with_name(f"{base_cm_path.stem}_{mode}{base_cm_path.suffix}")
        plot_confusion_matrix(cm, mode, cm_path)

    if eval_cfg.get("save_pr_curve", False):
        base_pr_path = Path(eval_cfg["pr_curve_file"])
        pr_path = base_pr_path.with_name(f"{base_pr_path.stem}_{mode}{base_pr_path.suffix}")
        plot_pr_curve(labels_np, probs_pos_np, mode, pr_path)

    return metrics


def save_metrics_json(
    cfg: dict,
    mode: str,
    metrics: Dict[str, float],
) -> None:
    """
    Save test metrics as JSON under outputs/logs/metrics_test_<mode>.json
    """
    logs_dir = Path(cfg["paths"].get("logs_dir", "outputs/logs"))
    logs_dir.mkdir(parents=True, exist_ok=True)
    out_path = logs_dir / f"metrics_test_{mode}.json"

    with out_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[test] Metrics for mode='{mode}' saved to {out_path}")


# -------------------- CLI entrypoint -------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate EGOFALLS model (fusion / video_only / audio_only) on held-out test set."
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--mode",
        type=str,
        default="fusion",
        choices=["fusion", "video_only", "audio_only"],
        help="Which trained model variant to evaluate.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = choose_device(cfg)
    print(f"[test] Using device: {device}")

    metrics = evaluate_mode_on_test(cfg, args.config, args.mode, device)

    print(
        f"\n=== Test Results (mode='{args.mode}') ===\n"
        f"loss={metrics['loss']:.4f} "
        f"acc={metrics['accuracy']:.3f} "
        f"F1(fall)={metrics['f1_pos']:.3f} "
        f"recall(fall)={metrics['recall_pos']:.3f} "
        f"precision(fall)={metrics['precision_pos']:.3f}"
    )
    if "roc_auc_fall" in metrics:
        print(f"ROC–AUC(fall)={metrics['roc_auc_fall']:.3f}")
    print(
        f"Confusion matrix counts (true rows, pred cols): "
        f"TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}, TN={metrics['tn']}"
    )

    save_metrics_json(cfg, args.mode, metrics)


if __name__ == "__main__":
    main()
    
"""
python -m src.test --config configs/config.yaml --mode fusion
python -m src.test --config configs/config.yaml --mode video_only
python -m src.test --config configs/config.yaml --mode audio_only



"""
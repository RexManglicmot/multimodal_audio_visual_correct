# src/prepare_metadata.py

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(config_path: Path) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def infer_and_rename_columns(df: pd.DataFrame, path_col_target: str,
                             label_col_target: str, subject_col_target: str) -> pd.DataFrame:
    """
    Try to infer common column names and rename them to the expected ones
    (path, label, subject_id).
    """
    col_map = {}

    # Path column
    if path_col_target not in df.columns:
        for cand in ["clip_path", "video_path", "filepath", "file_path", "clip"]:
            if cand in df.columns:
                col_map[cand] = path_col_target
                break

    # Label column (binary fall label)
    if label_col_target not in df.columns:
        for cand in ["label", "fall_label", "is_fall", "fall"]:
            if cand in df.columns:
                col_map[cand] = label_col_target
                break

    # Subject ID column
    if subject_col_target not in df.columns:
        for cand in ["subject_id", "participant_id", "actor_id", "person_id"]:
            if cand in df.columns:
                col_map[cand] = subject_col_target
                break

    if col_map:
        df = df.rename(columns=col_map)

    return df


def ensure_required_columns(df: pd.DataFrame, required_cols: list) -> None:
    """Raise a clear error if required columns are missing."""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in annotations file: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def build_metadata_full(cfg: dict) -> pd.DataFrame:
    """
    Build metadata_full.csv from raw annotations.
    If metadata_full already exists, just load and return it.
    """
    paths_cfg = cfg["paths"]
    data_cfg = cfg["data"]

    metadata_full_path = Path(paths_cfg["metadata_full"])
    data_root = Path(paths_cfg["data_root"])

    if metadata_full_path.exists():
        print(f"[prepare_metadata] Found existing {metadata_full_path}, loading it.")
        df_full = pd.read_csv(metadata_full_path)
        return df_full

    # Otherwise, build from raw annotations
    annotations_path = data_root / "annotations.csv"
    if not annotations_path.exists():
        raise FileNotFoundError(
            f"Could not find annotations file at {annotations_path}. "
            f"Please place EGOFALLS annotations as 'annotations.csv' there."
        )

    print(f"[prepare_metadata] Loading raw annotations from {annotations_path}")
    df = pd.read_csv(annotations_path)

    # Infer and rename columns to match config
    path_col = data_cfg["path_col"]
    label_col = data_cfg["label_col"]
    subject_col = data_cfg["subject_id_col"]

    df = infer_and_rename_columns(df, path_col_target=path_col,
                                  label_col_target=label_col,
                                  subject_col_target=subject_col)

    # Ensure we have the required columns
    ensure_required_columns(df, [path_col, label_col, subject_col])

    # Coerce label to int {0, 1}
    df[label_col] = df[label_col].astype(int)

    # (Optional) ensure relative paths, not absolute
    df[path_col] = df[path_col].astype(str)

    metadata_full_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(metadata_full_path, index=False)
    print(f"[prepare_metadata] Saved full metadata to {metadata_full_path} "
          f"with {len(df)} rows.")

    return df


def build_balanced_subset(df_full: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    From the full metadata, build a balanced subset of size `subset_size`,
    stratified by label (0 vs 1).

    If there are not enough samples in one class, automatically reduce the
    subset size to the maximum possible balanced size (2 * min_class_count).
    """
    data_cfg = cfg["data"]
    seed = cfg["reproducibility"]["seed"]
    subset_size_target = int(data_cfg["subset_size"])
    label_col = data_cfg["label_col"]

    # Adjust subset size if full dataset has fewer rows than requested
    if subset_size_target > len(df_full):
        print(f"[prepare_metadata] Requested subset_size={subset_size_target} "
              f"but only {len(df_full)} rows available. "
              f"Reducing subset_size to {len(df_full)}.")
        subset_size_target = len(df_full)

    # If we don't need to balance labels, just sample
    if not data_cfg.get("balance_labels", True):
        subset_df = df_full.sample(
            n=subset_size_target, random_state=seed, replace=False
        ).reset_index(drop=True)
        return subset_df

    # Construct a balanced subset
    counts = df_full[label_col].value_counts().to_dict()
    n_class_0 = counts.get(0, 0)
    n_class_1 = counts.get(1, 0)

    if n_class_0 == 0 or n_class_1 == 0:
        raise ValueError(
            f"[prepare_metadata] Cannot build balanced subset: "
            f"class counts = {counts}. Need both 0 and 1."
        )

    # Max possible balanced subset size
    max_balanced = 2 * min(n_class_0, n_class_1)
    if subset_size_target > max_balanced:
        print(
            f"[prepare_metadata] Requested subset_size={subset_size_target} "
            f"but max balanced size is {max_balanced} "
            f"(class counts: {counts}). "
            f"Reducing subset_size to {max_balanced}."
        )
        subset_size_target = max_balanced

    n_per_class = subset_size_target // 2

    # Sample from each class
    rng = np.random.RandomState(seed)

    df_class0 = df_full[df_full[label_col] == 0]
    df_class1 = df_full[df_full[label_col] == 1]

    subset_0 = df_class0.sample(
        n=n_per_class, random_state=rng.randint(0, 1_000_000), replace=False
    )
    subset_1 = df_class1.sample(
        n=n_per_class, random_state=rng.randint(0, 1_000_000), replace=False
    )

    subset_df = pd.concat([subset_0, subset_1], axis=0)
    # Shuffle the combined subset
    subset_df = subset_df.sample(
        frac=1.0, random_state=seed
    ).reset_index(drop=True)

    print(
        f"[prepare_metadata] Built balanced subset with {len(subset_df)} rows "
        f"({n_per_class} per class)."
    )
    return subset_df


def main():
    parser = argparse.ArgumentParser(
        description="Prepare EGOFALLS metadata_full.csv and metadata_2k.csv"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config YAML file.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = load_config(config_path)

    paths_cfg = cfg["paths"]
    metadata_subset_path = Path(paths_cfg["metadata_subset"])

    # 1) Build or load metadata_full.csv
    df_full = build_metadata_full(cfg)

    # 2) Build balanced subset
    subset_df = build_balanced_subset(df_full, cfg)

    # 3) Save subset
    metadata_subset_path.parent.mkdir(parents=True, exist_ok=True)
    subset_df.to_csv(metadata_subset_path, index=False)
    print(f"[prepare_metadata] Saved subset metadata to {metadata_subset_path}")

    # 4) Quick label distribution printout
    label_col = cfg["data"]["label_col"]
    print("[prepare_metadata] Full label distribution:")
    print(df_full[label_col].value_counts())
    print("[prepare_metadata] Subset label distribution:")
    print(subset_df[label_col].value_counts())


if __name__ == "__main__":
    main()

# src/build_annotations_from_folders.py

import argparse
from pathlib import Path

import pandas as pd


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}  # include MOV!


def infer_label_from_path(path: Path) -> int:
    """
    Infer label from a path by looking for 'fall' / 'falls' / 'nonfall' variants
    anywhere in the full relative path (case-insensitive).

    Rules (this order matters):
      - if we see 'nonfall', 'non-fall', 'nonfalls' -> label 0
      - elif we see 'fall' or 'falls'              -> label 1
      - else: raise so we notice weird cases.
    """
    path_str = str(path).lower()

    if "nonfall" in path_str or "non-fall" in path_str or "nonfalls" in path_str:
        return 0

    if "fall" in path_str or "falls" in path_str:
        return 1

    raise ValueError(f"Could not infer label from path: {path}")


def infer_subject_id(path: Path) -> str:
    """
    Use the top-level directory under data_root as subject/group ID.

    Example:
      S_D_WD/Outdoor/falls/... -> 'S_D_WD'
      S_F/Indoor/falls/...     -> 'S_F'
      S_FI/...                 -> 'S_FI'
    """
    return path.parts[0]


def build_annotations(data_root: Path, output_csv: Path) -> None:
    # collect ALL video files with allowed extensions
    video_paths = []
    for p in data_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            video_paths.append(p)

    if not video_paths:
        raise FileNotFoundError(
            f"No video files with extensions {VIDEO_EXTS} found under {data_root}. "
            f"Check that you extracted the zips correctly."
        )

    records = []
    n_fall_paths = 0
    n_nonfall_paths = 0
    n_failed = 0

    for full_path in sorted(video_paths):
        rel_path = full_path.relative_to(data_root)

        try:
            label = infer_label_from_path(rel_path)
        except ValueError as e:
            print(f"[WARN] {e}")
            n_failed += 1
            continue

        if label == 1:
            n_fall_paths += 1
        else:
            n_nonfall_paths += 1

        subject_id = infer_subject_id(rel_path)

        records.append(
            {
                "path": str(rel_path).replace("\\", "/"),
                "label": int(label),
                "subject_id": subject_id,
            }
        )

    df = pd.DataFrame(records).sort_values("path").reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"[build_annotations] Saved {len(df)} rows to {output_csv}")
    print("[build_annotations] Label distribution:")
    print(df["label"].value_counts())
    print(f"[build_annotations] fall paths (label=1): {n_fall_paths}")
    print(f"[build_annotations] nonfall paths (label=0): {n_nonfall_paths}")
    print(f"[build_annotations] failed to label (skipped): {n_failed}")


def main():
    parser = argparse.ArgumentParser(
        description="Scan EGOFALLS folders and build annotations.csv from falls/nonfalls dirs."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/raw/egofalls",
        help="Root folder containing S_D_WD, S_F, S_FI, S_H, etc.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/egofalls/annotations.csv",
        help="Where to write annotations CSV.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_csv = Path(args.output)

    build_annotations(data_root, output_csv)


if __name__ == "__main__":
    main()

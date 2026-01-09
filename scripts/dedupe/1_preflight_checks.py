"""Quick pre-flight checks to run before training.

These checks are intentionally lightweight so they can be executed right before
launching model training to validate that the data splits are in place and
consistent.

Usage
------
python scripts/preflight_checks.py \
    --train dataset/splits/train.csv \
    --val dataset/splits/validation.csv \
    --test dataset/splits/test.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

CRAIG_PREFIX = "_craig"
US_PREFIX = ""

REQUIRED_BASE_COLS = [
    "source_id_craig",
    "source_id_us",
    "brand",
    "model",
    "year",
    "price",
    "mileage",
    "city_region",
    "state",
]


def _require_columns(df: pd.DataFrame, cols: Sequence[str], split_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"[{split_name}] missing columns: {missing}")


def _check_non_empty(df: pd.DataFrame, split_name: str) -> None:
    if df.empty:
        raise SystemExit(f"[{split_name}] is empty")


def _check_duplicates(df: pd.DataFrame, id_col: str, split_name: str) -> None:
    """Warn (not fail) if duplicates found – duplicates are expected in record linkage."""
    dupes = df[df[id_col].duplicated()][id_col].nunique()
    if dupes:
        print(f"[{split_name}] info: {dupes} duplicate values in {id_col} (expected for record linkage)")


def _check_overlap(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    key_cols = ["source_id_craig", "source_id_us"]
    train_pairs = set(tuple(row) for row in train[key_cols].itertuples(index=False, name=None))
    for name, df in {"validation": val, "test": test}.items():
        overlap = train_pairs & set(tuple(row) for row in df[key_cols].itertuples(index=False, name=None))
        if overlap:
            raise SystemExit(f"Train/{name} overlap detected: {len(overlap)} pairs re-used")


def _check_nulls(df: pd.DataFrame, columns: Iterable[str], split_name: str) -> None:
    """Warn about nulls in key columns – Dedupe can handle them with has_missing=True."""
    bad = {col: int(df[col].isna().sum()) for col in columns}
    bad = {k: v for k, v in bad.items() if v}
    if bad:
        print(f"[{split_name}] info: null counts in key columns: {bad} (handled via has_missing)")


def run_checks(train_path: Path, val_path: Path, test_path: Path) -> None:
    print("Loading splits...")
    train_df = pd.read_csv(train_path, low_memory=False)
    val_df = pd.read_csv(val_path, low_memory=False)
    test_df = pd.read_csv(test_path, low_memory=False)

    for name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        _require_columns(df, REQUIRED_BASE_COLS, name)
        _check_non_empty(df, name)
        _check_duplicates(df, "source_id_craig", name)
        _check_duplicates(df, "source_id_us", name)
        _check_nulls(df, ["brand_craig", "brand", "model_craig", "model"], name)

    _check_overlap(train_df, val_df, test_df)

    print("✓ All pre-flight checks passed.")
    print("Splits -> train: %d | validation: %d | test: %d" % (len(train_df), len(val_df), len(test_df)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight data checks")
    parser.add_argument("--train", default="../dataset/splits/train.csv")
    parser.add_argument("--val", default="../dataset/splits/validation.csv")
    parser.add_argument("--test", default="../dataset/splits/test.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_checks(Path(args.train), Path(args.val), Path(args.test))

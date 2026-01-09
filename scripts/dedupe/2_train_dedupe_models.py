"""Train and evaluate multiple Dedupe pipelines on the vehicle record linkage task.

This script leaves the existing data and models untouched and writes new outputs
under ``output/dedupe_results/experiments``.

Features
- Runs lightweight pre-flight checks on the splits to guard against schema drift
- Trains multiple pipelines (different field configs) with supervised labels
  derived from the provided positive pairs plus synthetic negative pairs
- Logs timing for each training step and for inference
- Reports precision / recall / F1 on the held-out test split

Usage
------
python scripts/train_dedupe_models.py \
    --train dataset/splits/train.csv \
    --val dataset/splits/validation.csv \
    --test dataset/splits/test.csv \
    --out-dir output/dedupe_results/experiments

Requirements
- python -m pip install dedupe[performance]
- Data files produced by the previous pipeline (see README / docs)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

try:
    import dedupe
except ImportError as exc: 
    raise SystemExit(
        "Missing dependency: dedupe. Install it with `pip install dedupe[performance]`"
    ) from exc

# ----------------------------
# Configuration
# ----------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

PIPELINES = [
    {
        "name": "P1_textual_core",
        "fields": [
            dedupe.variables.String("brand", has_missing=True),
            dedupe.variables.String("model", has_missing=True),
            dedupe.variables.String("body_type", has_missing=True),
            dedupe.variables.Text("description", has_missing=True),
            dedupe.variables.Price("price", has_missing=True),
            dedupe.variables.Price("mileage", has_missing=True),
        ],
    },
    {
        "name": "P2_plus_location",
        "fields": [
            dedupe.variables.String("brand", has_missing=True),
            dedupe.variables.String("model", has_missing=True),
            dedupe.variables.String("body_type", has_missing=True),
            dedupe.variables.String("transmission", has_missing=True),
            dedupe.variables.String("fuel_type", has_missing=True),
            dedupe.variables.String("drive", has_missing=True),
            dedupe.variables.String("city_region", has_missing=True),
            dedupe.variables.String("state", has_missing=True),
            dedupe.variables.Text("description", has_missing=True),
            dedupe.variables.Price("price", has_missing=True),
            dedupe.variables.Price("mileage", has_missing=True),
            dedupe.variables.Price("year", has_missing=True),
        ],
    },
    {
        "name": "P3_minimal_fast",
        "fields": [
            dedupe.variables.String("brand", has_missing=True),
            dedupe.variables.String("model", has_missing=True),
            dedupe.variables.Price("year", has_missing=True),
        ],
    },
]

CRAIG_MAP = {
    "id": "source_id_craig",
    "brand": "brand_craig",
    "model": "model_craig",
    "year": "year_craig",
    "price": "price_craig",
    "mileage": "mileage_craig",
    "color": "color_craig",
    "description": "description_craig",
    "body_type": "body_type_craig",
    "transmission": "transmission_craig",
    "fuel_type": "fuel_type_craig",
    "drive": "drive_craig",
    "city_region": "city_region_craig",
    "state": "state_craig",
}

US_MAP = {
    "id": "source_id_us",
    "brand": "brand",
    "model": "model",
    "year": "year",
    "price": "price",
    "mileage": "mileage",
    "color": "color",
    "description": "description",
    "body_type": "body_type",
    "transmission": "transmission",
    "fuel_type": "fuel_type",
    "drive": "drive",
    "city_region": "city_region",
    "state": "state",
}

# ----------------------------
# Helpers
# ----------------------------

def _timer() -> Tuple[callable, callable]:
    start = time.perf_counter()

    def lap() -> float:
        nonlocal start
        now = time.perf_counter()
        delta = now - start
        start = now
        return delta

    def total() -> float:
        return time.perf_counter() - start

    return lap, total


NUMERIC_FIELDS = {"price", "mileage", "year"}


def _to_records_unique(df: pd.DataFrame, mapping: Dict[str, str], prefix: str) -> Dict[str, dict]:
    """Build record dict with unique keys (row index based) to avoid collisions."""
    records: Dict[str, dict] = {}
    for idx, row in enumerate(df.itertuples(index=False)):
        record_id = f"{prefix}_{idx}"
        rec = {}
        for key, col in mapping.items():
            if key == "id":
                continue
            val = getattr(row, col)
            if pd.isna(val):
                rec[key] = None
            elif key in NUMERIC_FIELDS:
                try:
                    rec[key] = float(val)
                except (ValueError, TypeError):
                    rec[key] = None
            else:
                rec[key] = str(val) if val else ""
        records[record_id] = rec
    return records


def _build_training_pairs_from_records(
    craig_records: Dict[str, dict],
    us_records: Dict[str, dict],
    negative_multiplier: float = 1.0,
) -> dict:
    """All pairs are matches (since each row in ground truth is a positive pair).
    Generate synthetic negatives by pairing non-corresponding indices.
    Returns pairs with keys (record IDs) instead of full record dicts."""
    craig_ids = list(craig_records.keys())
    us_ids = list(us_records.keys())
    
    # All index-aligned pairs are matches - store as (craig_key, us_key)
    match_keys = list(zip(craig_ids, us_ids))

    rng = random.Random(RANDOM_SEED)
    target_negatives = max(1, int(len(match_keys) * negative_multiplier))

    distinct_keys: List[Tuple[str, str]] = []
    attempts = 0
    max_attempts = target_negatives * 10
    while len(distinct_keys) < target_negatives and attempts < max_attempts:
        attempts += 1
        c_idx = rng.randrange(len(craig_ids))
        u_idx = rng.randrange(len(us_ids))
        if c_idx == u_idx:  # same row would be a match
            continue
        distinct_keys.append((craig_ids[c_idx], us_ids[u_idx]))

    return {"match": match_keys, "distinct": distinct_keys}


def _make_ground_truth(df: pd.DataFrame) -> set:
    return {
        (str(row[CRAIG_MAP["id"]]), str(row[US_MAP["id"]]))
        for _, row in df[[CRAIG_MAP["id"], US_MAP["id"]]].iterrows()
    }


def _ensure_columns(df: pd.DataFrame, required: Sequence[str], split_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {split_name}: {missing}")


def _evaluate_predictions(pred_pairs: set, truth_pairs: set) -> dict:
    tp = len(pred_pairs & truth_pairs)
    fp = len(pred_pairs - truth_pairs)
    fn = len(truth_pairs - pred_pairs)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


# ----------------------------
# Core pipeline
# ----------------------------

def run_pipeline(
    name: str,
    fields: list,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: Path,
) -> dict:
    print(f"\n=== Pipeline {name} ===")

    lap, total = _timer()

    # Build per-split dictionaries with unique keys (row-index based)
    craig_train = _to_records_unique(train_df, CRAIG_MAP, "craig_train")
    us_train = _to_records_unique(train_df, US_MAP, "us_train")
    craig_val = _to_records_unique(val_df, CRAIG_MAP, "craig_val")
    us_val = _to_records_unique(val_df, US_MAP, "us_val")
    craig_test = _to_records_unique(test_df, CRAIG_MAP, "craig_test")
    us_test = _to_records_unique(test_df, US_MAP, "us_test")

    pair_keys = _build_training_pairs_from_records(craig_train, us_train, negative_multiplier=1.0)

    print(f"Records => train: {len(craig_train)} x {len(us_train)}, val: {len(craig_val)} x {len(us_val)}, test: {len(craig_test)} x {len(us_test)}")
    print(f"Training pairs => match: {len(pair_keys['match'])}, distinct: {len(pair_keys['distinct'])}")

    # Build the actual training pairs (with full record dicts) for the JSON file
    training_pairs = {
        "match": [(craig_train[c], us_train[u]) for c, u in pair_keys["match"]],
        "distinct": [(craig_train[c], us_train[u]) for c, u in pair_keys["distinct"]],
    }

    # Write training pairs to temp JSON file (Dedupe's expected format)
    training_file = out_dir / f"{name}_training.json"
    with open(training_file, "w") as tf:
        json.dump(training_pairs, tf)

    linker = dedupe.RecordLink(fields, num_cores=4)

    # prepare_training with training_file (must pass open file handle)
    lap()
    with open(training_file, "r") as tf:
        linker.prepare_training(craig_train, us_train, training_file=tf, sample_size=15000)
    prep_time = lap()
    print(f"prepare_training: {prep_time:.2f}s")

    # train
    lap()
    linker.train()
    train_time = lap()
    print(f"train: {train_time:.2f}s")

    # Note: threshold() method was removed in dedupe 3.0
    # Use a fixed threshold (0.5 = default) or score-based filtering
    threshold = 0.5
    threshold_time = 0.0
    print(f"threshold (fixed): {threshold:.4f}")

    # inference on test set
    lap()
    linked_records = linker.join(craig_test, us_test, threshold)
    pred_pairs = set()
    for pair, _score in linked_records:
        if not pair or len(pair) != 2:
            continue
        pred_pairs.add((str(pair[0]), str(pair[1])))
    infer_time = lap()
    print(f"inference (test join): {infer_time:.2f}s | predicted pairs: {len(pred_pairs)}")

    # Ground truth for test: index-aligned pairs
    gt_test = {(f"craig_test_{i}", f"us_test_{i}") for i in range(len(test_df))}
    
    metrics = _evaluate_predictions(pred_pairs, gt_test)
    print(
        f"metrics -> precision: {metrics['precision']:.3f}, "
        f"recall: {metrics['recall']:.3f}, f1: {metrics['f1']:.3f}, "
        f"tp: {metrics['tp']}, fp: {metrics['fp']}, fn: {metrics['fn']}"
    )

    settings_path = out_dir / f"{name}_settings.json"
    with settings_path.open("wb") as sf:
        linker.write_settings(sf)

    summary = {
        "name": name,
        "threshold": threshold,
        "timings": {
            "prepare_training_sec": round(prep_time, 3),
            "train_sec": round(train_time, 3),
            "threshold_sec": round(threshold_time, 3),
            "inference_sec": round(infer_time, 3),
        },
        "metrics": metrics,
        "settings_file": str(settings_path.relative_to(out_dir.parent)),
    }

    result_path = out_dir / f"{name}_results.json"
    result_path.write_text(json.dumps(summary, indent=2))
    return summary


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate Dedupe pipelines")
    parser.add_argument("--train", default="dataset/splits/train.csv", help="Path to train split")
    parser.add_argument("--val", default="dataset/splits/validation.csv", help="Path to validation split")
    parser.add_argument("--test", default="dataset/splits/test.csv", help="Path to test split")
    parser.add_argument(
        "--out-dir",
        default="output/dedupe_results/experiments",
        help="Directory to store settings and metrics",
    )
    parser.add_argument("--negative-multiplier", type=float, default=1.0, help="#distinct pairs ~= multiplier * #match pairs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train, low_memory=False)
    val_df = pd.read_csv(args.val, low_memory=False)
    test_df = pd.read_csv(args.test, low_memory=False)

    required_cols = set(CRAIG_MAP.values()) | set(US_MAP.values())
    _ensure_columns(train_df, required_cols, "train")
    _ensure_columns(val_df, required_cols, "validation")
    _ensure_columns(test_df, required_cols, "test")

    all_summaries = []
    for pipeline in PIPELINES:
        summary = run_pipeline(
            pipeline["name"],
            pipeline["fields"],
            train_df,
            val_df,
            test_df,
            out_dir,
        )
        all_summaries.append(summary)

    aggregate = {"runs": all_summaries}
    aggregate_path = out_dir / "summary_all.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2))
    print(f"\nSaved aggregate metrics to {aggregate_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - safety net for CLI usage
        print(f"Fatal error: {exc}", file=sys.stderr)
        sys.exit(1)

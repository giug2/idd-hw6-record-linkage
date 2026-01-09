"""Train and evaluate Dedupe pipelines with MANUAL blocking strategies.

This script trains the same three pipelines (P1, P2, P3) but uses manually defined
blocking rules (B1, B2) instead of letting Dedupe learn its own blocking predicates.

The key difference from 2_train_dedupe_models.py:
- Training pairs are generated from candidates produced by manual blocking (B1/B2)
- This exposes the model to "hard negatives" (same brand/year but different model)
- The model learns to distinguish these difficult cases better

Outputs are saved under ``output/dedupe_results/manual_blocking_experiments``.

Usage
------
python scripts/dedupe/2b_train_with_manual_blocking.py \
    --train dataset/splits/train.csv \
    --test dataset/splits/test.csv \
    --blocking B1  # or B2 or Union
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import importlib.util

import pandas as pd

try:
    import dedupe
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: dedupe. Install with `pip install dedupe[performance]`"
    ) from exc

# ----------------------------
# Configuration
# ----------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Import blocking modules dynamically
def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Get base path for blocking modules
BASE_PATH = Path(__file__).parent.parent
b1_module = import_module_from_path("blocking_B1", str(BASE_PATH / "blocking" / "blocking_B1.py"))
b2_module = import_module_from_path("blocking_B2", str(BASE_PATH / "blocking" / "blocking_B2.py"))

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

NUMERIC_FIELDS = {"price", "mileage", "year"}

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


def _to_records_unique(df: pd.DataFrame, mapping: Dict[str, str], prefix: str) -> Dict[str, dict]:
    """Build record dict with unique keys (row index based)."""
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


def generate_blocking_candidates(df: pd.DataFrame, blocking_strategy: str) -> set:
    """
    Generate candidate pairs using manual blocking strategy.
    
    Args:
        df: DataFrame with aligned records (each row is a true match)
        blocking_strategy: 'B1', 'B2', or 'Union'
    
    Returns:
        Set of candidate pairs as (idx_craig, idx_us) tuples
    """
    candidates = set()
    
    if blocking_strategy in ['B1', 'Union']:
        # B1: Brand + Year
        blocks_craig = b1_module.blocking_B1(df, brand_col='brand_craig', year_col='year_craig')
        blocks_us = b1_module.blocking_B1(df, brand_col='brand', year_col='year')
        cand_b1 = b1_module.generate_candidate_pairs(blocks_craig, blocks_us)
        candidates.update(cand_b1)
        
    if blocking_strategy in ['B2', 'Union']:
        # B2: Brand + Model Prefix
        blocks_craig = b2_module.blocking_B2(df, brand_col='brand_craig', model_col='model_craig')
        blocks_us = b2_module.blocking_B2(df, brand_col='brand', model_col='model')
        cand_b2 = b1_module.generate_candidate_pairs(blocks_craig, blocks_us)
        candidates.update(cand_b2)
    
    return candidates


def _build_training_pairs_from_blocking(
    df: pd.DataFrame,
    craig_records: Dict[str, dict],
    us_records: Dict[str, dict],
    blocking_strategy: str,
    negative_multiplier: float = 2.0,
) -> dict:
    """
    Build training pairs using manual blocking to find hard negatives.
    
    Key insight: All candidate pairs that are NOT true matches (i.e., idx1 != idx2)
    are hard negatives - they passed the blocking filter but are different cars.
    """
    # Get all candidates from blocking
    candidates = generate_blocking_candidates(df, blocking_strategy)
    
    # Ground truth: row i in craig matches row i in us
    true_matches = set((i, i) for i in range(len(df)))
    
    # Separate matches from non-matches among candidates
    match_pairs = []
    distinct_pairs = []
    
    for idx1, idx2 in candidates:
        craig_key = f"craig_train_{idx1}"
        us_key = f"us_train_{idx2}"
        
        if idx1 == idx2:  # True match
            match_pairs.append((craig_key, us_key))
        else:  # Hard negative (blocking passed but not a match)
            distinct_pairs.append((craig_key, us_key))
    
    # Balance the dataset: limit distinct pairs
    rng = random.Random(RANDOM_SEED)
    target_distinct = int(len(match_pairs) * negative_multiplier)
    if len(distinct_pairs) > target_distinct:
        distinct_pairs = rng.sample(distinct_pairs, target_distinct)
    
    print(f"Blocking strategy: {blocking_strategy}")
    print(f"Total candidates from blocking: {len(candidates)}")
    print(f"Match pairs (true positives in blocking): {len(match_pairs)}")
    print(f"Distinct pairs (hard negatives): {len(distinct_pairs)}")
    
    return {"match": match_pairs, "distinct": distinct_pairs}


def _evaluate_predictions(pred_pairs: set, truth_pairs: set) -> dict:
    tp = len(pred_pairs & truth_pairs)
    fp = len(pred_pairs - truth_pairs)
    fn = len(truth_pairs - pred_pairs)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _ensure_columns(df: pd.DataFrame, required, split_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {split_name}: {missing}")


# ----------------------------
# Core pipeline
# ----------------------------

def run_pipeline_with_manual_blocking(
    name: str,
    fields: list,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    blocking_strategy: str,
    out_dir: Path,
) -> dict:
    """Train a pipeline using manual blocking for candidate generation."""
    
    full_name = f"{name}_manual_{blocking_strategy}"
    print(f"\n=== Pipeline {full_name} ===")

    lap, _ = _timer()

    # Build record dictionaries
    craig_train = _to_records_unique(train_df, CRAIG_MAP, "craig_train")
    us_train = _to_records_unique(train_df, US_MAP, "us_train")
    craig_test = _to_records_unique(test_df, CRAIG_MAP, "craig_test")
    us_test = _to_records_unique(test_df, US_MAP, "us_test")

    print(f"Records => train: {len(craig_train)}, test: {len(craig_test)}")

    # Build training pairs using manual blocking (key difference!)
    pair_keys = _build_training_pairs_from_blocking(
        train_df, craig_train, us_train, blocking_strategy, negative_multiplier=2.0
    )

    # Convert to full record pairs for training file
    training_pairs = {
        "match": [(craig_train[c], us_train[u]) for c, u in pair_keys["match"]],
        "distinct": [(craig_train[c], us_train[u]) for c, u in pair_keys["distinct"]],
    }

    # Save training pairs
    training_file = out_dir / f"{full_name}_training.json"
    with open(training_file, "w") as tf:
        json.dump(training_pairs, tf)

    # Initialize and train
    linker = dedupe.RecordLink(fields, num_cores=4)

    lap()
    with open(training_file, "r") as tf:
        linker.prepare_training(craig_train, us_train, training_file=tf, sample_size=15000)
    prep_time = lap()
    print(f"prepare_training: {prep_time:.2f}s")

    lap()
    linker.train()
    train_time = lap()
    print(f"train: {train_time:.2f}s")

    threshold = 0.5

    # Inference: Use the SAME manual blocking on test set
    lap()
    
    # Generate test candidates using manual blocking
    test_candidates = generate_blocking_candidates(test_df, blocking_strategy)
    print(f"Test candidates from {blocking_strategy}: {len(test_candidates)}")
    
    # Score only the blocked candidates
    dedupe_pairs = []
    pair_lookup = []
    for idx1, idx2 in test_candidates:
        craig_key = f"craig_test_{idx1}"
        us_key = f"us_test_{idx2}"
        r1 = craig_test[craig_key]
        r2 = us_test[us_key]
        dedupe_pairs.append(((craig_key, r1), (us_key, r2)))
        pair_lookup.append((craig_key, us_key))
    
    # Score pairs
    if dedupe_pairs:
        scores = linker.score(dedupe_pairs)
        
        # Handle structured array
        if scores.dtype.names:
            score_values = scores['score']
        else:
            score_values = scores
        
        pred_pairs = set()
        for i, score in enumerate(score_values):
            if score > threshold:
                pred_pairs.add(pair_lookup[i])
    else:
        pred_pairs = set()
    
    infer_time = lap()
    print(f"inference: {infer_time:.2f}s | predicted pairs: {len(pred_pairs)}")

    # Ground truth for test
    gt_test = {(f"craig_test_{i}", f"us_test_{i}") for i in range(len(test_df))}

    metrics = _evaluate_predictions(pred_pairs, gt_test)
    print(
        f"metrics -> precision: {metrics['precision']:.3f}, "
        f"recall: {metrics['recall']:.3f}, f1: {metrics['f1']:.3f}, "
        f"tp: {metrics['tp']}, fp: {metrics['fp']}, fn: {metrics['fn']}"
    )

    # Save model
    settings_path = out_dir / f"{full_name}_settings.json"
    with settings_path.open("wb") as sf:
        linker.write_settings(sf)

    summary = {
        "name": full_name,
        "base_pipeline": name,
        "blocking_strategy": blocking_strategy,
        "threshold": threshold,
        "timings": {
            "prepare_training_sec": round(prep_time, 3),
            "train_sec": round(train_time, 3),
            "inference_sec": round(infer_time, 3),
        },
        "metrics": metrics,
        "settings_file": str(settings_path.name),
        "training_stats": {
            "match_pairs": len(pair_keys["match"]),
            "distinct_pairs": len(pair_keys["distinct"]),
            "test_candidates": len(test_candidates),
        }
    }

    result_path = out_dir / f"{full_name}_results.json"
    result_path.write_text(json.dumps(summary, indent=2))
    return summary


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Dedupe pipelines with manual blocking strategies"
    )
    parser.add_argument("--train", default="dataset/splits/train.csv")
    parser.add_argument("--test", default="dataset/splits/test.csv")
    parser.add_argument(
        "--blocking", 
        choices=["B1", "B2", "Union", "all"],
        default="all",
        help="Which blocking strategy to use (default: all)"
    )
    parser.add_argument(
        "--out-dir",
        default="output/dedupe_results/manual_blocking_experiments",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train_df = pd.read_csv(args.train, low_memory=False)
    test_df = pd.read_csv(args.test, low_memory=False)

    required_cols = set(CRAIG_MAP.values()) | set(US_MAP.values())
    _ensure_columns(train_df, required_cols, "train")
    _ensure_columns(test_df, required_cols, "test")

    # Determine blocking strategies to run
    if args.blocking == "all":
        blocking_strategies = ["B1", "B2", "Union"]
    else:
        blocking_strategies = [args.blocking]

    all_summaries = []
    
    for blocking in blocking_strategies:
        print(f"\n{'='*60}")
        print(f"Running experiments with blocking strategy: {blocking}")
        print(f"{'='*60}")
        
        for pipeline in PIPELINES:
            summary = run_pipeline_with_manual_blocking(
                pipeline["name"],
                pipeline["fields"],
                train_df,
                test_df,
                blocking,
                out_dir,
            )
            all_summaries.append(summary)

    # Save aggregate results
    aggregate = {"runs": all_summaries}
    aggregate_path = out_dir / "summary_all.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2))
    
    # Print summary table
    print("\n" + "="*100)
    print(f"{'PIPELINE':<35} | {'PREC':<7} | {'REC':<7} | {'F1':<7} | {'TP':<5} | {'FP':<5} | {'FN':<5}")
    print("-"*100)
    for s in all_summaries:
        m = s['metrics']
        print(f"{s['name']:<35} | {m['precision']:.3f}   | {m['recall']:.3f}   | {m['f1']:.3f}   | {m['tp']:<5} | {m['fp']:<5} | {m['fn']:<5}")
    print("="*100)
    
    print(f"\nSaved results to {aggregate_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

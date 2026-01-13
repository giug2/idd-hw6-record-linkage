"""
Experiment: P3 Extended (Brand, Model, Year + Price, Mileage).

This script runs P3 Extended with:
1. Auto-Blocking (Dedupe's active learning)
2. Manual Blocking (B1, B2, Union)

It saves results in the standard output directories to be fully compatible with visualization scripts.
"""
import sys
import os
import json
import random
import time
import pandas as pd
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Set


try:
    import dedupe
except ImportError:
    print("Dedupe not installed.")
    sys.exit(1)


# Configuration
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR_AUTO = BASE_DIR / "output/dedupe_results/experiments"
OUTPUT_DIR_MANUAL = BASE_DIR / "output/dedupe_results/manual_blocking_experiments"

os.makedirs(OUTPUT_DIR_AUTO, exist_ok=True)
os.makedirs(OUTPUT_DIR_MANUAL, exist_ok=True)

# P3 Extended Pipeline Definition
P3_EXTENDED_PREFIX = "P3_extended"
P3_EXTENDED_FIELDS = [
    dedupe.variables.String("brand", has_missing=True),
    dedupe.variables.String("model", has_missing=True),
    dedupe.variables.Price("year", has_missing=True),
    dedupe.variables.Price("price", has_missing=True),
    dedupe.variables.Price("mileage", has_missing=True),
]

CRAIG_MAP = {
    "brand": "brand_craig",
    "model": "model_craig",
    "year": "year_craig",
    "price": "price_craig",
    "mileage": "mileage_craig",
}
US_MAP = {
    "brand": "brand",
    "model": "model",
    "year": "year",
    "price": "price",
    "mileage": "mileage",
}

NUMERIC_FIELDS = {"year", "price", "mileage"}

# Import blocking modules dynamically (copied from 2b)
def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

BLOCKING_DIR = BASE_DIR / "scripts/blocking"
b1_module = import_module_from_path("blocking_B1", str(BLOCKING_DIR / "blocking_B1.py"))
b2_module = import_module_from_path("blocking_B2", str(BLOCKING_DIR / "blocking_B2.py"))


class Timer:
    def __init__(self):
        self.start = time.time()
        self.last = self.start
    
    def lap(self):
        now = time.time()
        diff = now - self.last
        self.last = now
        return diff
    
    def total(self):
        return time.time() - self.start

def _to_records_unique(df: pd.DataFrame, mapping: dict, prefix: str) -> Dict[str, dict]:
    records = {}
    for idx, row in enumerate(df.itertuples(index=False)):
        record_id = f"{prefix}_{idx}"
        rec = {}
        for key, col in mapping.items():
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

def _evaluate_predictions(pred_pairs: set, truth_pairs: set) -> dict:
    tp = len(pred_pairs & truth_pairs)
    fp = len(pred_pairs - truth_pairs)
    fn = len(truth_pairs - pred_pairs)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

# ---------------------------------------------------------
# Part 1: Auto-Blocking Logic (from 2_train_dedupe_models.py)
# ---------------------------------------------------------
def _build_auto_training_pairs(craig_records, us_records):
    """Synthetic negatives + aligned positives (easy negatives)"""
    craig_ids = list(craig_records.keys())
    us_ids = list(us_records.keys())
    
    # 1. Matches (Aligned)
    match_pairs = []
    # Assuming index alignment: craig_train_i matches us_train_i
    # Since keys are ordered by enumerate in _to_records_unique, we can zip keys
    # But safer to verify index suffix
    # Actually, simplistic assumption: keys generated in same order match
    
    match_keys = list(zip(craig_ids, us_ids)) # (craig_train_0, us_train_0) etc.
    
    # 2. Negatives (Random)
    rng = random.Random(RANDOM_SEED)
    distinct_keys = []
    target_negatives = len(match_keys) * 1 # 1:1 ratio
    
    while len(distinct_keys) < target_negatives:
        c_idx = rng.randrange(len(craig_ids))
        u_idx = rng.randrange(len(us_ids))
        if c_idx != u_idx:
            distinct_keys.append((craig_ids[c_idx], us_ids[u_idx]))
            
    return {"match": match_keys, "distinct": distinct_keys}

def run_auto_blocking(train_df, test_df):
    print("\n" + "="*50)
    print(f"Running P3 Extended with AUTO-BLOCKING")
    print("="*50)
    
    timer = Timer()
    
    # Prepare Data
    craig_train = _to_records_unique(train_df, CRAIG_MAP, "craig_train")
    us_train = _to_records_unique(train_df, US_MAP, "us_train")
    craig_test = _to_records_unique(test_df, CRAIG_MAP, "craig_test")
    us_test = _to_records_unique(test_df, US_MAP, "us_test")
    
    # Build Training Pairs
    pair_keys = _build_auto_training_pairs(craig_train, us_train)
    training_data = {
        "match": [(craig_train[c], us_train[u]) for c, u in pair_keys["match"]],
        "distinct": [(craig_train[c], us_train[u]) for c, u in pair_keys["distinct"]]
    }
    
    # Save training file
    training_file = OUTPUT_DIR_AUTO / f"{P3_EXTENDED_PREFIX}_training.json"
    with open(training_file, "w") as f:
        json.dump(training_data, f)
        
    linker = dedupe.RecordLink(P3_EXTENDED_FIELDS, num_cores=4)
    
    # Prepare Training
    timer.lap()
    with open(training_file, "r") as f:
        linker.prepare_training(craig_train, us_train, training_file=f, sample_size=10000)
    prep_time = timer.lap()
    
    # Train
    linker.train()
    train_time = timer.lap()
    
    # Save Settings
    settings_file = OUTPUT_DIR_AUTO / f"{P3_EXTENDED_PREFIX}_settings.json"
    with open(settings_file, "wb") as f:
        linker.write_settings(f)
        
    # Inference
    print("Running inference on test set...")
    threshold = 0.5
    linked_records = linker.join(craig_test, us_test, threshold)
    pred_pairs = set()
    for pair, _ in linked_records:
        if len(pair) == 2:
            pred_pairs.add((str(pair[0]), str(pair[1])))
    infer_time = timer.lap()
    
    # Evaluate
    gt_test = {(f"craig_test_{i}", f"us_test_{i}") for i in range(len(test_df))}
    metrics = _evaluate_predictions(pred_pairs, gt_test)
    
    # Result object
    result = {
        "name": P3_EXTENDED_PREFIX,
        "metrics": metrics,
        "timings": {
            "prepare_training_sec": prep_time,
            "train_sec": train_time,
            "inference_sec": infer_time
        }
    }
    
    # Save Result
    with open(OUTPUT_DIR_AUTO / f"{P3_EXTENDED_PREFIX}_results.json", "w") as f:
        json.dump(result, f, indent=2)
        
    print(f"Auto-Blocking Results: F1={metrics['f1']:.4f}")
    return result


# ---------------------------------------------------------
# Part 2: Manual-Blocking Logic (from 2b_train_with_manual_blocking.py)
# ---------------------------------------------------------
def _generate_manual_candidates(df, blocking_strategy):
    """Generate candidates (craig_idx, us_idx) using blocking modules"""
    candidates = set()
    
    # Prepare DF for blocking modules (they expect specific columns)
    # blocking_B1 expects 'brand_craig', 'year_craig' etc. which we have in aligned DF
    
    if blocking_strategy in ['B1', 'Union']:
        blocks_craig = b1_module.blocking_B1(df, brand_col='brand_craig', year_col='year_craig')
        blocks_us = b1_module.blocking_B1(df, brand_col='brand', year_col='year')
        cand_b1 = b1_module.generate_candidate_pairs(blocks_craig, blocks_us)
        candidates.update(cand_b1)
        
    if blocking_strategy in ['B2', 'Union']:
        blocks_craig = b2_module.blocking_B2(df, brand_col='brand_craig', model_col='model_craig')
        blocks_us = b2_module.blocking_B2(df, brand_col='brand', model_col='model')
        cand_b2 = b1_module.generate_candidate_pairs(blocks_craig, blocks_us)
        candidates.update(cand_b2)
        
    return candidates


def _build_manual_training_pairs(df, craig_records, us_records, blocking_strategy):
    """Hard negatives from manual blocking"""
    candidates = _generate_manual_candidates(df, blocking_strategy)
    
    match_pairs = []
    distinct_pairs = []
    
    for idx1, idx2 in candidates:
        craig_key = f"craig_train_{idx1}"
        us_key = f"us_train_{idx2}"
        
        if idx1 == idx2: # Match
            match_pairs.append((craig_key, us_key))
        else: # Hard Negative
            distinct_pairs.append((craig_key, us_key))
            
    # Subsample negatives (2:1 ratio max)
    rng = random.Random(RANDOM_SEED)
    target_distinct = len(match_pairs) * 2
    if len(distinct_pairs) > target_distinct:
        distinct_pairs = rng.sample(distinct_pairs, target_distinct)
        
    return {"match": match_pairs, "distinct": distinct_pairs}


def run_manual_blocking_exp(train_df, test_df, blocking_strategy):
    full_name = f"{P3_EXTENDED_PREFIX}_manual_{blocking_strategy}"
    print("\n" + "="*50)
    print(f"Running P3 Extended with MANUAL BLOCKING: {blocking_strategy}")
    print("="*50)
    
    timer = Timer()
    
    craig_train = _to_records_unique(train_df, CRAIG_MAP, "craig_train")
    us_train = _to_records_unique(train_df, US_MAP, "us_train")
    craig_test = _to_records_unique(test_df, CRAIG_MAP, "craig_test")
    us_test = _to_records_unique(test_df, US_MAP, "us_test")
    
    # 1. Build Training Data from Hard Negatives
    pair_keys = _build_manual_training_pairs(train_df, craig_train, us_train, blocking_strategy)
    
    if not pair_keys["match"]:
        print("Warning: No matches found in blocking!")
        return None

    training_data = {
        "match": [(craig_train[c], us_train[u]) for c, u in pair_keys["match"]],
        "distinct": [(craig_train[c], us_train[u]) for c, u in pair_keys["distinct"]]
    }
    
    training_file = OUTPUT_DIR_MANUAL / f"{full_name}_training.json"
    with open(training_file, "w") as f:
        json.dump(training_data, f)
        
    linker = dedupe.RecordLink(P3_EXTENDED_FIELDS, num_cores=4)
    
    # 2. Train
    timer.lap()
    with open(training_file, "r") as f:
        linker.prepare_training(craig_train, us_train, training_file=f, sample_size=10000)
    prep_time = timer.lap()
    
    linker.train()
    train_time = timer.lap()
    
    # Save Settings
    with open(OUTPUT_DIR_MANUAL / f"{full_name}_settings.json", "wb") as f:
        linker.write_settings(f)
        
    # 3. Inference (using MANUAL Blocking to generate candidates)
    # We must replicate the manual blocking on test set and then classify those pairs
    test_candidates = _generate_manual_candidates(test_df, blocking_strategy)
    print(f"Test candidates from {blocking_strategy}: {len(test_candidates)}")
    
    # Using linker.score to score specific pairs
    dedupe_pairs = []
    pair_lookup = []
    
    for idx1, idx2 in test_candidates:
        c_key = f"craig_test_{idx1}"
        u_key = f"us_test_{idx2}"
        if c_key in craig_test and u_key in us_test:
            dedupe_pairs.append(
                ((c_key, craig_test[c_key]), (u_key, us_test[u_key]))
            )
            pair_lookup.append((c_key, u_key))

    threshold = 0.5
    pred_pairs = set()
    
    if dedupe_pairs:
        try:
            scores = linker.score(dedupe_pairs)
            # Handle structured array if necessary (Dedupe 2.0+ usually returns structured array)
            # but sometimes just floats depending on version
            
            # scores is usually a structured array/list with 'score' field?
            # Dedupe docs says: returns a numpy array of fields 'pairs' and 'score'
            # But wait, linker.score takes list of pairs and returns just scores or structured?
            # 2b check: if scores.dtype.names: score_values = scores['score']
            
            # Let's verify what `scores` is. It is likely a numpy array.
            # If it has dtype names (like 'score'), extract.
            score_values = scores
            if hasattr(scores, 'dtype') and scores.dtype.names and 'score' in scores.dtype.names:
                score_values = scores['score']
            
            for i, score in enumerate(score_values):
                if score > threshold:
                    pred_pairs.add(pair_lookup[i])
                    
        except Exception as e:
            print(f"Classification error: {e}")
            import traceback
            traceback.print_exc()
            
    infer_time = timer.lap()
    
    # Evaluate
    gt_test = {(f"craig_test_{i}", f"us_test_{i}") for i in range(len(test_df))}
    metrics = _evaluate_predictions(pred_pairs, gt_test)
    
    result = {
        "name": full_name,
        "metrics": metrics,
        "timings": {
            "prepare_training_sec": prep_time,
            "train_sec": train_time,
            "inference_sec": infer_time
        }
    }
    
    # Save Result
    with open(OUTPUT_DIR_MANUAL / f"{full_name}_results.json", "w") as f:
        json.dump(result, f, indent=2)
        
    print(f"{blocking_strategy} Results: F1={metrics['f1']:.4f}")
    return result


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    print("Loading data...")
    train_df = pd.read_csv(BASE_DIR / "dataset/splits/train.csv")
    test_df = pd.read_csv(BASE_DIR / "dataset/splits/test.csv")
    
    # 1. Auto Blocking
    run_auto_blocking(train_df, test_df)
    
    # 2. Manual Blocking
    strategies = ["B1", "B2", "Union"]
    for strat in strategies:
        run_manual_blocking_exp(train_df, test_df, strat)
        
    # Combine results into summary for file writer
    print("\nDone.")

if __name__ == "__main__":
    main()

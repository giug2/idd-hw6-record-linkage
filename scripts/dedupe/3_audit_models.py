import pandas as pd
import numpy as np
import dedupe
import os
import json
from typing import Set, Tuple, Dict, List
from sklearn.utils import resample

# Configuration
DATA_DIR = 'dataset/splits'
OUTPUT_DIR = 'output/dedupe_results/experiments'
MODEL_NAME = 'P3_minimal_fast'  # Auditing the best model
SETTINGS_FILE = os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_settings.json')

# Column Mappings (from train_dedupe_models.py)
CRAIG_MAP = {
    "id": "source_id_craig",
    "brand": "brand_craig",
    "model": "model_craig",
    "year": "year_craig",
    "price": "price_craig",
}
US_MAP = {
    "id": "source_id_us",
    "brand": "brand",
    "model": "model",
    "year": "year",
    "price": "price",
}

def load_data():
    print("Loading splits...")
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), low_memory=False)
    test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'), low_memory=False)
    return train, test

def get_ids(df: pd.DataFrame, source: str) -> Set[str]:
    """Extract set of IDs for a given source (craig or us)."""
    col = CRAIG_MAP["id"] if source == 'craig' else US_MAP["id"]
    return set(df[col].astype(str).unique())

def check_data_leakage(train: pd.DataFrame, test: pd.DataFrame):
    print("\n=== 1. Data Leakage & Overlap Analysis ===")
    
    # 1. Record Overlap (Strict Leakage)
    # Do the same records appear in both Train and Test?
    for source in ['craig', 'us']:
        train_ids = get_ids(train, source)
        test_ids = get_ids(test, source)
        overlap = train_ids.intersection(test_ids)
        
        print(f"[{source.upper()}] Train IDs: {len(train_ids)}, Test IDs: {len(test_ids)}")
        if overlap:
            print(f"⚠️  WARNING: Found {len(overlap)} overlapping records in {source} between Train and Test!")
            print("    This means the model has seen these specific entities during training.")
        else:
            print(f"✅  Clean: No record overlap for {source}.")

    # 2. Pair Leakage (Ground Truth Leakage)
    # Do the same MATCH PAIRS appear in both?
    train_pairs = set(zip(train[CRAIG_MAP["id"]].astype(str), train[US_MAP["id"]].astype(str)))
    test_pairs = set(zip(test[CRAIG_MAP["id"]].astype(str), test[US_MAP["id"]].astype(str)))
    
    pair_overlap = train_pairs.intersection(test_pairs)
    if pair_overlap:
        print(f"⚠️  CRITICAL: Found {len(pair_overlap)} exact ground truth pairs in both Train and Test!")
    else:
        print(f"✅  Clean: No ground truth pair leakage.")

NUMERIC_FIELDS = {"price", "mileage", "year"}

def _to_records_dict(df: pd.DataFrame, mapping: Dict[str, str], prefix: str) -> Dict[str, dict]:
    """Convert DataFrame to Dedupe record dictionary."""
    records = {}
    for idx, row in enumerate(df.itertuples(index=False)):
        record_id = f"{prefix}_{idx}" # Use row index as key for dedupe
        # Store original IDs to map back later
        orig_craig_id = str(getattr(row, CRAIG_MAP["id"])) if prefix.startswith("craig") else None
        orig_us_id = str(getattr(row, US_MAP["id"])) if prefix.startswith("us") else None
        
        rec = {"_orig_id": orig_craig_id or orig_us_id}
        
        for key, col in mapping.items():
            if key == "id": continue
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

def audit_model_performance(test_df: pd.DataFrame):
    print(f"\n=== 2. Model Robustness Audit ({MODEL_NAME}) ===")
    
    if not os.path.exists(SETTINGS_FILE):
        print(f"Settings file not found: {SETTINGS_FILE}")
        return

    print("Loading pre-trained model settings...")
    with open(SETTINGS_FILE, 'rb') as f:
        linker = dedupe.StaticRecordLink(f)

    # Prepare Test Data
    print("Preparing test records...")
    craig_records = _to_records_dict(test_df, CRAIG_MAP, "craig_test")
    us_records = _to_records_dict(test_df, US_MAP, "us_test")
    
    # Run Inference
    print("Running inference on Test set...")
    threshold = 0.5
    linked_records = linker.join(craig_records, us_records, threshold)
    
    # Extract Predicted Pairs (using original IDs)
    pred_pairs = []
    for (craig_key, us_key), score in linked_records:
        c_id = craig_records[craig_key]["_orig_id"]
        u_id = us_records[us_key]["_orig_id"]
        pred_pairs.append({"craig_id": c_id, "us_id": u_id, "score": score})
    
    pred_df = pd.DataFrame(pred_pairs)
    
    # Ground Truth Pairs
    gt_pairs = set(zip(test_df[CRAIG_MAP["id"]].astype(str), test_df[US_MAP["id"]].astype(str)))
    
    # --- A. Bootstrap Confidence Intervals ---
    print("\n--- A. Statistical Significance (Bootstrap 95% CI) ---")
    n_iterations = 1000
    f1_scores = []
    
    # We bootstrap the *Test Set Rows* (Ground Truth), and check which predictions match them
    # This is a bit tricky. Easier to bootstrap the list of (Prediction vs Truth) outcomes.
    # Let's construct a list of all relevant pairs (Union of Pred and GT) to evaluate.
    # Actually, standard approach: Bootstrap the Test DataFrame, re-calculate TP/FP/FN.
    
    # Pre-calculate TP/FP/FN status for the full set
    # We need to know for every row in test_df if it was found (TP) or missed (FN).
    # And for every prediction, if it is correct (TP) or wrong (FP).
    
    # Let's simplify: Bootstrap the F1 score by resampling the Test Set (Ground Truth units).
    # For each resampled test set, we count:
    # TP: Preds that match a pair in this resampled GT
    # FN: Pairs in this resampled GT that are not in Preds
    # FP: Preds that are NOT in this resampled GT (This is tricky, FP is usually global. 
    # But if we treat the test set as "the world", FPs are preds not in GT).
    
    # Let's stick to a simpler proxy: Bootstrap the list of predictions? No, that ignores Recall.
    # Bootstrap the Test DataFrame (the ground truth instances).
    
    pred_set = set(zip(pred_df['craig_id'], pred_df['us_id'])) if not pred_df.empty else set()
    
    for i in range(n_iterations):
        # Resample the Ground Truth (Test DataFrame)
        sample_test = resample(test_df)
        sample_gt = set(zip(sample_test[CRAIG_MAP["id"]].astype(str), sample_test[US_MAP["id"]].astype(str)))
        
        tp = len(pred_set.intersection(sample_gt))
        fn = len(sample_gt - pred_set)
        # FP: Predictions that are NOT in the sample_gt. 
        # Note: In standard bootstrap, we assume the "population" of negatives is vast. 
        # Here, we just count how many of our fixed predictions are wrong relative to the resampled GT.
        fp = len(pred_set - sample_gt)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
        
    lower = np.percentile(f1_scores, 2.5)
    upper = np.percentile(f1_scores, 97.5)
    mean_f1 = np.mean(f1_scores)
    
    print(f"F1 Score (Mean): {mean_f1:.4f}")
    print(f"95% Confidence Interval: [{lower:.4f}, {upper:.4f}]")
    if (upper - lower) > 0.1:
        print("⚠️  Wide CI indicates high variance/instability in test performance.")
    else:
        print("✅  Narrow CI indicates stable performance.")

    # --- B. Subgroup Analysis (Bias Check) ---
    print("\n--- B. Subgroup Analysis (Bias Check) ---")
    # We will check Recall per subgroup (easier to define than Precision per subgroup for unlinked records)
    
    # Define subgroups based on Year
    test_df['year_group'] = pd.cut(test_df[US_MAP['year']], bins=[0, 2000, 2010, 2015, 2025], labels=['<2000', '2000-2010', '2010-2015', '2015+'])
    
    print(f"{'Subgroup':<15} | {'Size':<5} | {'Recall':<8}")
    print("-" * 35)
    
    for group in test_df['year_group'].unique():
        if pd.isna(group): continue
        sub_df = test_df[test_df['year_group'] == group]
        sub_gt = set(zip(sub_df[CRAIG_MAP["id"]].astype(str), sub_df[US_MAP["id"]].astype(str)))
        
        if not sub_gt: continue
        
        tp_sub = len(pred_set.intersection(sub_gt))
        recall_sub = tp_sub / len(sub_gt)
        print(f"{str(group):<15} | {len(sub_gt):<5} | {recall_sub:.4f}")

    # --- C. Impossible Matches (Sanity Check) ---
    print("\n--- C. Impossible Match Analysis ---")
    # Check predicted pairs for logic violations (e.g. Year diff > 2)
    
    # We need to join back the data to the predictions
    # Create lookups
    craig_lookup = test_df.set_index(CRAIG_MAP["id"])
    us_lookup = test_df.set_index(US_MAP["id"])
    
    # Note: pred_df contains IDs. We need to find the attributes.
    # Since test_df is aligned, we can just look up by ID.
    # But wait, pred_df might contain FPs that are NOT in test_df (if we ran on full dataset).
    # But here we ran on test_df only. So all IDs in pred_df MUST be in test_df (or at least in the records we passed).
    
    # Actually, we passed `craig_test` and `us_test` derived from `test_df`.
    # So all IDs are present.
    
    violations = 0
    total_preds = len(pred_df)
    
    for _, row in pred_df.iterrows():
        c_id = row['craig_id']
        u_id = row['us_id']
        
        # Get years
        # Note: In the test split, rows are aligned. But predictions might link row i to row j.
        # We need to find the row in test_df where id == c_id
        try:
            c_year = craig_lookup.loc[c_id, CRAIG_MAP["year"]]
            u_year = us_lookup.loc[u_id, US_MAP["year"]]
            
            if pd.notna(c_year) and pd.notna(u_year):
                if abs(c_year - u_year) > 1: # Allow 1 year diff
                    violations += 1
        except KeyError:
            pass # Should not happen given setup
            
    print(f"Total Predictions: {total_preds}")
    print(f"Year Mismatch Violations (>1 year diff): {violations} ({violations/total_preds*100:.2f}%)")
    
    if violations / total_preds > 0.05:
        print("⚠️  High rate of year mismatches! Model might be ignoring Year.")
    else:
        print("✅  Low rate of impossible matches.")

if __name__ == "__main__":
    train_df, test_df = load_data()
    check_data_leakage(train_df, test_df)
    audit_model_performance(test_df)

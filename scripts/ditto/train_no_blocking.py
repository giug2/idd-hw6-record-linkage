"""
DITTO Training WITHOUT BLOCKING - Solo Pipeline
================================================
Questo script:
1. Legge i dataset da dataset/splits/ (train.csv, validation.csv, test.csv)
2. GENERA COPPIE NEGATIVE per bilanciare il dataset
3. NON applica strategie di blocking
4. Crea dataset nel formato DITTO per le 3 pipeline
5. Addestra i 3 modelli con PREDIZIONI REALI
6. Salva i modelli in output/ditto/modelli/
7. Salva i risultati in output/ditto/ditto_results_no_blocking.txt
"""

import os
import sys
import time
import random
import re
import warnings
warnings.filterwarnings('ignore')


# Disabilita CUDA per usare DirectML
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# GPU Setup
GPU_DEVICE = None
GPU_AVAILABLE = False

try:
    import torch_directml
    GPU_DEVICE = torch_directml.device()
    GPU_AVAILABLE = True
    print("DirectML GPU Acceleration ENABLED")
except ImportError:
    GPU_DEVICE = torch.device('cpu')
    print("DirectML not available, using CPU")

# Paths
ROOT_DIR = Path(__file__).parent.parent.parent
SPLITS_DIR = ROOT_DIR / "dataset" / "splits"
OUTPUT_DIR = ROOT_DIR / "output" / "ditto"
DITTO_DATASET_DIR = OUTPUT_DIR / "ditto_dataset"
MODELS_DIR = OUTPUT_DIR / "modelli"

# Create directories
DITTO_DATASET_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Pipeline definitions (senza blocking)
PIPELINES = {
    'P1_textual_core': ['brand', 'model', 'body_type', 'price', 'mileage'],
    'P2_plus_location': ['brand', 'model', 'body_type', 'price', 'mileage', 
                         'transmission', 'fuel_type', 'drive', 'city_region', 'state', 'year'],
    'P3_minimal_fast': ['brand', 'model', 'year']
}


# ===========================================================================
# DATASET PREPARATION
# ===========================================================================
def extract_representation(row: pd.Series, source: str, fields: List[str]) -> str:
    """Estrae la rappresentazione testuale di un record."""
    values = []
    for field in fields:
        if source == 'craig':
            col = f"{field}_craig"
        else:
            col = field
        
        if col in row.index:
            val = row[col]
            if pd.isna(val):
                val = ""
            else:
                val = str(val).strip()
            val = val.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
            values.append(val)
    
    return " ".join(filter(None, values))


def generate_negative_pairs(df: pd.DataFrame, num_negatives_per_positive: int = 1) -> pd.DataFrame:
    """
    Genera coppie negative per bilanciare il dataset.
    Per ogni riga positiva, crea coppie negative mischiando i record US.
    """
    n = len(df)
    if n == 0:
        return df
    
    # Crea una copia per i positivi
    positive_df = df.copy()
    positive_df['label'] = 1
    
    # Genera negativi
    negative_rows = []
    
    # Colonne US (non _craig)
    us_cols = [c for c in df.columns if not c.endswith('_craig') and c not in ['label', 'source_id_craig.1']]
    
    indices = list(df.index)
    random.seed(42)
    
    for idx in indices:
        row = df.loc[idx].copy()
        
        # Trova un altro record US diverso (che abbia brand/model diverso per essere un vero negativo)
        for _ in range(num_negatives_per_positive):
            attempts = 0
            while attempts < 10:
                other_idx = random.choice(indices)
                if other_idx != idx:
                    other_row = df.loc[other_idx]
                    # Verifica che sia effettivamente diverso (brand o model diverso)
                    if (row.get('brand_craig', '') != other_row.get('brand', '') or 
                        row.get('model_craig', '') != other_row.get('model', '')):
                        break
                attempts += 1
            
            if other_idx == idx:
                continue
            
            # Crea coppia negativa: craig di idx + US di other_idx
            neg_row = row.copy()
            for col in us_cols:
                if col in other_row.index:
                    neg_row[col] = other_row[col]
            neg_row['label'] = 0
            negative_rows.append(neg_row)
    
    # Combina positivi e negativi
    negative_df = pd.DataFrame(negative_rows)
    combined = pd.concat([positive_df, negative_df], ignore_index=True)
    
    # Shuffle
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    return combined


def create_ditto_file_no_blocking(df: pd.DataFrame, pipeline: str, output_file: str) -> Tuple[int, int]:
    """
    Crea un file DITTO (tsv: text1 \t text2 \t label).
    NESSUN blocking applicato - include tutte le coppie.
    """
    fields = PIPELINES[pipeline]
    
    matches = 0
    non_matches = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, row in df.iterrows():
            label = int(row.get('label', 1))
            
            # Estrai rappresentazioni (nessun blocking filter)
            craig_repr = extract_representation(row, 'craig', fields)
            us_repr = extract_representation(row, 'us', fields)
            
            if craig_repr.strip() and us_repr.strip():
                f.write(f"{craig_repr}\t{us_repr}\t{label}\n")
                if label == 1:
                    matches += 1
                else:
                    non_matches += 1
    
    return matches, non_matches


def prepare_all_datasets():
    """Prepara tutti i dataset per le 3 configurazioni senza blocking."""
    print("="*70)
    print("PREPARAZIONE DATASET SENZA BLOCKING")
    print("="*70)
    
    # Carica dataset originali
    print("\nCaricamento dataset da dataset/splits/...")
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    valid_df = pd.read_csv(SPLITS_DIR / "validation.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    
    print(f"  Train originale: {len(train_df)} coppie (tutte positive)")
    print(f"  Valid originale: {len(valid_df)} coppie (tutte positive)")
    print(f"  Test originale: {len(test_df)} coppie (tutte positive)")
    
    # Genera coppie negative
    print("\nGenerazione coppie negative (1:1 ratio)...")
    train_balanced = generate_negative_pairs(train_df, num_negatives_per_positive=1)
    valid_balanced = generate_negative_pairs(valid_df, num_negatives_per_positive=1)
    test_balanced = generate_negative_pairs(test_df, num_negatives_per_positive=1)
    
    print(f"  Train bilanciato: {len(train_balanced)} ({(train_balanced['label']==1).sum()} pos, {(train_balanced['label']==0).sum()} neg)")
    print(f"  Valid bilanciato: {len(valid_balanced)} ({(valid_balanced['label']==1).sum()} pos, {(valid_balanced['label']==0).sum()} neg)")
    print(f"  Test bilanciato: {len(test_balanced)} ({(test_balanced['label']==1).sum()} pos, {(test_balanced['label']==0).sum()} neg)")
    
    # Crea dataset per ogni pipeline (senza blocking)
    results = {}
    
    for pipeline in PIPELINES.keys():
        config_name = f"{pipeline}_NO_BLOCKING"
        print(f"\n--- {config_name} ---")
        
        # Directory
        config_dir = DITTO_DATASET_DIR / config_name
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # Crea files (senza blocking)
        train_m, train_nm = create_ditto_file_no_blocking(
            train_balanced, pipeline,
            str(config_dir / "train.txt")
        )
        valid_m, valid_nm = create_ditto_file_no_blocking(
            valid_balanced, pipeline,
            str(config_dir / "valid.txt")
        )
        test_m, test_nm = create_ditto_file_no_blocking(
            test_balanced, pipeline,
            str(config_dir / "test.txt")
        )
        
        results[config_name] = {
            'train': (train_m, train_nm),
            'valid': (valid_m, valid_nm),
            'test': (test_m, test_nm)
        }
        
        print(f"  Train: {train_m} pos, {train_nm} neg")
        print(f"  Valid: {valid_m} pos, {valid_nm} neg")
        print(f"  Test: {test_m} pos, {test_nm} neg")
    
    return results


# ===========================================================================
# TRAINING
# ===========================================================================
sys.path.insert(0, str(Path(__file__).parent))
from ditto_light.dataset import DittoDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def train_and_evaluate(config_name: str, epochs: int = 15, batch_size: int = 32) -> Dict:
    """Addestra e valuta un modello DITTO."""
    
    print(f"\n{'='*80}")
    print(f"Training: {config_name}")
    print(f"Device: {'GPU (DirectML)' if GPU_AVAILABLE else 'CPU'}")
    print(f"{'='*80}")
    
    # Paths
    config_dir = DITTO_DATASET_DIR / config_name
    model_dir = MODELS_DIR / config_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = DittoDataset(str(config_dir / "train.txt"), lm='distilbert', max_len=256)
    valid_dataset = DittoDataset(str(config_dir / "valid.txt"), lm='distilbert', max_len=256)
    test_dataset = DittoDataset(str(config_dir / "test.txt"), lm='distilbert', max_len=256)
    
    print(f"   Train: {len(train_dataset)}")
    print(f"   Valid: {len(valid_dataset)}")
    print(f"   Test: {len(test_dataset)}")
    
    # Initialize model
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2
    )
    
    device = GPU_DEVICE if GPU_AVAILABLE else torch.device('cpu')
    model = model.to(device)
    print(f"Model moved to device: {device}")
    
    # DataLoaders - usa DittoDataset.pad come collate_fn!
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
        collate_fn=DittoDataset.pad
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=DittoDataset.pad
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=DittoDataset.pad
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # Training loop
    print(f"Training ({epochs} epochs, batch_size={batch_size})...")
    start_time = time.time()
    
    best_val_f1 = 0
    best_model_state = None
    best_epoch = 0
    best_threshold = 0.5
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            outputs = model(x)
            logits = outputs.logits
            
            loss = torch.nn.CrossEntropyLoss()(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f"  Epoch {epoch+1}, step {batch_idx}, loss: {loss.item():.4f}")
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        val_probs = []
        
        with torch.no_grad():
            for batch in valid_loader:
                x, y = batch
                x = x.to(device)
                
                outputs = model(x)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)[:, 1]  # Prob of class 1
                
                val_probs.extend(probs.cpu().numpy())
                val_labels.extend(y.numpy())
        
        # Find best threshold
        best_t = 0.5
        best_f1_t = 0
        for t in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
            preds = [1 if p > t else 0 for p in val_probs]
            f1 = f1_score(val_labels, preds, zero_division=0)
            if f1 > best_f1_t:
                best_f1_t = f1
                best_t = t
        
        val_preds = [1 if p > best_t else 0 for p in val_probs]
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        val_prec = precision_score(val_labels, val_preds, zero_division=0)
        val_rec = recall_score(val_labels, val_preds, zero_division=0)
        
        print(f"  Epoch {epoch+1}: val_f1={val_f1:.4f}, val_prec={val_prec:.4f}, val_rec={val_rec:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
            best_threshold = best_t
            print(f"  -> New best model! F1={val_f1:.4f}")
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f}s")
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Save model with config name
    model_path = model_dir / f"{config_name}_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': best_epoch,
        'threshold': best_threshold
    }, model_path)
    print(f"Model saved to {model_path}")
    
    # Test evaluation with REAL predictions
    print("Computing test metrics with REAL model predictions...")
    model.eval()
    test_probs = []
    test_labels = []
    
    inference_start = time.time()
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x = x.to(device)
            
            outputs = model(x)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[:, 1]
            
            test_probs.extend(probs.cpu().numpy())
            test_labels.extend(y.numpy())
    
    inference_time = time.time() - inference_start
    
    # Compute metrics with best threshold
    test_preds = [1 if p > best_threshold else 0 for p in test_probs]
    test_f1 = f1_score(test_labels, test_preds, zero_division=0)
    test_prec = precision_score(test_labels, test_preds, zero_division=0)
    test_rec = recall_score(test_labels, test_preds, zero_division=0)
    test_acc = accuracy_score(test_labels, test_preds)
    
    print(f"TEST RESULTS (REAL): F1={test_f1:.4f} | P={test_prec:.4f} | R={test_rec:.4f} | Acc={test_acc:.4f}")
    
    return {
        'f1_score': test_f1,
        'precision': test_prec,
        'recall': test_rec,
        'accuracy': test_acc,
        'training_time': training_time,
        'inference_time': inference_time,
        'test_samples': len(test_labels),
        'threshold': best_threshold,
        'model_path': str(model_path)
    }


def save_results(results: Dict, output_file: str):
    """Salva i risultati nel file txt."""
    with open(output_file, 'w', encoding='utf-8') as f:
        device_str = "GPU (AMD DirectML)" if GPU_AVAILABLE else "CPU"
        
        f.write("="*95 + "\n")
        f.write("DITTO TRAINING RESULTS - NO BLOCKING (Solo Pipeline)\n")
        f.write("Training con predizioni REALI del modello DITTO (senza strategie di blocking)\n")
        f.write(f"Device: {device_str}\n")
        f.write("Entity Resolution for Automotive Data (Craigslist vs US Cars)\n")
        f.write("="*95 + "\n\n")
        
        for config, metrics in results.items():
            f.write(f"Configuration: {config}\n")
            f.write("-"*95 + "\n")
            for key, value in sorted(metrics.items()):
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.6f}\n")
                else:
                    f.write(f"  {key}: {value}\n")
            f.write("\n")
        
        # Ranking
        f.write("\n" + "="*95 + "\n")
        f.write("PERFORMANCE RANKING (Sorted by F1 Score)\n")
        f.write("="*95 + "\n\n")
        
        sorted_results = sorted(results.items(), key=lambda x: x[1].get('f1_score', 0), reverse=True)
        
        f.write(f"{'Rank':<6} {'Configuration':<38} {'F1':<12} {'Precision':<12} {'Recall':<12}\n")
        f.write("-"*95 + "\n")
        
        for i, (config, metrics) in enumerate(sorted_results, 1):
            f1 = metrics.get('f1_score', 0)
            prec = metrics.get('precision', 0)
            rec = metrics.get('recall', 0)
            f.write(f"{i:<6} {config:<38} {f1:<12.6f} {prec:<12.6f} {rec:<12.6f}\n")
        
        # Timing
        f.write("\n" + "="*95 + "\n")
        f.write("TIMING ANALYSIS\n")
        f.write("="*95 + "\n\n")
        
        f.write(f"{'Configuration':<38} {'Training (s)':<20} {'Inference (s)':<20}\n")
        f.write("-"*95 + "\n")
        
        for config, metrics in sorted(results.items()):
            train_time = metrics.get('training_time', 0)
            infer_time = metrics.get('inference_time', 0)
            f.write(f"{config:<38} {train_time:<20.2f} {infer_time:<20.6f}\n")
        
        # Stats
        f.write("\n" + "="*95 + "\n")
        f.write("GLOBAL STATISTICS\n")
        f.write("="*95 + "\n\n")
        
        avg_f1 = np.mean([m['f1_score'] for m in results.values()])
        avg_prec = np.mean([m['precision'] for m in results.values()])
        avg_rec = np.mean([m['recall'] for m in results.values()])
        total_time = sum([m['training_time'] for m in results.values()])
        
        f.write(f"Average F1 Score:      {avg_f1:.6f}\n")
        f.write(f"Average Precision:     {avg_prec:.6f}\n")
        f.write(f"Average Recall:        {avg_rec:.6f}\n")
        f.write(f"Total Training Time:   {total_time:.2f}s ({total_time/60:.2f} minutes)\n")
        f.write(f"Configurations:        {len(results)}\n")
        f.write(f"Device:                {device_str}\n")
        f.write(f"Blocking:              NONE (solo pipeline)\n")
        
        if GPU_AVAILABLE:
            f.write("\nGPU ACCELERATION:\n")
            f.write("  - PyTorch DirectML enabled for AMD Radeon RX 6700 XT\n")
            f.write("  - DITTO modified to support DirectML device\n")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("DITTO TRAINING - NO BLOCKING (Solo Pipeline)")
    print("="*80)
    print(f"Epochs: {args.n_epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Device: {'GPU (DirectML)' if GPU_AVAILABLE else 'CPU'}")
    print(f"Pipelines: P1_textual_core, P2_plus_location, P3_minimal_fast")
    print(f"Blocking: NONE")
    print("="*80)
    
    # Set seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Step 1: Prepare datasets
    print("\n" + "="*80)
    print("STEP 1: PREPARAZIONE DATASET (SENZA BLOCKING)")
    print("="*80)
    prepare_all_datasets()
    
    # Step 2: Train all models
    print("\n" + "="*80)
    print("STEP 2: ADDESTRAMENTO MODELLI (3 configurazioni)")
    print("="*80)
    
    configs = [
        'P1_textual_core_NO_BLOCKING',
        'P2_plus_location_NO_BLOCKING',
        'P3_minimal_fast_NO_BLOCKING',
    ]
    
    results = {}
    total_start = time.time()
    
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/3] {config}")
        try:
            metrics = train_and_evaluate(config, epochs=args.n_epochs, batch_size=args.batch_size)
            results[config] = metrics
        except Exception as e:
            print(f"Error training {config}: {e}")
            import traceback
            traceback.print_exc()
            results[config] = {
                'f1_score': 0.0, 'precision': 0.0, 'recall': 0.0,
                'accuracy': 0.0, 'training_time': 0.0, 'inference_time': 0.0,
                'test_samples': 0, 'threshold': 0.5, 'model_path': ''
            }
    
    total_time = time.time() - total_start
    
    # Step 3: Save results
    output_file = OUTPUT_DIR / "ditto_results_no_blocking.txt"
    save_results(results, str(output_file))
    
    # Find best model
    best_config = max(results.items(), key=lambda x: x[1].get('f1_score', 0))
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE - NO BLOCKING")
    print("="*80)
    print(f"Total time: {total_time/60:.2f} minutes ({total_time:.0f}s)")
    print(f"Results: {output_file}")
    print(f"Models saved in: {MODELS_DIR}")
    print(f"Best model: {best_config[0]}")
    print(f"Device: {'GPU (DirectML)' if GPU_AVAILABLE else 'CPU'}")
    print(f"Configurations: {len(results)}/3 completed")
    print("="*80)


if __name__ == "__main__":
    main()

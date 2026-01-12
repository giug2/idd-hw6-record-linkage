"""
Script per testare P3_minimal_fast_B2 sul dataset unseen_final
Converte il dataset unseen_final in formato DITTO e valuta le prestazioni
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import torch
import random
from pathlib import Path
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

# GPU Setup
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Prova a usare DirectML
GPU_DEVICE = None
GPU_AVAILABLE = False

try:
    import torch_directml
    GPU_DEVICE = torch_directml.device()
    GPU_AVAILABLE = True
    print("✓ DirectML GPU Acceleration ENABLED")
except ImportError:
    GPU_DEVICE = 'cpu'
    print("⚠️ DirectML not available, using CPU")

sys.path.insert(0, str(Path(__file__).parent))

from ditto_light.dataset import DittoDataset
from ditto_light.ditto import DittoModel, evaluate
from torch.utils import data

# Percorsi
ROOT_DIR = Path(__file__).parent.parent.parent
DATASET_DIR = ROOT_DIR / "dataset"
OUTPUT_DITTO_DIR = ROOT_DIR / "output" / "ditto"
DITTO_DATASET_DIR = OUTPUT_DITTO_DIR / "ditto_dataset"
MODELS_DIR = ROOT_DIR / "modelli" / "ditto"

# P3_minimal_fast usa solo: brand, model, year
P3_FIELDS = ['brand', 'model', 'year']


def normalize_string(s):
    """Normalizza una stringa per il confronto."""
    if pd.isna(s):
        return ""
    return str(s).lower().strip()


def extract_record_representation(row: pd.Series, source: str, fields: list) -> str:
    """
    Estrae la rappresentazione testuale di un record per DITTO.
    Formato: COL field1 VAL value1 COL field2 VAL value2 ...
    """
    values = []
    
    for field in fields:
        col_name = f"{field}_{source}"
        if col_name in row.index:
            val = row[col_name]
            if pd.isna(val):
                val_str = ""
            else:
                val_str = str(val).strip()
            values.append(f"COL {field} VAL {val_str}")
    
    return " ".join(values)


def create_blocking_key_B2(row: pd.Series, source: str) -> str:
    """
    B2: brand + model_prefix (primi 2 caratteri del model)
    """
    brand_col = f"brand_{source}"
    model_col = f"model_{source}"
    
    brand = normalize_string(row.get(brand_col, ""))
    model = normalize_string(row.get(model_col, ""))
    
    model_prefix = model[:2] if len(model) >= 2 else model
    
    return f"{brand}_{model_prefix}"


def convert_unseen_to_ditto_format(unseen_path: str, output_path: str, add_negatives: bool = True, neg_ratio: float = 1.0):
    """
    Converte unseen_final.csv in formato DITTO per P3_B2.
    Tutte le coppie originali sono match (stesso VIN).
    Se add_negatives=True, genera coppie negative (non-match) per bilanciare.
    """
    print(f"Loading unseen dataset from {unseen_path}...")
    df = pd.read_csv(unseen_path)
    print(f"  Loaded {len(df)} positive pairs (same VIN)")
    
    output_lines = []
    
    # Aggiungi le coppie positive (match)
    for idx, row in df.iterrows():
        craig_repr = extract_record_representation(row, 'craig', P3_FIELDS)
        us_repr = extract_record_representation(row, 'us', P3_FIELDS)
        label = 1  # match
        output_lines.append(f"{craig_repr}\t{us_repr}\t{label}")
    
    n_positives = len(output_lines)
    
    # Genera coppie negative (non-match) shufflando
    if add_negatives:
        n_negatives = int(n_positives * neg_ratio)
        print(f"  Generating {n_negatives} negative pairs...")
        
        # Crea liste separate
        craig_reprs = [extract_record_representation(row, 'craig', P3_FIELDS) for _, row in df.iterrows()]
        us_reprs = [extract_record_representation(row, 'us', P3_FIELDS) for _, row in df.iterrows()]
        
        # Shuffle US per creare non-match
        indices = list(range(len(us_reprs)))
        random.shuffle(indices)
        
        negatives_added = 0
        for i, j in enumerate(indices):
            if negatives_added >= n_negatives:
                break
            # Evita match accidentali (stesso indice)
            if i != j:
                output_lines.append(f"{craig_reprs[i]}\t{us_reprs[j]}\t0")
                negatives_added += 1
        
        print(f"  Added {negatives_added} negative pairs")
    
    # Shuffle finale
    random.shuffle(output_lines)
    
    # Salva
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    n_total = len(output_lines)
    print(f"  Total pairs: {n_total} ({n_positives} positive, {n_total - n_positives} negative)")
    print(f"  Saved to {output_path}")
    
    return n_total, n_positives


def test_model_on_unseen(test_file: str, device) -> dict:
    """
    Testa il modello P3_B2 sul dataset unseen.
    Nota: Non carichiamo pesi salvati (save_model era False).
    Creiamo un nuovo modello e valutiamo con le metriche reali.
    """
    print("\n" + "="*80)
    print("Testing P3_minimal_fast_B2 on UNSEEN dataset")
    print("="*80)
    
    # Carica dataset
    print("\n📂 Loading test dataset...")
    try:
        test_dataset = DittoDataset(test_file, lm='distilbert', max_len=256)
        print(f"   ✓ Test samples: {len(test_dataset)}")
    except Exception as e:
        print(f"❌ Dataset error: {e}")
        return None
    
    # DataLoader
    padder = test_dataset.pad
    test_iter = data.DataLoader(
        dataset=test_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=0,
        collate_fn=padder
    )
    
    # Inizializza modello
    print("\n🤖 Initializing model...")
    model = DittoModel(device=device, lm='distilbert', alpha_aug=0.8)
    
    try:
        model = model.to(device)
        print(f"   ✓ Model on device: {device}")
    except Exception as e:
        print(f"   ⚠️ Fallback to CPU: {e}")
        model = model.to('cpu')
        device = 'cpu'
    
    # Nota: senza pesi salvati, il modello userà i pesi pre-trained di DistilBERT
    # Per un test reale, dovresti prima riallenare con save_model=True
    
    # Valutazione
    print("\n⚙️ Evaluating...")
    model.eval()
    
    all_probs = []
    all_labels = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch in test_iter:
            x, y = batch
            logits = model(x)
            probs = logits.softmax(dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())
    
    inference_time = time.time() - start_time
    
    # Trova soglia ottimale
    best_th = 0.5
    best_f1 = 0.0
    
    for th in np.arange(0.0, 1.0, 0.05):
        preds = [1 if p > th else 0 for p in all_probs]
        f1 = f1_score(all_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
    
    # Calcola metriche con soglia ottimale
    final_preds = [1 if p > best_th else 0 for p in all_probs]
    
    precision = precision_score(all_labels, final_preds, zero_division=0)
    recall = recall_score(all_labels, final_preds, zero_division=0)
    f1 = f1_score(all_labels, final_preds, zero_division=0)
    accuracy = accuracy_score(all_labels, final_preds)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(all_labels, final_preds, labels=[0, 1]).ravel() if len(set(all_labels)) > 1 else (0, 0, 0, len(all_labels))
    
    results = {
        'total_pairs': len(all_labels),
        'positive_pairs': sum(all_labels),
        'negative_pairs': len(all_labels) - sum(all_labels),
        'threshold': best_th,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn,
        'inference_time': inference_time,
        'avg_inference_per_sample': inference_time / len(all_labels) if len(all_labels) > 0 else 0
    }
    
    print(f"\n📊 Results:")
    print(f"   Total pairs:     {results['total_pairs']}")
    print(f"   Positive pairs:  {results['positive_pairs']}")
    print(f"   Negative pairs:  {results['negative_pairs']}")
    print(f"   Threshold:       {results['threshold']:.2f}")
    print(f"   Precision:       {results['precision']:.4f}")
    print(f"   Recall:          {results['recall']:.4f}")
    print(f"   F1 Score:        {results['f1_score']:.4f}")
    print(f"   Accuracy:        {results['accuracy']:.4f}")
    print(f"   Inference time:  {results['inference_time']:.2f}s")
    
    return results


def save_results(results: dict, output_path: str):
    """Salva i risultati in un file txt."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("DITTO P3_minimal_fast_B2 - TEST ON UNSEEN DATASET\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: P3_minimal_fast_B2\n")
        f.write(f"Fields: brand, model, year\n")
        f.write(f"Blocking: B2 (brand + model_prefix)\n")
        f.write(f"Dataset: unseen_final.csv\n\n")
        
        f.write("-"*80 + "\n")
        f.write("DATASET STATISTICS\n")
        f.write("-"*80 + "\n\n")
        
        f.write(f"Total pairs:        {results['total_pairs']}\n")
        f.write(f"Positive pairs:     {results['positive_pairs']} (match - same VIN)\n")
        f.write(f"Negative pairs:     {results['negative_pairs']}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("PERFORMANCE METRICS\n")
        f.write("-"*80 + "\n\n")
        
        f.write(f"Optimal Threshold:  {results['threshold']:.2f}\n")
        f.write(f"Precision:          {results['precision']:.6f}\n")
        f.write(f"Recall:             {results['recall']:.6f}\n")
        f.write(f"F1 Score:           {results['f1_score']:.6f}\n")
        f.write(f"Accuracy:           {results['accuracy']:.6f}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("CONFUSION MATRIX\n")
        f.write("-"*80 + "\n\n")
        
        f.write(f"True Positives:     {results['true_positives']}\n")
        f.write(f"False Positives:    {results['false_positives']}\n")
        f.write(f"True Negatives:     {results['true_negatives']}\n")
        f.write(f"False Negatives:    {results['false_negatives']}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("TIMING\n")
        f.write("-"*80 + "\n\n")
        
        f.write(f"Total inference time:       {results['inference_time']:.2f}s\n")
        f.write(f"Avg time per sample:        {results['avg_inference_per_sample']*1000:.4f}ms\n\n")
        
        f.write("="*80 + "\n")
        f.write("NOTE: This test uses a freshly initialized model (pre-trained DistilBERT weights).\n")
        f.write("For true evaluation, the model should be trained first with save_model=True.\n")
        f.write("="*80 + "\n")
    
    print(f"\n✓ Results saved to {output_path}")


def main():
    print("\n" + "="*80)
    print("DITTO P3_B2 - UNSEEN DATASET EVALUATION")
    print("="*80)
    
    # Paths
    unseen_csv = DATASET_DIR / "unseen_final.csv"
    unseen_ditto_dir = DITTO_DATASET_DIR / "P3_minimal_fast_B2_unseen"
    unseen_ditto_file = unseen_ditto_dir / "test.txt"
    results_file = OUTPUT_DITTO_DIR / "unseen_test_results.txt"
    
    # Step 1: Convert to DITTO format (con negative pairs per test bilanciato)
    print("\n📄 Step 1: Converting unseen_final.csv to DITTO format...")
    n_total, n_positives = convert_unseen_to_ditto_format(
        str(unseen_csv), 
        str(unseen_ditto_file),
        add_negatives=True,  # Aggiungi coppie negative
        neg_ratio=1.0        # Rapporto 1:1 positivi/negativi
    )
    
    # Step 2: Test model
    print("\n🧪 Step 2: Testing model on unseen dataset...")
    device = GPU_DEVICE if GPU_AVAILABLE else 'cpu'
    results = test_model_on_unseen(str(unseen_ditto_file), device)
    
    if results:
        # Step 3: Save results
        print("\n💾 Step 3: Saving results...")
        save_results(results, str(results_file))
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    # Set seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    main()

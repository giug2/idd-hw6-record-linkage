"""
Test P3_minimal_fast_B2 sul dataset unsee_completo.csv
======================================================
Usa il modello già addestrato (model.pt) senza riaddestrare.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import torch
import re
from pathlib import Path
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix


# GPU Setup
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

GPU_DEVICE = None
GPU_AVAILABLE = False

try:
    import torch_directml
    GPU_DEVICE = torch_directml.device()
    GPU_AVAILABLE = True
    print(" DirectML GPU Acceleration ENABLED")
except ImportError:
    GPU_DEVICE = torch.device('cpu')
    print(" DirectML not available, using CPU")

sys.path.insert(0, str(Path(__file__).parent))


from ditto_light.dataset import DittoDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Percorsi
ROOT_DIR = Path(__file__).parent.parent.parent
DATASET_DIR = ROOT_DIR / "dataset"
OUTPUT_DITTO_DIR = ROOT_DIR / "output" / "ditto"
MODELS_DIR = OUTPUT_DITTO_DIR / "modelli"

# Configurazione
MODEL_NAME = "P3_minimal_fast_B2"
P3_FIELDS = ['brand', 'model', 'year']


def normalize_string(s):
    """Normalizza una stringa."""
    if pd.isna(s):
        return ""
    return str(s).lower().strip()


def get_model_prefix(model, length=2):
    """Estrae il prefisso del modello per blocking B2."""
    s = normalize_string(model)
    s = re.sub(r'[^a-z0-9]', '', s)
    return s[:length] if len(s) >= length else s


def extract_representation(row: pd.Series, source: str, fields: list) -> str:
    """Estrae la rappresentazione testuale per DITTO."""
    values = []
    for field in fields:
        if source == 'craig':
            col_name = f"{field}_craig"
        else:
            col_name = field
        
        if col_name in row.index:
            val = row[col_name]
            if pd.isna(val):
                val_str = ""
            else:
                val_str = str(val).strip()
            values.append(val_str)
    return " ".join(filter(None, values))


def get_blocking_key_B2_craig(row: pd.Series) -> str:
    """B2: brand + model_prefix per Craig"""
    brand = normalize_string(row.get('brand_craig', ""))
    brand = re.sub(r'[^a-z0-9]', '', brand)
    model_prefix = get_model_prefix(row.get('model_craig', ""))
    return f"{brand}_{model_prefix}" if brand and model_prefix else None


def get_blocking_key_B2_us(row: pd.Series) -> str:
    """B2: brand + model_prefix per US"""
    brand = normalize_string(row.get('brand', ""))
    brand = re.sub(r'[^a-z0-9]', '', brand)
    model_prefix = get_model_prefix(row.get('model', ""))
    return f"{brand}_{model_prefix}" if brand and model_prefix else None


def prepare_unseen_dataset(unseen_path: str, output_path: str):
    """
    Prepara il dataset unseen_completo nel formato DITTO.
    Tutte le coppie sono match (stesso VIN) -> label=1
    Applichiamo blocking B2 per filtrare.
    """
    print(f"Caricamento {unseen_path}...")
    df = pd.read_csv(unseen_path)
    print(f"  Righe totali: {len(df)}")
    
    passed_blocking = 0
    skipped_blocking = 0
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, row in df.iterrows():
            # Blocking B2
            craig_key = get_blocking_key_B2_craig(row)
            us_key = get_blocking_key_B2_us(row)
            
            # Tutte le coppie sono match (stesso VIN)
            # Ma filtriamo solo quelle che passano il blocking
            if craig_key and us_key and craig_key == us_key:
                craig_repr = extract_representation(row, 'craig', P3_FIELDS)
                us_repr = extract_representation(row, 'us', P3_FIELDS)
                
                if craig_repr.strip() and us_repr.strip():
                    f.write(f"{craig_repr}\t{us_repr}\t1\n")
                    passed_blocking += 1
            else:
                skipped_blocking += 1
    
    print(f"  Passano blocking B2: {passed_blocking}")
    print(f"  Scartate dal blocking: {skipped_blocking}")
    return passed_blocking


def load_model(model_dir: Path, device, model_name: str):
    """Carica il modello salvato."""
    model_path = model_dir / f"{model_name}_model.pt"
    print(f"Caricamento modello da {model_path}...")
    
    # Carica il checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Inizializza il modello
    model = AutoModelForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2
    )
    
    # Carica i pesi
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    threshold = checkpoint.get('threshold', 0.5)
    print(f"  Threshold: {threshold}")
    
    return model, threshold


def evaluate_on_unseen(model, threshold, test_file: str, device, batch_size: int = 32):
    """Valuta il modello sul dataset unseen."""
    print(f"Caricamento dataset da {test_file}...")
    
    # Carica dataset
    dataset = DittoDataset(test_file, lm='distilbert', max_len=256)
    print(f"  Campioni: {len(dataset)}")
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=DittoDataset.pad
    )
    
    # Inference
    print("Esecuzione inferenza...")
    all_probs = []
    all_labels = []
    
    start_time = time.time()
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = x.to(device)
            
            outputs = model(x)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1]
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.numpy())
    
    inference_time = time.time() - start_time
    
    # Calcola metriche
    predictions = [1 if p > threshold else 0 for p in all_probs]
    
    accuracy = accuracy_score(all_labels, predictions)
    precision = precision_score(all_labels, predictions, zero_division=0)
    recall = recall_score(all_labels, predictions, zero_division=0)
    f1 = f1_score(all_labels, predictions, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'threshold': threshold,
        'total_samples': len(all_labels),
        'predicted_matches': sum(predictions),
        'actual_matches': sum(all_labels),
        'inference_time': inference_time,
        'confusion_matrix': cm
    }


def save_results(results: dict, output_file: str, model_name: str, dataset_name: str):
    """Salva i risultati nel file txt."""
    with open(output_file, 'w', encoding='utf-8') as f:
        device_str = "GPU (AMD DirectML)" if GPU_AVAILABLE else "CPU"
        
        f.write("="*80 + "\n")
        f.write("DITTO UNSEEN COMPLETO TEST RESULTS\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Device: {device_str}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write("DATASET INFO:\n")
        f.write("-"*80 + "\n")
        f.write(f"  Source: {dataset_name}\n")
        f.write(f"  Total samples (after blocking): {results['total_samples']}\n")
        f.write(f"  All samples are TRUE matches (same VIN)\n\n")
        
        f.write("MODEL INFO:\n")
        f.write("-"*80 + "\n")
        f.write(f"  Model: {model_name}\n")
        f.write(f"  Pipeline: P3_minimal_fast (brand, model, year)\n")
        f.write(f"  Blocking: B2 (brand + model_prefix)\n")
        f.write(f"  Threshold: {results['threshold']:.2f}\n")
        f.write(f"  Training: 15 epochs (pre-trained model)\n\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write("-"*80 + "\n")
        f.write(f"  Accuracy:   {results['accuracy']:.6f}\n")
        f.write(f"  Precision:  {results['precision']:.6f}\n")
        f.write(f"  Recall:     {results['recall']:.6f}\n")
        f.write(f"  F1 Score:   {results['f1_score']:.6f}\n\n")
        
        f.write("PREDICTION SUMMARY:\n")
        f.write("-"*80 + "\n")
        f.write(f"  Total pairs evaluated: {results['total_samples']}\n")
        f.write(f"  Actual matches (ground truth): {results['actual_matches']}\n")
        f.write(f"  Predicted as match: {results['predicted_matches']}\n")
        f.write(f"  Correctly identified: {results['predicted_matches']} / {results['actual_matches']}\n\n")
        
        f.write("CONFUSION MATRIX:\n")
        f.write("-"*80 + "\n")
        cm = results['confusion_matrix']
        f.write(f"                    Predicted\n")
        f.write(f"                    Non-Match    Match\n")
        if cm.shape[0] > 1:
            f.write(f"  Actual Non-Match     {cm[0][0]:<10}   {cm[0][1]}\n")
            f.write(f"  Actual Match         {cm[1][0]:<10}   {cm[1][1]}\n\n")
        else:
            f.write(f"  Actual Match         {0:<10}   {cm[0][0] if cm[0][0] else 0}\n\n")
        
        f.write("TIMING:\n")
        f.write("-"*80 + "\n")
        f.write(f"  Inference time: {results['inference_time']:.2f}s\n")
        f.write(f"  Throughput: {results['total_samples'] / results['inference_time']:.1f} pairs/second\n\n")
        
        f.write("NOTE:\n")
        f.write("-"*80 + "\n")
        f.write("  Il dataset unsee_completo.csv contiene coppie con lo stesso VIN,\n")
        f.write("  quindi tutte le coppie sono TRUE MATCHES.\n")
        f.write("  La Recall indica quante coppie vere il modello identifica correttamente.\n")
        f.write("  Un Recall < 1.0 indica False Negatives (match non riconosciuti).\n")


def main():
    print("\n" + "="*80)
    print("TEST P3_minimal_fast_B2 SU DATASET UNSEEN_COMPLETO")
    print("="*80)
    print(f"Modello: {MODEL_NAME}")
    print(f"Device: {'GPU (DirectML)' if GPU_AVAILABLE else 'CPU'}")
    
    # Paths
    unseen_csv = DATASET_DIR / "unsee_completo.csv"
    unseen_ditto = OUTPUT_DITTO_DIR / "ditto_dataset" / f"{MODEL_NAME}_unseen_completo" / "test.txt"
    model_dir = MODELS_DIR / MODEL_NAME
    output_file = OUTPUT_DITTO_DIR / "unseen_completo_test_results.txt"
    
    # Crea directory se necessario
    unseen_ditto.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepara dataset unseen in formato DITTO
    print("\n" + "-"*80)
    print("STEP 1: Preparazione dataset unseen_completo")
    print("-"*80)
    num_samples = prepare_unseen_dataset(str(unseen_csv), str(unseen_ditto))
    
    if num_samples == 0:
        print("ERRORE: Nessun campione passa il blocking!")
        return
    
    # Carica modello
    print("\n" + "-"*80)
    print("STEP 2: Caricamento modello (pre-trained, 15 epochs)")
    print("-"*80)
    device = GPU_DEVICE if GPU_AVAILABLE else torch.device('cpu')
    model, threshold = load_model(model_dir, device, MODEL_NAME)
    
    # Valuta
    print("\n" + "-"*80)
    print("STEP 3: Valutazione su unseen_completo")
    print("-"*80)
    results = evaluate_on_unseen(model, threshold, str(unseen_ditto), device)
    
    # Salva risultati
    print("\n" + "-"*80)
    print("STEP 4: Salvataggio risultati")
    print("-"*80)
    save_results(results, str(output_file), MODEL_NAME, "dataset/unsee_completo.csv")
    print(f"Risultati salvati in: {output_file}")
    
    # Stampa riepilogo
    print("\n" + "="*80)
    print("RISULTATI UNSEEN_COMPLETO TEST")
    print("="*80)
    print(f"Modello:    {MODEL_NAME}")
    print(f"Campioni:   {results['total_samples']}")
    print(f"F1 Score:   {results['f1_score']:.6f}")
    print(f"Precision:  {results['precision']:.6f}")
    print(f"Recall:     {results['recall']:.6f}")
    print(f"Accuracy:   {results['accuracy']:.6f}")
    print("="*80)


if __name__ == "__main__":
    main()

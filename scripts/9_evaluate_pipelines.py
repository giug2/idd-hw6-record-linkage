"""
Valutazione delle Pipeline di Record Linkage

Questo script valuta le prestazioni di 2 pipeline:
- B1-RecordLinkage: Blocking (brand, year) + RecordLinkage
- B2-RecordLinkage: Blocking (VIN prefix) + RecordLinkage

Metriche calcolate:
- Precision
- Recall
- F1-measure
- Tempo di training
- Tempo di inferenza
- Statistiche di blocking (reduction ratio, coppie candidate)
"""

import pandas as pd
import numpy as np
import recordlinkage
from recordlinkage.index import Block
import time
import os
import re
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
SPLITS_DIR = os.path.join(DATA_DIR, 'splits')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# Crea directory output se non esiste
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_data():
    """Carica i dataset train, validation e test."""
    print("\n" + "="*60)
    print("CARICAMENTO DATI")
    print("="*60)
    
    train_df = pd.read_csv(os.path.join(SPLITS_DIR, 'train.csv'), low_memory=False)
    val_df = pd.read_csv(os.path.join(SPLITS_DIR, 'validation.csv'), low_memory=False)
    test_df = pd.read_csv(os.path.join(SPLITS_DIR, 'test.csv'), low_memory=False)
    
    print(f"  Train set:      {len(train_df):,} record")
    print(f"  Validation set: {len(val_df):,} record")
    print(f"  Test set:       {len(test_df):,} record")
    print(f"  Totale:         {len(train_df) + len(val_df) + len(test_df):,} record")
    
    return train_df, val_df, test_df


def normalize_brand(brand):
    """Normalizza il nome del brand per uniformità."""
    if pd.isna(brand) or brand is None:
        return "unknown"
    
    brand = str(brand).lower().strip()
    
    brand_mapping = {
        'chevrolet': 'chevrolet', 'chevy': 'chevrolet',
        'mercedes-benz': 'mercedes-benz', 'mercedes': 'mercedes-benz',
        'volkswagen': 'volkswagen', 'vw': 'volkswagen',
        'land rover': 'land rover', 'landrover': 'land rover',
        'alfa romeo': 'alfa romeo', 'alfa-romeo': 'alfa romeo',
        'rolls-royce': 'rolls-royce', 'rolls royce': 'rolls-royce',
        'aston martin': 'aston martin', 'aston-martin': 'aston martin',
    }
    
    return brand_mapping.get(brand, brand)


def get_vin_prefix(vin, length=8):
    """Estrae il prefisso VIN (8 caratteri = WMI + VDS parziale)."""
    if pd.isna(vin) or vin is None:
        return None
    
    vin = str(vin).upper().strip()
    vin = re.sub(r'[^A-Z0-9]', '', vin)
    
    if len(vin) < length:
        return None
    
    return vin[:length]


def prepare_dataframes_for_linkage(df):
    """
    Prepara due DataFrame separati per le due sorgenti dal dataset combinato.
    
    Il dataset originale ha record già allineati (ogni riga contiene dati
    da entrambe le sorgenti). Creiamo due DataFrame separati per simulare
    il task di record linkage.
    
    Returns:
        craig_df: DataFrame con record da Craigslist
        us_df: DataFrame con record da US Used Cars
        true_links: MultiIndex con le coppie vere (ground truth)
    """
    # Colonne per Craigslist (suffisso _craig)
    craig_df = pd.DataFrame({
        'source_id': df['source_id_craig'].values,
        'vin': df['vin'].values,
        'brand': df['brand_craig'].values,
        'model': df['model_craig'].values,
        'year': df['year_craig'].values,
        'price': df['price_craig'].values,
        'mileage': df['mileage_craig'].values,
        'color': df['color_craig'].values if 'color_craig' in df.columns else None,
    })
    craig_df.index = pd.Index([f'craig_{i}' for i in range(len(craig_df))], name='id')
    
    # Colonne per US Used Cars (senza suffisso o con suffisso _us)
    us_df = pd.DataFrame({
        'source_id': df['source_id_us'].values if 'source_id_us' in df.columns else df['source_id'].values,
        'vin': df['vin_us'].values if 'vin_us' in df.columns else df['vin'].values,
        'brand': df['brand'].values,
        'model': df['model'].values,
        'year': df['year'].values,
        'price': df['price'].values,
        'mileage': df['mileage'].values,
        'color': df['color'].values if 'color' in df.columns else None,
    })
    us_df.index = pd.Index([f'us_{i}' for i in range(len(us_df))], name='id')
    
    # Ground truth: ogni riga i del dataset originale è un match (craig_i, us_i)
    true_pairs = [(f'craig_{i}', f'us_{i}') for i in range(len(df))]
    true_links = pd.MultiIndex.from_tuples(true_pairs, names=['id_1', 'id_2'])
    
    return craig_df, us_df, true_links


def calculate_metrics(predicted_pairs, true_links):
    """
    Calcola precision, recall e F1-measure.
    
    Args:
        predicted_pairs: coppie predette dal modello
        true_links: coppie vere (ground truth)
    
    Returns:
        precision, recall, f1, true_positives, false_positives, false_negatives
    """
    if len(predicted_pairs) == 0:
        return 0.0, 0.0, 0.0, 0, 0, len(true_links)
    
    pred_set = set(predicted_pairs)
    true_set = set(true_links)
    
    true_positives = len(pred_set.intersection(true_set))
    false_positives = len(pred_set - true_set)
    false_negatives = len(true_set - pred_set)
    
    precision = true_positives / len(pred_set) if len(pred_set) > 0 else 0
    recall = true_positives / len(true_set) if len(true_set) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1, true_positives, false_positives, false_negatives


# ============================================================================
# BLOCKING STRATEGIES
# ============================================================================

def blocking_B1(craig_df, us_df):
    """
    Strategia B1: Blocking su (brand normalizzato, year).
    
    Crea blocchi basati sulla combinazione di brand e anno.
    """
    c_df = craig_df.copy()
    u_df = us_df.copy()
    
    # Normalizza brand
    c_df['brand_norm'] = c_df['brand'].apply(normalize_brand)
    u_df['brand_norm'] = u_df['brand'].apply(normalize_brand)
    
    # Normalizza year come stringa
    c_df['year_str'] = c_df['year'].apply(lambda x: str(int(x)) if pd.notna(x) else 'unknown')
    u_df['year_str'] = u_df['year'].apply(lambda x: str(int(x)) if pd.notna(x) else 'unknown')
    
    # Crea blocking key combinata
    c_df['block_key'] = c_df['brand_norm'] + '_' + c_df['year_str']
    u_df['block_key'] = u_df['brand_norm'] + '_' + u_df['year_str']
    
    # Usa recordlinkage Block
    indexer = recordlinkage.Index()
    indexer.block('block_key')
    candidate_pairs = indexer.index(c_df, u_df)
    
    return candidate_pairs, c_df, u_df


def blocking_B2(craig_df, us_df):
    """
    Strategia B2: Blocking su VIN prefix (8 caratteri).
    
    Crea blocchi basati sui primi 8 caratteri del VIN.
    """
    c_df = craig_df.copy()
    u_df = us_df.copy()
    
    # Estrai prefisso VIN
    c_df['vin_prefix'] = c_df['vin'].apply(get_vin_prefix)
    u_df['vin_prefix'] = u_df['vin'].apply(get_vin_prefix)
    
    # Filtra record con VIN valido
    c_valid = c_df[c_df['vin_prefix'].notna()].copy()
    u_valid = u_df[u_df['vin_prefix'].notna()].copy()
    
    # Usa recordlinkage Block
    indexer = recordlinkage.Index()
    indexer.block('vin_prefix')
    candidate_pairs = indexer.index(c_valid, u_valid)
    
    return candidate_pairs, c_df, u_df


def analyze_blocking(candidate_pairs, craig_df, us_df, true_links, name):
    """Analizza le statistiche della strategia di blocking."""
    n_candidates = len(candidate_pairs)
    n_craig = len(craig_df)
    n_us = len(us_df)
    n_total_pairs = n_craig * n_us
    
    # Reduction ratio
    reduction_ratio = 1 - (n_candidates / n_total_pairs) if n_total_pairs > 0 else 0
    
    # Pairs completeness (quante coppie vere sono nei candidati)
    true_set = set(true_links)
    candidates_set = set(candidate_pairs)
    true_in_candidates = len(true_set.intersection(candidates_set))
    pairs_completeness = true_in_candidates / len(true_set) if len(true_set) > 0 else 0
    
    print(f"\n  Statistiche Blocking {name}:")
    print(f"    Record Craigslist:     {n_craig:,}")
    print(f"    Record US Used Cars:   {n_us:,}")
    print(f"    Coppie totali:         {n_total_pairs:,}")
    print(f"    Coppie candidate:      {n_candidates:,}")
    print(f"    Reduction Ratio:       {reduction_ratio:.4f} ({reduction_ratio*100:.2f}%)")
    print(f"    Coppie vere totali:    {len(true_links):,}")
    print(f"    Coppie vere in cand.:  {true_in_candidates:,}")
    print(f"    Pairs Completeness:    {pairs_completeness:.4f} ({pairs_completeness*100:.2f}%)")
    
    return {
        'n_candidates': n_candidates,
        'n_total_pairs': n_total_pairs,
        'reduction_ratio': reduction_ratio,
        'pairs_completeness': pairs_completeness,
        'true_in_candidates': true_in_candidates
    }


# ============================================================================
# COMPARISON RULES
# ============================================================================

def create_comparison_rules():
    """
    Definisce le regole di comparazione per il record linkage.
    
    Regole:
    - VIN: exact match (identificatore univoco)
    - Brand: string similarity (Jaro-Winkler)
    - Model: string similarity (Jaro-Winkler, più permissivo)
    - Year: exact match
    - Price: numeric comparison con threshold
    - Mileage: numeric comparison con threshold
    - Color: exact match
    """
    compare = recordlinkage.Compare()
    
    # VIN: match esatto (molto importante se disponibile)
    compare.exact('vin', 'vin', label='vin_exact')
    
    # Brand: similarità stringa (Jaro-Winkler con soglia alta)
    compare.string('brand', 'brand', method='jarowinkler', threshold=0.85, label='brand_sim')
    
    # Model: similarità stringa (più permissivo perché può variare)
    compare.string('model', 'model', method='jarowinkler', threshold=0.75, label='model_sim')
    
    # Year: match esatto
    compare.exact('year', 'year', label='year_exact')
    
    # Price: confronto numerico (con tolleranza - scala gaussiana)
    compare.numeric('price', 'price', method='gauss', scale=5000, label='price_sim')
    
    # Mileage: confronto numerico (con tolleranza)
    compare.numeric('mileage', 'mileage', method='gauss', scale=10000, label='mileage_sim')
    
    # Color: match esatto
    compare.exact('color', 'color', label='color_exact')
    
    return compare


# ============================================================================
# PIPELINE: RecordLinkage
# ============================================================================

def run_recordlinkage_pipeline(train_df, test_df, blocking_strategy, pipeline_name):
    """
    Esegue la pipeline di Record Linkage completa.
    
    Fasi:
    1. Preparazione dati
    2. Blocking
    3. Calcolo features (comparison)
    4. Training classificatore
    5. Predizione su test set
    6. Calcolo metriche
    
    Args:
        train_df: DataFrame di training
        test_df: DataFrame di test
        blocking_strategy: funzione di blocking (blocking_B1 o blocking_B2)
        pipeline_name: nome della pipeline per logging
    
    Returns:
        dict con risultati e metriche
    """
    print(f"\n{'='*70}")
    print(f"PIPELINE: {pipeline_name}")
    print(f"{'='*70}")
    
    results = {
        'pipeline': pipeline_name,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # ========== PREPARAZIONE DATI ==========
    print("\n[1/5] Preparazione dati...")
    c_train, u_train, true_train = prepare_dataframes_for_linkage(train_df)
    c_test, u_test, true_test = prepare_dataframes_for_linkage(test_df)
    
    results['train_size'] = len(train_df)
    results['test_size'] = len(test_df)
    results['true_matches_train'] = len(true_train)
    results['true_matches_test'] = len(true_test)
    
    # ========== BLOCKING - TRAINING ==========
    print("\n[2/5] Blocking (Training)...")
    start_blocking_train = time.time()
    train_pairs, c_train_blocked, u_train_blocked = blocking_strategy(c_train, u_train)
    blocking_train_time = time.time() - start_blocking_train
    
    blocking_stats_train = analyze_blocking(
        train_pairs, c_train, u_train, true_train, 
        f"{pipeline_name} - Training"
    )
    results['blocking_train_time'] = blocking_train_time
    results['train_candidates'] = blocking_stats_train['n_candidates']
    results['train_reduction_ratio'] = blocking_stats_train['reduction_ratio']
    results['train_pairs_completeness'] = blocking_stats_train['pairs_completeness']
    
    # ========== TRAINING ==========
    print("\n[3/5] Training classificatore...")
    start_train = time.time()
    # ========== TRAINING ==========
    print("\n[3/5] Training classificatore...")
    start_train = time.time()
    
    # Crea regole di comparazione
    compare = create_comparison_rules()
    
    # Calcola features di training
    print("  Calcolo features di training...")
    features_train = compare.compute(train_pairs, c_train_blocked, u_train_blocked)
    print(f"    Features shape: {features_train.shape}")
    
    # Match index per training (intersezione con true links)
    match_index_train = true_train.intersection(train_pairs)
    print(f"    True matches nei candidati: {len(match_index_train):,} / {len(true_train):,}")
    
    # Feature statistics
    print(f"    Feature means:")
    for col in features_train.columns:
        print(f"      {col}: {features_train[col].mean():.4f}")
    
    # Addestra classificatore Logistic Regression
    print("  Addestramento Logistic Regression...")
    classifier = recordlinkage.LogisticRegressionClassifier()
    classifier.fit(features_train, match_index_train)
    
    training_time = time.time() - start_train
    results['training_time'] = training_time
    print(f"  Training completato in {training_time:.2f}s")
    
    # ========== BLOCKING - TEST ==========
    print("\n[4/5] Blocking (Test)...")
    start_blocking_test = time.time()
    test_pairs, c_test_blocked, u_test_blocked = blocking_strategy(c_test, u_test)
    blocking_test_time = time.time() - start_blocking_test
    
    blocking_stats_test = analyze_blocking(
        test_pairs, c_test, u_test, true_test,
        f"{pipeline_name} - Test"
    )
    results['blocking_test_time'] = blocking_test_time
    results['test_candidates'] = blocking_stats_test['n_candidates']
    results['test_reduction_ratio'] = blocking_stats_test['reduction_ratio']
    results['test_pairs_completeness'] = blocking_stats_test['pairs_completeness']
    
    # ========== INFERENCE ==========
    print("\n[5/5] Inference e valutazione...")
    start_inference = time.time()
    
    # Calcola features di test
    print("  Calcolo features di test...")
    features_test = compare.compute(test_pairs, c_test_blocked, u_test_blocked)
    print(f"    Features shape: {features_test.shape}")
    
    # Predici con probabilità
    print("  Predizione...")
    proba = classifier.prob(features_test)
    print(f"    Probabilità: min={proba.min():.4f}, max={proba.max():.4f}, mean={proba.mean():.4f}")
    
    # Applica soglia
    threshold = 0.5
    predictions = features_test.index[proba >= threshold]
    print(f"    Predizioni con soglia {threshold}: {len(predictions):,}")
    
    # Se nessuna predizione, prova soglia più bassa
    if len(predictions) == 0:
        threshold = 0.3
        predictions = features_test.index[proba >= threshold]
        print(f"    Predizioni con soglia {threshold}: {len(predictions):,}")
        results['threshold_used'] = threshold
    else:
        results['threshold_used'] = 0.5
    
    inference_time = time.time() - start_inference
    results['inference_time'] = inference_time
    results['total_inference_time'] = blocking_test_time + inference_time
    print(f"  Inference completato in {inference_time:.2f}s")
    
    # ========== METRICHE ==========
    print("\n  Calcolo metriche...")
    precision, recall, f1, tp, fp, fn = calculate_metrics(predictions, true_test)
    
    results['precision'] = precision
    results['recall'] = recall
    results['f1'] = f1
    results['true_positives'] = tp
    results['false_positives'] = fp
    results['false_negatives'] = fn
    results['predictions'] = len(predictions)
    
    # ========== RIEPILOGO ==========
    print(f"\n{'='*50}")
    print(f"RISULTATI - {pipeline_name}")
    print(f"{'='*50}")
    print(f"  Metriche di Valutazione:")
    print(f"    Precision:        {precision:.4f} ({precision*100:.2f}%)")
    print(f"    Recall:           {recall:.4f} ({recall*100:.2f}%)")
    print(f"    F1-measure:       {f1:.4f} ({f1*100:.2f}%)")
    print(f"")
    print(f"  Dettagli Predizioni:")
    print(f"    True Positives:   {tp:,}")
    print(f"    False Positives:  {fp:,}")
    print(f"    False Negatives:  {fn:,}")
    print(f"    Totale Predetti:  {len(predictions):,}")
    print(f"")
    print(f"  Tempi di Esecuzione:")
    print(f"    Training time:    {training_time:.2f}s")
    print(f"    Inference time:   {inference_time:.2f}s")
    print(f"    Blocking (train): {blocking_train_time:.2f}s")
    print(f"    Blocking (test):  {blocking_test_time:.2f}s")
    print(f"    Totale:           {training_time + inference_time + blocking_train_time + blocking_test_time:.2f}s")
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Funzione principale per la valutazione delle pipeline."""
    
    print("\n" + "="*70)
    print("   VALUTAZIONE PIPELINE DI RECORD LINKAGE")
    print("   B1-RecordLinkage e B2-RecordLinkage")
    print("="*70)
    print(f"   Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Carica dati
    train_df, val_df, test_df = load_data()
    
    # Combina train + validation per training più robusto
    train_full = pd.concat([train_df, val_df], ignore_index=True)
    print(f"\n  Training set combinato (train+val): {len(train_full):,} record")
    
    all_results = []
    
    # ========== PIPELINE 1: B1-RecordLinkage ==========
    try:
        results_b1 = run_recordlinkage_pipeline(
            train_full, test_df, 
            blocking_B1, 
            "B1-RecordLinkage"
        )
        all_results.append(results_b1)
    except Exception as e:
        print(f"\n  ERRORE in B1-RecordLinkage: {e}")
        import traceback
        traceback.print_exc()
        all_results.append({
            'pipeline': 'B1-RecordLinkage',
            'error': str(e)
        })
    
    # ========== PIPELINE 2: B2-RecordLinkage ==========
    try:
        results_b2 = run_recordlinkage_pipeline(
            train_full, test_df, 
            blocking_B2, 
            "B2-RecordLinkage"
        )
        all_results.append(results_b2)
    except Exception as e:
        print(f"\n  ERRORE in B2-RecordLinkage: {e}")
        import traceback
        traceback.print_exc()
        all_results.append({
            'pipeline': 'B2-RecordLinkage',
            'error': str(e)
        })
    
    # ========== RIEPILOGO FINALE ==========
    print("\n" + "="*80)
    print("   RIEPILOGO FINALE - CONFRONTO PIPELINE")
    print("="*80)
    
    results_df = pd.DataFrame(all_results)
    
    # Colonne principali da visualizzare
    main_cols = [
        'pipeline', 'precision', 'recall', 'f1', 
        'training_time', 'inference_time',
        'train_candidates', 'test_candidates',
        'predictions', 'true_positives', 'false_positives', 'false_negatives'
    ]
    
    available_cols = [c for c in main_cols if c in results_df.columns]
    
    print("\n  Metriche Principali:")
    print("-"*80)
    
    for _, row in results_df.iterrows():
        if 'error' in row and pd.notna(row.get('error')):
            print(f"\n  {row['pipeline']}: ERRORE - {row['error']}")
            continue
        
        print(f"\n  {row['pipeline']}:")
        print(f"    Precision:      {row.get('precision', 0):.4f}")
        print(f"    Recall:         {row.get('recall', 0):.4f}")
        print(f"    F1-measure:     {row.get('f1', 0):.4f}")
        print(f"    Training Time:  {row.get('training_time', 0):.2f}s")
        print(f"    Inference Time: {row.get('inference_time', 0):.2f}s")
        print(f"    Candidate Pairs (test): {row.get('test_candidates', 0):,}")
        print(f"    Predictions:    {row.get('predictions', 0):,}")
    
    # Tabella comparativa
    print("\n" + "-"*80)
    print("  Tabella Comparativa:")
    print("-"*80)
    
    if len([r for r in all_results if 'error' not in r or pd.isna(r.get('error'))]) > 0:
        comparison_cols = ['pipeline', 'precision', 'recall', 'f1', 'training_time', 'inference_time']
        comparison_cols = [c for c in comparison_cols if c in results_df.columns]
        print(results_df[comparison_cols].to_string(index=False))
    
    # Salva risultati
    output_file = os.path.join(OUTPUT_DIR, 'pipeline_evaluation_results.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\n  Risultati salvati in: {output_file}")
    
    # Genera report markdown
    generate_markdown_report(all_results)
    
    return results_df


def generate_markdown_report(results):
    """Genera un report in formato Markdown."""
    
    report_file = os.path.join(OUTPUT_DIR, 'EVALUATION_RESULTS.md')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Risultati Valutazione Pipeline di Record Linkage\n\n")
        f.write(f"**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Sommario\n\n")
        f.write("Questo documento riporta i risultati della valutazione delle pipeline di Record Linkage:\n\n")
        f.write("1. **B1-RecordLinkage**: Blocking su (brand, year) + RecordLinkage\n")
        f.write("2. **B2-RecordLinkage**: Blocking su VIN prefix (8 caratteri) + RecordLinkage\n\n")
        
        f.write("## Metriche di Valutazione\n\n")
        f.write("| Pipeline | Precision | Recall | F1-measure |\n")
        f.write("|----------|-----------|--------|------------|\n")
        
        for r in results:
            if 'error' in r and pd.notna(r.get('error')):
                f.write(f"| {r['pipeline']} | ERROR | ERROR | ERROR |\n")
            else:
                f.write(f"| {r['pipeline']} | {r.get('precision', 0):.4f} | {r.get('recall', 0):.4f} | {r.get('f1', 0):.4f} |\n")
        
        f.write("\n## Tempi di Esecuzione\n\n")
        f.write("| Pipeline | Training Time (s) | Inference Time (s) |\n")
        f.write("|----------|-------------------|--------------------|\n")
        
        for r in results:
            if 'error' in r and pd.notna(r.get('error')):
                f.write(f"| {r['pipeline']} | ERROR | ERROR |\n")
            else:
                f.write(f"| {r['pipeline']} | {r.get('training_time', 0):.2f} | {r.get('inference_time', 0):.2f} |\n")
        
        f.write("\n## Statistiche Blocking\n\n")
        f.write("| Pipeline | Candidate Pairs (Test) | Reduction Ratio | Pairs Completeness |\n")
        f.write("|----------|------------------------|-----------------|--------------------|\n")
        
        for r in results:
            if 'error' in r and pd.notna(r.get('error')):
                f.write(f"| {r['pipeline']} | ERROR | ERROR | ERROR |\n")
            else:
                f.write(f"| {r['pipeline']} | {r.get('test_candidates', 0):,} | {r.get('test_reduction_ratio', 0):.4f} | {r.get('test_pairs_completeness', 0):.4f} |\n")
        
        f.write("\n## Dettagli Predizioni\n\n")
        f.write("| Pipeline | True Positives | False Positives | False Negatives | Total Predictions |\n")
        f.write("|----------|----------------|-----------------|-----------------|-------------------|\n")
        
        for r in results:
            if 'error' in r and pd.notna(r.get('error')):
                f.write(f"| {r['pipeline']} | ERROR | ERROR | ERROR | ERROR |\n")
            else:
                f.write(f"| {r['pipeline']} | {r.get('true_positives', 0):,} | {r.get('false_positives', 0):,} | {r.get('false_negatives', 0):,} | {r.get('predictions', 0):,} |\n")
        
        f.write("\n## Conclusioni\n\n")
        
        # Trova la pipeline migliore
        valid_results = [r for r in results if 'error' not in r or pd.isna(r.get('error'))]
        if valid_results:
            best_f1 = max(valid_results, key=lambda x: x.get('f1', 0))
            best_precision = max(valid_results, key=lambda x: x.get('precision', 0))
            best_recall = max(valid_results, key=lambda x: x.get('recall', 0))
            
            f.write(f"- **Miglior F1-measure**: {best_f1['pipeline']} ({best_f1.get('f1', 0):.4f})\n")
            f.write(f"- **Miglior Precision**: {best_precision['pipeline']} ({best_precision.get('precision', 0):.4f})\n")
            f.write(f"- **Miglior Recall**: {best_recall['pipeline']} ({best_recall.get('recall', 0):.4f})\n")
    
    print(f"  Report Markdown salvato in: {report_file}")


if __name__ == "__main__":
    main()

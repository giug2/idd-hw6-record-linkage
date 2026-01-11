"""
Valutazione delle Pipeline di Record Linkage

Questo script valuta le prestazioni di 6 pipeline (3 configurazioni x 2 blocking):

Configurazioni di Confronto:
- P1_textual_core: brand, model, body_type, description, price, mileage
- P2_plus_location: P1 + transmission, fuel_type, drive, city_region, state, year
- P3_minimal_fast: brand, model, year, price, mileage

Strategie di Blocking:
- B1: brand normalizzato + year
- B2: brand normalizzato + model prefix (2 caratteri)

Metriche calcolate:
- Precision
- Recall
- F1-measure
- Tempo di training
- Tempo di inferenza
- Statistiche di blocking (reduction ratio, coppie candidate)
"""

import os
import re
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import recordlinkage

warnings.filterwarnings('ignore')

# Path per importare i moduli di blocking
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'blocking'))
from blocking_B1 import normalize_brand


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
SPLITS_DIR = os.path.join(DATA_DIR, 'splits')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

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
    # Helper per estrarre colonne in modo sicuro
    def safe_get(col_name, default=None):
        return df[col_name].values if col_name in df.columns else default
    
    # Colonne per Craigslist (suffisso _craig)
    craig_df = pd.DataFrame({
        'source_id': safe_get('source_id_craig'),
        'vin': safe_get('vin'),
        'brand': safe_get('brand_craig'),
        'model': safe_get('model_craig'),
        'year': safe_get('year_craig'),
        'price': safe_get('price_craig'),
        'mileage': safe_get('mileage_craig'),
        'color': safe_get('color_craig'),
        'body_type': safe_get('body_type_craig'),
        'description': safe_get('description_craig'),
        'transmission': safe_get('transmission_craig'),
        'fuel_type': safe_get('fuel_type_craig'),
        'drive': safe_get('drive_craig'),
        'city_region': safe_get('city_region_craig'),
        'state': safe_get('state_craig'),
    })
    craig_df.index = pd.Index([f'craig_{i}' for i in range(len(craig_df))], name='id')
    
    # Colonne per US Used Cars (senza suffisso o colonne specifiche)
    us_df = pd.DataFrame({
        'source_id': safe_get('source_id_us', safe_get('source_id')),
        'vin': safe_get('vin_us', safe_get('vin')),
        'brand': safe_get('brand'),
        'model': safe_get('model'),
        'year': safe_get('year'),
        'price': safe_get('price'),
        'mileage': safe_get('mileage'),
        'color': safe_get('color'),
        'body_type': safe_get('body_type'),
        'description': safe_get('description'),
        'transmission': safe_get('transmission'),
        'fuel_type': safe_get('fuel_type'),
        'drive': safe_get('drive'),
        'city_region': safe_get('city_region'),
        'state': safe_get('state'),
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
# BLOCKING STRATEGIES (usando i moduli esterni)
# ============================================================================

def blocking_B1(craig_df, us_df):
    """Strategia B1: Blocking su (brand normalizzato, year)."""
    c_df = craig_df.copy()
    u_df = us_df.copy()
    
    # Aggiungi block_key ai dataframe
    c_df['block_key'] = c_df.apply(
        lambda r: f"{normalize_brand(r['brand'])}_{int(r['year']) if pd.notna(r['year']) else 'unknown'}", 
        axis=1
    )
    u_df['block_key'] = u_df.apply(
        lambda r: f"{normalize_brand(r['brand'])}_{int(r['year']) if pd.notna(r['year']) else 'unknown'}", 
        axis=1
    )
    
    # Usa recordlinkage Block per generare candidate pairs
    indexer = recordlinkage.Index()
    indexer.block('block_key')
    candidate_pairs = indexer.index(c_df, u_df)
    
    return candidate_pairs, c_df, u_df


def blocking_B2(craig_df, us_df):
    """Strategia B2: Blocking su Brand + Model Prefix (2 caratteri)."""
    c_df = craig_df.copy()
    u_df = us_df.copy()
    
    def create_block_key(brand, model):
        """Crea chiave: brand_norm + model[:2]"""
        if pd.isna(brand) or brand is None:
            return None
        brand_norm = str(brand).lower().strip()
        brand_norm = re.sub(r'[^a-z0-9]', '', brand_norm)
        
        if pd.isna(model) or model is None:
            return None
        model_norm = str(model).lower().strip()
        model_norm = re.sub(r'[^a-z0-9]', '', model_norm)
        
        if not brand_norm or not model_norm or len(model_norm) < 2:
            return None
        return f"{brand_norm}_{model_norm[:2]}"
    
    # Aggiungi block_key ai dataframe
    c_df['block_key_b2'] = c_df.apply(lambda r: create_block_key(r['brand'], r['model']), axis=1)
    u_df['block_key_b2'] = u_df.apply(lambda r: create_block_key(r['brand'], r['model']), axis=1)
    
    # Filtra record con chiave valida
    c_valid = c_df[c_df['block_key_b2'].notna()].copy()
    u_valid = u_df[u_df['block_key_b2'].notna()].copy()
    
    # Usa recordlinkage Block
    indexer = recordlinkage.Index()
    indexer.block('block_key_b2')
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
# COMPARISON RULES - 3 CONFIGURAZIONI
# ============================================================================

def create_comparison_P1_textual_core():
    """
    P1_textual_core: brand, model, body_type, description, price, mileage
    
    Configurazione focalizzata sui campi testuali core e attributi numerici principali.
    """
    compare = recordlinkage.Compare()
    
    # Brand: similarità stringa (Jaro-Winkler)
    compare.string('brand', 'brand', method='jarowinkler', threshold=0.85, label='brand_sim')
    
    # Model: similarità stringa
    compare.string('model', 'model', method='jarowinkler', threshold=0.75, label='model_sim')
    
    # Body type: similarità stringa
    compare.string('body_type', 'body_type', method='jarowinkler', threshold=0.8, label='body_type_sim')
    
    # Description: usando Jaro (più veloce di Levenshtein per testi lunghi)
    compare.string('description', 'description', method='jaro', threshold=0.6, label='description_sim')
    
    # Price: confronto numerico (scala gaussiana)
    compare.numeric('price', 'price', method='gauss', scale=5000, label='price_sim')
    
    # Mileage: confronto numerico
    compare.numeric('mileage', 'mileage', method='gauss', scale=10000, label='mileage_sim')
    
    return compare


def create_comparison_P2_plus_location():
    """
    P2_plus_location: P1 + transmission, fuel_type, drive, city_region, state, year
    
    Configurazione estesa con attributi di location e caratteristiche tecniche.
    """
    compare = recordlinkage.Compare()
    
    # === Campi da P1 ===
    # Brand: similarità stringa (Jaro-Winkler)
    compare.string('brand', 'brand', method='jarowinkler', threshold=0.85, label='brand_sim')
    
    # Model: similarità stringa
    compare.string('model', 'model', method='jarowinkler', threshold=0.75, label='model_sim')
    
    # Body type: similarità stringa
    compare.string('body_type', 'body_type', method='jarowinkler', threshold=0.8, label='body_type_sim')
    
    # Description: usando Jaro (più veloce)
    compare.string('description', 'description', method='jaro', threshold=0.6, label='description_sim')
    
    # Price: confronto numerico
    compare.numeric('price', 'price', method='gauss', scale=5000, label='price_sim')
    
    # Mileage: confronto numerico
    compare.numeric('mileage', 'mileage', method='gauss', scale=10000, label='mileage_sim')
    
    # === Campi aggiuntivi ===
    # Transmission: match esatto
    compare.exact('transmission', 'transmission', label='transmission_exact')
    
    # Fuel type: match esatto
    compare.exact('fuel_type', 'fuel_type', label='fuel_type_exact')
    
    # Drive: match esatto
    compare.exact('drive', 'drive', label='drive_exact')
    
    # City/Region: similarità stringa
    compare.string('city_region', 'city_region', method='jarowinkler', threshold=0.8, label='city_region_sim')
    
    # State: match esatto
    compare.exact('state', 'state', label='state_exact')
    
    # Year: match esatto
    compare.exact('year', 'year', label='year_exact')
    
    return compare


def create_comparison_P3_minimal_fast():
    """
    P3_minimal_fast: brand, model, year, price, mileage
    
    Configurazione minimale e veloce con campi essenziali.
    Include price e mileage per dare varianza sufficiente al classificatore.
    """
    compare = recordlinkage.Compare()
    
    # Brand: similarità stringa (Jaro-Winkler)
    compare.string('brand', 'brand', method='jarowinkler', threshold=0.85, label='brand_sim')
    
    # Model: similarità stringa
    compare.string('model', 'model', method='jarowinkler', threshold=0.75, label='model_sim')
    
    # Year: match esatto (più discriminativo)
    compare.exact('year', 'year', label='year_exact')
    
    # Price: confronto numerico (necessario per discriminare)
    compare.numeric('price', 'price', method='gauss', scale=5000, label='price_sim')
    
    # Mileage: confronto numerico (necessario per discriminare)
    compare.numeric('mileage', 'mileage', method='gauss', scale=10000, label='mileage_sim')
    
    return compare


# Dizionario delle configurazioni disponibili
COMPARISON_CONFIGS = {
    'P1_textual_core': create_comparison_P1_textual_core,
    'P2_plus_location': create_comparison_P2_plus_location,
    'P3_minimal_fast': create_comparison_P3_minimal_fast,
}


# ============================================================================
# PIPELINE: RecordLinkage
# ============================================================================

def run_recordlinkage_pipeline(train_df, test_df, blocking_strategy, comparison_config, pipeline_name):
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
        comparison_config: nome della configurazione di confronto (P1, P2, P3)
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
    
    # Crea regole di comparazione in base alla configurazione
    compare_func = COMPARISON_CONFIGS.get(comparison_config)
    if compare_func is None:
        raise ValueError(f"Configurazione di confronto '{comparison_config}' non trovata")
    compare = compare_func()
    print(f"  Configurazione: {comparison_config}")
    
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
# LOGGER
# ============================================================================

class Logger:
    """Classe per duplicare l'output su terminale e file."""
    
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Funzione principale per la valutazione delle pipeline."""
    log_file = os.path.join(OUTPUT_DIR, 'full_execution_log.txt')
    sys.stdout = Logger(log_file)
    
    print("\n" + "="*70)
    print("   VALUTAZIONE PIPELINE DI RECORD LINKAGE")
    print("   3 Configurazioni (P1, P2, P3) x 2 Blocking (B1, B2) = 6 Pipeline")
    print("="*70)
    print(f"   Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Carica dati
    train_df, val_df, test_df = load_data()
    
    # Combina train + validation per training più robusto
    train_full = pd.concat([train_df, val_df], ignore_index=True)
    print(f"\n  Training set combinato (train+val): {len(train_full):,} record")
    
    all_results = []
    
    # Definizione delle 6 pipeline (3 configurazioni x 2 strategie di blocking)
    pipelines = [
        # Con Blocking B1 (brand + year)
        ('P1_textual_core_B1', 'P1_textual_core', blocking_B1),
        ('P2_plus_location_B1', 'P2_plus_location', blocking_B1),
        ('P3_minimal_fast_B1', 'P3_minimal_fast', blocking_B1),
        # Con Blocking B2 (brand + model prefix)
        ('P1_textual_core_B2', 'P1_textual_core', blocking_B2),
        ('P2_plus_location_B2', 'P2_plus_location', blocking_B2),
        ('P3_minimal_fast_B2', 'P3_minimal_fast', blocking_B2),
    ]
    
    for pipeline_name, comparison_config, blocking_strategy in pipelines:
        try:
            results = run_recordlinkage_pipeline(
                train_full, test_df, 
                blocking_strategy,
                comparison_config,
                pipeline_name
            )
            all_results.append(results)
        except Exception as e:
            print(f"\n  ERRORE in {pipeline_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'pipeline': pipeline_name,
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
    
    print(f"  Log completo salvato in: {log_file}")
    
    return results_df


def generate_markdown_report(results):
    """Genera un report in formato Markdown."""
    
    report_file = os.path.join(OUTPUT_DIR, 'EVALUATION_RESULTS.md')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Risultati Valutazione Pipeline di Record Linkage\n\n")
        f.write(f"**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Sommario\n\n")
        f.write("Questo documento riporta i risultati della valutazione delle pipeline di Record Linkage:\n\n")
        f.write("### Configurazioni di Confronto:\n")
        f.write("1. **P1_textual_core**: brand, model, body_type, description, price, mileage\n")
        f.write("2. **P2_plus_location**: P1 + transmission, fuel_type, drive, city_region, state, year\n")
        f.write("3. **P3_minimal_fast**: brand, model, year, price, mileage\n\n")
        f.write("### Strategie di Blocking:\n")
        f.write("- **B1**: brand normalizzato + year\n")
        f.write("- **B2**: brand normalizzato + model prefix (2 caratteri)\n\n")
        f.write("Ogni configurazione è testata con entrambe le strategie di blocking (6 pipeline totali).\n\n")
        
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

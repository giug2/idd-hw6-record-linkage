"""
Valutazione delle Pipeline di Record Linkage SENZA BLOCKING

Questo script valuta le prestazioni delle 3 pipeline SENZA blocking (confronto completo NxM):

Configurazioni di Confronto:
- P1_textual_core
- P2_plus_location
- P3_minimal_fast

Strategia di Blocking:
- NONE (Full Index): Confronto cartesiano completo

Metriche calcolate:
- Precision
- Recall
- F1-measure
- Tempo di training
- Tempo di inferenza
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

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
SPLITS_DIR = os.path.join(DATA_DIR, 'splits')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'record_linkage_no_blocking')

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
    """
    def safe_get(col_name, default=None):
        return df[col_name].values if col_name in df.columns else default
    
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
    
    true_pairs = [(f'craig_{i}', f'us_{i}') for i in range(len(df))]
    true_links = pd.MultiIndex.from_tuples(true_pairs, names=['id_1', 'id_2'])
    
    return craig_df, us_df, true_links


def calculate_metrics(predicted_pairs, true_links):
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
# BLOCKING STRATEGY: FULL (NO BLOCKING)
# ============================================================================
def blocking_FULL(craig_df, us_df):
    """Strategia FULL: Nessun blocking, confronto completo NxM."""
    print("    Generazione Indice Completo...")
    indexer = recordlinkage.Index()
    indexer.full()
    candidate_pairs = indexer.index(craig_df, us_df)
    return candidate_pairs, craig_df, us_df


def analyze_blocking(candidate_pairs, craig_df, us_df, true_links, name):
    n_candidates = len(candidate_pairs)
    n_craig = len(craig_df)
    n_us = len(us_df)
    n_total_pairs = n_craig * n_us
    
    reduction_ratio = 1 - (n_candidates / n_total_pairs) if n_total_pairs > 0 else 0
    
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
        'reduction_ratio': reduction_ratio,
        'pairs_completeness': pairs_completeness
    }


# ============================================================================
# COMPARISON RULES
# ============================================================================
def create_comparison_P1_textual_core():
    compare = recordlinkage.Compare()
    compare.string('brand', 'brand', method='jarowinkler', threshold=0.85, label='brand_sim')
    compare.string('model', 'model', method='jarowinkler', threshold=0.75, label='model_sim')
    compare.string('body_type', 'body_type', method='jarowinkler', threshold=0.8, label='body_type_sim')
    compare.string('description', 'description', method='jaro', threshold=0.6, label='description_sim')
    compare.numeric('price', 'price', method='gauss', scale=5000, label='price_sim')
    compare.numeric('mileage', 'mileage', method='gauss', scale=10000, label='mileage_sim')
    return compare

def create_comparison_P2_plus_location():
    compare = recordlinkage.Compare()
    # P1 part
    compare.string('brand', 'brand', method='jarowinkler', threshold=0.85, label='brand_sim')
    compare.string('model', 'model', method='jarowinkler', threshold=0.75, label='model_sim')
    compare.string('body_type', 'body_type', method='jarowinkler', threshold=0.8, label='body_type_sim')
    compare.string('description', 'description', method='jaro', threshold=0.6, label='description_sim')
    compare.numeric('price', 'price', method='gauss', scale=5000, label='price_sim')
    compare.numeric('mileage', 'mileage', method='gauss', scale=10000, label='mileage_sim')
    # P2 extras
    compare.exact('transmission', 'transmission', label='transmission_exact')
    compare.exact('fuel_type', 'fuel_type', label='fuel_type_exact')
    compare.exact('drive', 'drive', label='drive_exact')
    compare.string('city_region', 'city_region', method='jarowinkler', threshold=0.8, label='city_region_sim')
    compare.exact('state', 'state', label='state_exact')
    compare.exact('year', 'year', label='year_exact')
    return compare

def create_comparison_P3_minimal_fast():
    compare = recordlinkage.Compare()
    compare.string('brand', 'brand', method='jarowinkler', threshold=0.85, label='brand_sim')
    compare.string('model', 'model', method='jarowinkler', threshold=0.75, label='model_sim')
    compare.exact('year', 'year', label='year_exact')
    compare.numeric('price', 'price', method='gauss', scale=5000, label='price_sim')
    compare.numeric('mileage', 'mileage', method='gauss', scale=10000, label='mileage_sim')
    return compare

COMPARISON_CONFIGS = {
    'P1_textual_core': create_comparison_P1_textual_core,
    'P2_plus_location': create_comparison_P2_plus_location,
    'P3_minimal_fast': create_comparison_P3_minimal_fast,
}


def compute_features_in_chunks(compare, pairs, df_a, df_b, chunk_size=50000):
    """Calcola le features in batch per mostrare il progresso."""
    n_pairs = len(pairs)
    if n_pairs == 0:
        return compare.compute(pairs, df_a, df_b)
        
    n_chunks = (n_pairs // chunk_size) + (1 if n_pairs % chunk_size > 0 else 0)
    
    print(f"    Elaborazione in chunk...")
    
    results = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n_pairs)
        
        # Estrai chunk di coppie
        chunk_pairs = pairs[start:end]
        
        # Calcola features
        chunk_features = compare.compute(chunk_pairs, df_a, df_b)
        results.append(chunk_features)
    
    print("    Calcolo features completato!")
    return pd.concat(results)


# ============================================================================
# PIPELINE
# ============================================================================
def run_recordlinkage_pipeline(train_df, test_df, blocking_strategy, comparison_config, pipeline_name):
    print(f"\n{'='*70}")
    print(f"PIPELINE: {pipeline_name}")
    print(f"{'='*70}")
    
    results = {
        'pipeline': pipeline_name,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    print("\n[1/5] Preparazione dati...")
    c_train, u_train, true_train = prepare_dataframes_for_linkage(train_df)
    c_test, u_test, true_test = prepare_dataframes_for_linkage(test_df)
    
    results['train_size'] = len(train_df)
    results['test_size'] = len(test_df)
    
    print("\n[2/5] Blocking (Training) - FULL INDEX...")
    start_blocking_train = time.time()
    train_pairs, c_train_blocked, u_train_blocked = blocking_strategy(c_train, u_train)
    blocking_train_time = time.time() - start_blocking_train
    
    blocking_stats_train = analyze_blocking(train_pairs, c_train, u_train, true_train, f"{pipeline_name} - Training")
    results['blocking_train_time'] = blocking_train_time
    results['train_candidates'] = blocking_stats_train['n_candidates']
    
    print("\n[3/5] Training classificatore...")
    start_train = time.time()
    compare_func = COMPARISON_CONFIGS.get(comparison_config)
    compare = compare_func()
    print(f"  Configurazione: {comparison_config}")

    # --- OTTIMIZZAZIONE TRAINING ---
    # Invece di calcolare features per 3M di coppie e poi buttare via i negativi,
    # campioniamo i negativi DIRETTAMENTE sull'indice. Molto più veloce.
    
    print("  Ottimizzazione coppie di training...")
    # Identifica i positivi e i negativi sugli indici
    true_matches_index = true_train.intersection(train_pairs)
    potential_negatives_index = train_pairs.difference(true_train)
    
    n_pos = len(true_matches_index)
    n_neg_total = len(potential_negatives_index)
    
    # Campioniamo i negativi (ratio 1:10 o min 5000)
    # Questo evita di calcolare features pesanti (string similarity) per milioni di coppie inutili
    n_neg_to_keep = min(n_neg_total, max(n_pos * 10, 5000))
    
    print(f"    Coppie totali disponibili: {len(train_pairs):,} (Pos: {n_pos}, Neg: {n_neg_total:,})")

    if n_neg_total > n_neg_to_keep:
        # Usa numpy per scegliere indici casuali velocemente
        neg_indices_sampled = np.random.choice(potential_negatives_index, n_neg_to_keep, replace=False)
        # Ricostruisci MultiIndex
        negatives_sampled = pd.MultiIndex.from_tuples(neg_indices_sampled, names=['id_1', 'id_2'])
    else:
        negatives_sampled = potential_negatives_index
        
    # Unisci positivi e negativi campionati
    train_pairs_optimized = true_matches_index.union(negatives_sampled)

    print("  Calcolo features di training (set ottimizzato)...")
    # Usiamo il set ridotto
    features_train = compute_features_in_chunks(compare, train_pairs_optimized, c_train_blocked, u_train_blocked, chunk_size=len(train_pairs_optimized))
    
    # Ricalcola match index sul set ridotto (dovrebbe essere identico a true_matches_index)
    match_index_train_opt = true_train.intersection(train_pairs_optimized)
    
    print("  Addestramento Logistic Regression...")
    classifier = recordlinkage.LogisticRegressionClassifier()
    classifier.fit(features_train, match_index_train_opt)
    training_time = time.time() - start_train
    results['training_time'] = training_time
    
    print("\n[4/5] Blocking (Test) - FULL INDEX...")
    start_blocking_test = time.time()
    test_pairs, c_test_blocked, u_test_blocked = blocking_strategy(c_test, u_test)
    blocking_test_time = time.time() - start_blocking_test
    
    blocking_stats_test = analyze_blocking(test_pairs, c_test, u_test, true_test, f"{pipeline_name} - Test")
    results['blocking_test_time'] = blocking_test_time
    results['test_candidates'] = blocking_stats_test['n_candidates']
    results['test_reduction_ratio'] = blocking_stats_test['reduction_ratio']
    results['test_pairs_completeness'] = blocking_stats_test['pairs_completeness']
    
    print("\n[5/5] Inference e valutazione...")
    start_inference = time.time()
    print("  Calcolo features di test...")
    # features_test = compare.compute(test_pairs, c_test_blocked, u_test_blocked)
    features_test = compute_features_in_chunks(compare, test_pairs, c_test_blocked, u_test_blocked)
    
    print("  Predizione...")
    proba = classifier.prob(features_test)
    
    threshold = 0.5
    predictions = features_test.index[proba >= threshold]
    print(f"    Predizioni con soglia {threshold}: {len(predictions):,}")
    
    inference_time = time.time() - start_inference
    
    print("\n  Calcolo metriche...")
    precision, recall, f1, tp, fp, fn = calculate_metrics(predictions, true_test)
    
    results['precision'] = precision
    results['recall'] = recall
    results['f1'] = f1
    results['true_positives'] = tp
    results['false_positives'] = fp
    results['false_negatives'] = fn
    results['inference_time'] = inference_time
    results['predictions'] = len(predictions)
    
    # ========== RIEPILOGO DETTAGLIATO ==========
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
    print(f"  Coeff. Classificatore (Feature Importance):")
    try:
        # Tenta di stampare i coefficienti se disponibili
        params = classifier.algorithm.coef_[0]
        feature_names = features_train.columns
        for name, coef in zip(feature_names, params):
            print(f"    {name:<20}: {coef:.4f}")
        print(f"    {'intercept':<20}: {classifier.algorithm.intercept_[0]:.4f}")
    except:
        print("    (Coefficienti non disponibili)")
    
    print(f"")
    print(f"  Tempi di Esecuzione:")
    print(f"    Training time:    {training_time:.2f}s")
    print(f"    Inference time:   {inference_time:.2f}s")
    print(f"    Totale:           {training_time + inference_time + blocking_train_time + blocking_test_time:.2f}s")

    return results


class Logger:
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


def main():
    log_file = os.path.join(OUTPUT_DIR, 'full_execution_log_no_blocking.txt')
    sys.stdout = Logger(log_file)
    
    print("\n" + "="*70)
    print("   VALUTAZIONE PIPELINE SENZA BLOCKING")
    print("="*70)
    
    train_df, val_df, test_df = load_data()
    train_full = pd.concat([train_df, val_df], ignore_index=True)
    
    all_results = []
    
    pipelines = [
        ('P1_textual_core_NO_BLOCKING', 'P1_textual_core', blocking_FULL),
        ('P2_plus_location_NO_BLOCKING', 'P2_plus_location', blocking_FULL),
        ('P3_minimal_fast_NO_BLOCKING', 'P3_minimal_fast', blocking_FULL),
    ]
    
    for pipeline_name, comparison_config, blocking_strategy in pipelines:
        try:
            results = run_recordlinkage_pipeline(train_full, test_df, blocking_strategy, comparison_config, pipeline_name)
            all_results.append(results)
        except Exception as e:
            print(f"\n  ERRORE in {pipeline_name}: {e}")
            import traceback
            traceback.print_exc()
    
    results_df = pd.DataFrame(all_results)
    
    print("\n" + "="*80)
    print("   RIEPILOGO FINALE")
    print("="*80)
    
    cols = ['pipeline', 'precision', 'recall', 'f1', 'training_time', 'inference_time']
    if len(results_df) > 0:
        print(results_df[[c for c in cols if c in results_df.columns]].to_string(index=False))
        
        output_file = os.path.join(OUTPUT_DIR, 'pipeline_evaluation_results.csv')
        results_df.to_csv(output_file, index=False)
        print(f"\n  Risultati salvati in: {output_file}")
        
        # Markdown Report
        report_file = os.path.join(OUTPUT_DIR, 'EVALUATION_RESULTS.md')
        
        # Semplice formatter markdown manuale per evitare dipendenza tabulate
        def simple_to_markdown(df):
            if df.empty: return ""
            headers = df.columns.tolist()
            # Header
            md = "| " + " | ".join(headers) + " |\n"
            # Separator
            md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
            # Data
            for _, row in df.iterrows():
                # Formattazione valori: 4 decimali per float, stringa per altri
                vals = []
                for x in row:
                    if isinstance(x, (float, np.float64)):
                        vals.append(f"{x:.4f}")
                    else:
                        vals.append(str(x))
                md += "| " + " | ".join(vals) + " |\n"
            return md
            
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Risultati Pipeline SENZA Blocking\n\n")
            f.write(simple_to_markdown(results_df[[c for c in cols if c in results_df.columns]]))
            
    print(f"\nLog completo su: {log_file}")

if __name__ == "__main__":
    main()

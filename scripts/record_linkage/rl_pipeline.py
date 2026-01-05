"""
Record Linkage con la libreria Python RecordLinkage

Questo script implementa il record linkage usando la libreria recordlinkage.
Supporta entrambe le strategie di blocking (B1 e B2) e calcola le metriche
di valutazione: precision, recall, F1-measure.
"""

import pandas as pd
import numpy as np
import recordlinkage
from recordlinkage.index import Block
from recordlinkage.compare import Exact, String, Numeric
import time
import os
import warnings
warnings.filterwarnings('ignore')


def load_data(train_path, val_path=None, test_path=None):
    """
    Carica i dataset di training, validation e test.
    
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    train_df = pd.read_csv(train_path, low_memory=False)
    val_df = pd.read_csv(val_path, low_memory=False) if val_path else None
    test_df = pd.read_csv(test_path, low_memory=False) if test_path else None
    
    return train_df, val_df, test_df


def prepare_dataframes_for_linkage(df):
    """
    Prepara due DataFrame separati per le due sorgenti dal dataset combinato.
    
    Il dataset originale ha record già allineati (ogni riga contiene dati
    da entrambe le sorgenti). Creiamo due DataFrame separati per simulare
    il task di record linkage.
    """
    # Colonne per Craigslist
    craig_cols = {
        'source_id_craig': 'source_id',
        'vin': 'vin',
        'brand_craig': 'brand',
        'model_craig': 'model',
        'year_craig': 'year',
        'price_craig': 'price',
        'mileage_craig': 'mileage',
        'color_craig': 'color',
        'body_type_craig': 'body_type',
        'transmission_craig': 'transmission',
        'fuel_type_craig': 'fuel_type'
    }
    
    # Colonne per US Used Cars
    us_cols = {
        'source_id_us': 'source_id',
        'vin_us': 'vin',
        'brand': 'brand',
        'model': 'model',
        'year': 'year',
        'price': 'price',
        'mileage': 'mileage',
        'color': 'color',
        'body_type': 'body_type',
        'transmission': 'transmission',
        'fuel_type': 'fuel_type'
    }
    
    # Crea DataFrame per Craigslist
    craig_df = df[[c for c in craig_cols.keys() if c in df.columns]].copy()
    craig_df = craig_df.rename(columns={k: v for k, v in craig_cols.items() if k in df.columns})
    craig_df.index = pd.Index([f'craig_{i}' for i in range(len(craig_df))], name='id')
    
    # Crea DataFrame per US Used Cars
    us_df = df[[c for c in us_cols.keys() if c in df.columns]].copy()
    us_df = us_df.rename(columns={k: v for k, v in us_cols.items() if k in df.columns})
    us_df.index = pd.Index([f'us_{i}' for i in range(len(us_df))], name='id')
    
    # Ground truth: le coppie vere sono quelle con lo stesso indice originale
    # (dato che ogni riga del dataset originale rappresenta un match)
    true_pairs = [(f'craig_{i}', f'us_{i}') for i in range(len(df))]
    true_links = pd.MultiIndex.from_tuples(true_pairs, names=['id_1', 'id_2'])
    
    return craig_df, us_df, true_links


def normalize_brand(brand):
    """Normalizza il nome del brand."""
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


def create_blocking_B1(craig_df, us_df):
    """
    Strategia B1: Blocking su (brand, year).
    
    Usa recordlinkage.Index con Block su brand e year.
    """
    # Normalizza brand
    craig_df = craig_df.copy()
    us_df = us_df.copy()
    craig_df['brand_norm'] = craig_df['brand'].apply(normalize_brand)
    us_df['brand_norm'] = us_df['brand'].apply(normalize_brand)
    
    # Converti year in string per blocking (gestisce NaN)
    craig_df['year_str'] = craig_df['year'].apply(
        lambda x: str(int(x)) if pd.notna(x) else 'unknown'
    )
    us_df['year_str'] = us_df['year'].apply(
        lambda x: str(int(x)) if pd.notna(x) else 'unknown'
    )
    
    # Crea blocking key combinata
    craig_df['block_key'] = craig_df['brand_norm'] + '_' + craig_df['year_str']
    us_df['block_key'] = us_df['brand_norm'] + '_' + us_df['year_str']
    
    # Debug: stampa alcune chiavi
    print(f"  Esempio chiavi Craig: {craig_df['block_key'].head(3).tolist()}")
    print(f"  Esempio chiavi US: {us_df['block_key'].head(3).tolist()}")
    print(f"  Chiavi uniche Craig: {craig_df['block_key'].nunique()}")
    print(f"  Chiavi uniche US: {us_df['block_key'].nunique()}")
    
    # Usa recordlinkage Block
    indexer = recordlinkage.Index()
    indexer.block('block_key')
    
    candidate_pairs = indexer.index(craig_df, us_df)
    
    return candidate_pairs, craig_df, us_df


def create_blocking_B2(craig_df, us_df):
    """
    Strategia B2: Blocking su Brand + Model Prefix (2 caratteri).
    
    Più specifico di B1 ma tollerante a variazioni nel nome modello.
    """
    craig_df = craig_df.copy()
    us_df = us_df.copy()
    
    import re
    
    def normalize_string(s):
        """Normalizza stringa: lowercase, solo alfanumerici."""
        if pd.isna(s) or s is None:
            return None
        s = str(s).lower().strip()
        s = re.sub(r'[^a-z0-9]', '', s)
        return s if len(s) > 0 else None
    
    def get_model_prefix(model, length=2):
        """Estrae i primi 2 caratteri del modello normalizzato."""
        normalized = normalize_string(model)
        if normalized is None:
            return None
        return normalized[:length] if len(normalized) >= length else normalized
    
    def create_block_key(brand, model):
        """Crea chiave: brand_norm + model[:2]"""
        brand_norm = normalize_brand(brand)  # Usa la funzione esistente
        model_prefix = get_model_prefix(model)
        if brand_norm is None or model_prefix is None:
            return None
        return f"{brand_norm}_{model_prefix}"
    
    # Crea blocking key
    craig_df['block_key_b2'] = craig_df.apply(
        lambda r: create_block_key(r['brand'], r['model']), axis=1
    )
    us_df['block_key_b2'] = us_df.apply(
        lambda r: create_block_key(r['brand'], r['model']), axis=1
    )
    
    # Filtra record senza chiave valida
    craig_valid = craig_df[craig_df['block_key_b2'].notna()].copy()
    us_valid = us_df[us_df['block_key_b2'].notna()].copy()
    
    indexer = recordlinkage.Index()
    indexer.block('block_key_b2')
    
    candidate_pairs = indexer.index(craig_valid, us_valid)
    
    return candidate_pairs, craig_df, us_df


def define_comparison_rules():
    """
    Definisce le regole di comparazione per il record linkage.
    
    Regole:
    - VIN: exact match (se disponibile)
    - Brand: string similarity (Jaro-Winkler)
    - Model: string similarity (Jaro-Winkler)
    - Year: exact match
    - Price: numeric comparison con threshold
    - Mileage: numeric comparison con threshold
    - Color: string similarity
    """
    compare = recordlinkage.Compare()
    
    # VIN: match esatto (molto importante se disponibile)
    compare.exact('vin', 'vin', label='vin_exact')
    
    # Brand: similarità stringa
    compare.string('brand', 'brand', method='jarowinkler', threshold=0.85, label='brand_sim')
    
    # Model: similarità stringa (più permissivo perché può variare)
    compare.string('model', 'model', method='jarowinkler', threshold=0.75, label='model_sim')
    
    # Year: match esatto
    compare.exact('year', 'year', label='year_exact')
    
    # Price: confronto numerico (con tolleranza del 20%)
    compare.numeric('price', 'price', method='gauss', scale=5000, label='price_sim')
    
    # Mileage: confronto numerico (con tolleranza)
    compare.numeric('mileage', 'mileage', method='gauss', scale=10000, label='mileage_sim')
    
    # Color: match esatto
    compare.exact('color', 'color', label='color_exact')
    
    return compare


def train_classifier(features, true_links, method='ecm'):
    """
    Addestra un classificatore per il record linkage.
    
    Args:
        features: DataFrame con le feature di comparazione
        true_links: MultiIndex con le coppie vere
        method: 'ecm' (unsupervised) o 'logreg' (supervised)
    
    Returns:
        Classificatore addestrato e tempo di training
    """
    start_time = time.time()
    
    # Filtra true_links per avere solo le coppie presenti in features
    match_index = true_links.intersection(features.index)
    
    if method == 'ecm':
        # ECM (Expectation-Conditional Maximization) - unsupervised
        classifier = recordlinkage.ECMClassifier(binarize=0.5)
        classifier.fit(features)
    
    elif method == 'logreg':
        # Logistic Regression - supervised
        # recordlinkage vuole match_index (MultiIndex), non array binario
        classifier = recordlinkage.LogisticRegressionClassifier()
        classifier.fit(features, match_index)
    
    elif method == 'nb':
        # Naive Bayes - supervised
        classifier = recordlinkage.NaiveBayesClassifier(binarize=0.5)
        classifier.fit(features, match_index)
    
    elif method == 'svm':
        # SVM - supervised
        classifier = recordlinkage.SVMClassifier()
        classifier.fit(features, match_index)
    
    else:
        raise ValueError(f"Metodo non supportato: {method}")
    
    training_time = time.time() - start_time
    
    return classifier, training_time


def classify_and_evaluate(classifier, features, true_links):
    """
    Classifica le coppie e calcola le metriche.
    
    Returns:
        dict: dizionario con precision, recall, F1, tempo di inferenza
    """
    start_time = time.time()
    
    # Predizione
    predicted_links = classifier.predict(features)
    
    inference_time = time.time() - start_time
    
    # Calcolo metriche
    # True Positives: coppie predette che sono vere
    tp = len(predicted_links.intersection(true_links))
    
    # False Positives: coppie predette che non sono vere
    fp = len(predicted_links.difference(true_links))
    
    # False Negatives: coppie vere non predette
    fn = len(true_links.difference(predicted_links))
    
    # Metriche
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'predicted_links': len(predicted_links),
        'true_links': len(true_links),
        'inference_time': inference_time
    }


def run_pipeline(df, blocking_strategy='B1', classifier_method='ecm', verbose=True):
    """
    Esegue l'intera pipeline di record linkage.
    
    Args:
        df: DataFrame con i dati
        blocking_strategy: 'B1' (brand+year) o 'B2' (VIN prefix)
        classifier_method: 'ecm', 'logreg', 'nb', 'svm'
        verbose: stampa output dettagliato
    
    Returns:
        dict: risultati della pipeline
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"PIPELINE: {blocking_strategy}-RecordLinkage ({classifier_method})")
        print(f"{'='*60}")
    
    # Prepara i dati
    craig_df, us_df, true_links = prepare_dataframes_for_linkage(df)
    
    if verbose:
        print(f"\nRecord Craigslist: {len(craig_df)}")
        print(f"Record US Used Cars: {len(us_df)}")
        print(f"Match veri (ground truth): {len(true_links)}")
    
    # Blocking
    if verbose:
        print(f"\nApplicazione blocking {blocking_strategy}...")
    
    start_blocking = time.time()
    if blocking_strategy == 'B1':
        candidate_pairs, craig_df, us_df = create_blocking_B1(craig_df, us_df)
    else:  # B2
        candidate_pairs, craig_df, us_df = create_blocking_B2(craig_df, us_df)
    blocking_time = time.time() - start_blocking
    
    if verbose:
        print(f"Coppie candidate: {len(candidate_pairs):,}")
        print(f"Tempo blocking: {blocking_time:.2f}s")
    
    if len(candidate_pairs) == 0:
        print("ATTENZIONE: Nessuna coppia candidata generata!")
        return None
    
    # Comparazione
    if verbose:
        print("\nComparazione dei record...")
    
    start_compare = time.time()
    compare = define_comparison_rules()
    features = compare.compute(candidate_pairs, craig_df, us_df)
    compare_time = time.time() - start_compare
    
    if verbose:
        print(f"Feature calcolate: {features.shape}")
        print(f"Tempo comparazione: {compare_time:.2f}s")
    
    # Filtra le coppie vere che sono nelle candidate pairs
    true_links_in_candidates = true_links.intersection(candidate_pairs)
    
    if verbose:
        print(f"\nMatch veri nelle coppie candidate: {len(true_links_in_candidates)}")
        if len(true_links) > 0:
            coverage = len(true_links_in_candidates) / len(true_links) * 100
            print(f"Copertura blocking: {coverage:.2f}%")
    
    # Training
    if verbose:
        print(f"\nTraining classificatore ({classifier_method})...")
    
    classifier, training_time = train_classifier(features, true_links_in_candidates, method=classifier_method)
    
    if verbose:
        print(f"Tempo training: {training_time:.2f}s")
    
    # Valutazione
    if verbose:
        print("\nClassificazione e valutazione...")
    
    results = classify_and_evaluate(classifier, features, true_links_in_candidates)
    results['blocking_time'] = blocking_time
    results['compare_time'] = compare_time
    results['training_time'] = training_time
    results['total_time'] = blocking_time + compare_time + training_time + results['inference_time']
    results['candidate_pairs'] = len(candidate_pairs)
    results['blocking_coverage'] = len(true_links_in_candidates) / len(true_links) if len(true_links) > 0 else 0
    
    if verbose:
        print(f"\n{'='*60}")
        print("RISULTATI")
        print(f"{'='*60}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall:    {results['recall']:.4f}")
        print(f"F1-score:  {results['f1']:.4f}")
        print(f"\nTrue Positives:  {results['tp']}")
        print(f"False Positives: {results['fp']}")
        print(f"False Negatives: {results['fn']}")
        print(f"\nTempo totale:    {results['total_time']:.2f}s")
        print(f"  - Blocking:    {results['blocking_time']:.2f}s")
        print(f"  - Comparazione:{results['compare_time']:.2f}s")
        print(f"  - Training:    {results['training_time']:.2f}s")
        print(f"  - Inferenza:   {results['inference_time']:.2f}s")
    
    return results


def main():
    """Funzione principale."""
    
    # Percorsi - risale di 2 livelli (scripts/record_linkage/ → root)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_path = os.path.join(base_dir, "dataset", "splits", "train.csv")
    val_path = os.path.join(base_dir, "dataset", "splits", "validation.csv")
    test_path = os.path.join(base_dir, "dataset", "splits", "test.csv")
    
    print("="*70)
    print("RECORD LINKAGE CON LIBRERIA RECORDLINKAGE")
    print("="*70)
    
    # Carica i dati
    print("\nCaricamento dataset...")
    train_df, val_df, test_df = load_data(train_path, val_path, test_path)
    print(f"Training set: {len(train_df)} record")
    if val_df is not None:
        print(f"Validation set: {len(val_df)} record")
    if test_df is not None:
        print(f"Test set: {len(test_df)} record")
    
    # Risultati per tutte le combinazioni
    all_results = []
    
    # Pipeline B1-RecordLinkage (ECM - unsupervised)
    results_b1_ecm = run_pipeline(train_df, blocking_strategy='B1', classifier_method='ecm')
    if results_b1_ecm:
        results_b1_ecm['pipeline'] = 'B1-RecordLinkage-ECM'
        all_results.append(results_b1_ecm)
    
    # Pipeline B2-RecordLinkage (ECM - unsupervised)
    results_b2_ecm = run_pipeline(train_df, blocking_strategy='B2', classifier_method='ecm')
    if results_b2_ecm:
        results_b2_ecm['pipeline'] = 'B2-RecordLinkage-ECM'
        all_results.append(results_b2_ecm)
    
    # Pipeline B1-RecordLinkage (LogReg - supervised)
    results_b1_lr = run_pipeline(train_df, blocking_strategy='B1', classifier_method='logreg')
    if results_b1_lr:
        results_b1_lr['pipeline'] = 'B1-RecordLinkage-LogReg'
        all_results.append(results_b1_lr)
    
    # Pipeline B2-RecordLinkage (LogReg - supervised)
    results_b2_lr = run_pipeline(train_df, blocking_strategy='B2', classifier_method='logreg')
    if results_b2_lr:
        results_b2_lr['pipeline'] = 'B2-RecordLinkage-LogReg'
        all_results.append(results_b2_lr)
    
    # Riepilogo finale
    print("\n" + "="*70)
    print("RIEPILOGO COMPARATIVO")
    print("="*70)
    
    if all_results:
        print(f"\n{'Pipeline':<30} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Time (s)':>10}")
        print("-" * 70)
        for r in all_results:
            print(f"{r['pipeline']:<30} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1']:>10.4f} {r['total_time']:>10.2f}")
    
    return all_results


if __name__ == "__main__":
    results = main()

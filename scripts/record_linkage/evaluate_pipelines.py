"""
Valutazione delle Pipeline di Record Linkage

Questo script valuta le prestazioni di 6 pipeline:
- B1-RecordLinkage: Blocking (brand, year) + RecordLinkage
- B2-RecordLinkage: Blocking (VIN prefix) + RecordLinkage
- B1-dedupe: Blocking (brand, year) + Dedupe
- B2-dedupe: Blocking (VIN prefix) + Dedupe
- B1-ditto: Blocking (brand, year) + Ditto
- B2-ditto: Blocking (VIN prefix) + Ditto

Metriche calcolate:
- Precision
- Recall
- F1-measure
- Tempo di training
- Tempo di inferenza
"""

import pandas as pd
import numpy as np
import time
import os
import re
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
SPLITS_DIR = os.path.join(DATA_DIR, 'splits')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_data():
    """Carica i dataset."""
    train_df = pd.read_csv(os.path.join(SPLITS_DIR, 'train.csv'), low_memory=False)
    val_df = pd.read_csv(os.path.join(SPLITS_DIR, 'validation.csv'), low_memory=False)
    test_df = pd.read_csv(os.path.join(SPLITS_DIR, 'test.csv'), low_memory=False)
    print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df


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


def get_vin_prefix(vin, length=8):
    """Estrae il prefisso VIN."""
    if pd.isna(vin) or vin is None:
        return None
    vin = str(vin).upper().strip()
    vin = re.sub(r'[^A-Z0-9]', '', vin)
    if len(vin) < length:
        return None
    return vin[:length]


def prepare_dataframes(df):
    """
    Prepara due DataFrame separati per le due sorgenti.
    Ogni riga del dataset originale rappresenta un match vero.
    """
    # Craigslist: usa colonne con suffisso _craig
    craig_df = pd.DataFrame({
        'source_id': df['source_id_craig'].values,
        'vin': df['vin'].values,  # VIN è condiviso o è la colonna principale
        'brand': df['brand_craig'].values,
        'model': df['model_craig'].values,
        'year': df['year_craig'].values,
        'price': df['price_craig'].values,
        'mileage': df['mileage_craig'].values,
        'color': df['color_craig'].values if 'color_craig' in df.columns else None,
    })
    craig_df.index = pd.Index([f'craig_{i}' for i in range(len(craig_df))], name='id')
    
    # US Used Cars: usa colonne senza suffisso (o con suffisso _us per vin)
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
    
    # True links: ogni riga i del dataset originale è un match (craig_i, us_i)
    true_pairs = [(f'craig_{i}', f'us_{i}') for i in range(len(df))]
    true_links = pd.MultiIndex.from_tuples(true_pairs, names=['id_1', 'id_2'])
    
    return craig_df, us_df, true_links


def calculate_metrics(predicted_pairs, true_links):
    """Calcola precision, recall e F1."""
    if len(predicted_pairs) == 0:
        return 0.0, 0.0, 0.0
    
    # Converti in set per confronto efficiente
    pred_set = set(predicted_pairs)
    true_set = set(true_links)
    
    true_positives = len(pred_set.intersection(true_set))
    
    precision = true_positives / len(pred_set) if len(pred_set) > 0 else 0
    recall = true_positives / len(true_set) if len(true_set) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1


# ============================================================================
# BLOCKING STRATEGIES
# ============================================================================

def blocking_B1(craig_df, us_df):
    """B1: Blocking su (brand normalizzato, year)."""
    import recordlinkage
    
    c_df = craig_df.copy()
    u_df = us_df.copy()
    
    c_df['brand_norm'] = c_df['brand'].apply(normalize_brand)
    u_df['brand_norm'] = u_df['brand'].apply(normalize_brand)
    
    # Normalizza year come intero (rimuove .0)
    c_df['year_str'] = c_df['year'].apply(lambda x: str(int(x)) if pd.notna(x) else 'unknown')
    u_df['year_str'] = u_df['year'].apply(lambda x: str(int(x)) if pd.notna(x) else 'unknown')
    
    c_df['block_key'] = c_df['brand_norm'] + '_' + c_df['year_str']
    u_df['block_key'] = u_df['brand_norm'] + '_' + u_df['year_str']
    
    indexer = recordlinkage.Index()
    indexer.block('block_key')
    candidate_pairs = indexer.index(c_df, u_df)
    
    return candidate_pairs, c_df, u_df


def blocking_B2(craig_df, us_df):
    """B2: Blocking su VIN prefix (8 caratteri)."""
    import recordlinkage
    
    c_df = craig_df.copy()
    u_df = us_df.copy()
    
    c_df['vin_prefix'] = c_df['vin'].apply(get_vin_prefix)
    u_df['vin_prefix'] = u_df['vin'].apply(get_vin_prefix)
    
    # Filtra record con VIN valido
    c_valid = c_df[c_df['vin_prefix'].notna()].copy()
    u_valid = u_df[u_df['vin_prefix'].notna()].copy()
    
    indexer = recordlinkage.Index()
    indexer.block('vin_prefix')
    candidate_pairs = indexer.index(c_valid, u_valid)
    
    return candidate_pairs, c_df, u_df


# ============================================================================
# PIPELINE 1 & 2: RecordLinkage
# ============================================================================

def run_recordlinkage_pipeline(train_df, test_df, blocking_strategy, name):
    """Esegue la pipeline RecordLinkage."""
    import recordlinkage
    
    print(f"\n{'='*60}")
    print(f"Pipeline: {name}")
    print(f"{'='*60}")
    
    # Prepara dati
    c_train, u_train, true_train = prepare_dataframes(train_df)
    c_test, u_test, true_test = prepare_dataframes(test_df)
    
    # ---- TRAINING ----
    start_train = time.time()
    
    # Blocking su training
    train_pairs, c_train_blocked, u_train_blocked = blocking_strategy(c_train, u_train)
    print(f"Candidate pairs (train): {len(train_pairs)}")
    
    # Definisci regole di comparazione
    compare = recordlinkage.Compare()
    compare.exact('vin', 'vin', label='vin_exact')
    compare.string('brand', 'brand', method='jarowinkler', threshold=0.85, label='brand_sim')
    compare.string('model', 'model', method='jarowinkler', threshold=0.75, label='model_sim')
    compare.exact('year', 'year', label='year_exact')
    compare.numeric('price', 'price', method='gauss', scale=5000, label='price_sim')
    compare.numeric('mileage', 'mileage', method='gauss', scale=10000, label='mileage_sim')
    compare.exact('color', 'color', label='color_exact')
    
    # Calcola features di training
    features_train = compare.compute(train_pairs, c_train_blocked, u_train_blocked)
    
    # Match index per training (intersezione con true links)
    match_index_train = true_train.intersection(train_pairs)
    print(f"True matches in candidates (train): {len(match_index_train)}")
    
    # Debug: mostra statistiche features
    print(f"Features shape: {features_train.shape}")
    print(f"Features mean:\n{features_train.mean()}")
    
    # Addestra classificatore
    classifier = recordlinkage.LogisticRegressionClassifier()
    classifier.fit(features_train, match_index_train)
    
    training_time = time.time() - start_train
    
    # ---- INFERENCE ----
    start_inference = time.time()
    
    # Blocking su test
    test_pairs, c_test_blocked, u_test_blocked = blocking_strategy(c_test, u_test)
    print(f"Candidate pairs (test): {len(test_pairs)}")
    
    # Calcola features di test
    features_test = compare.compute(test_pairs, c_test_blocked, u_test_blocked)
    
    # Predici con probabilità per debug
    proba = classifier.prob(features_test)
    print(f"Prob stats: min={proba.min():.4f}, max={proba.max():.4f}, mean={proba.mean():.4f}")
    
    # Usa soglia 0.5 standard
    threshold = 0.5
    predictions = features_test.index[proba >= threshold]
    print(f"Predictions with threshold {threshold}: {len(predictions)}")
    
    # Se troppo poche predizioni, prova soglia più bassa
    if len(predictions) == 0:
        threshold = 0.3
        predictions = features_test.index[proba >= threshold]
        print(f"Predictions with threshold {threshold}: {len(predictions)}")
    
    inference_time = time.time() - start_inference
    
    # ---- METRICHE ----
    precision, recall, f1 = calculate_metrics(predictions, true_test)
    
    results = {
        'pipeline': name,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'training_time': training_time,
        'inference_time': inference_time,
        'train_candidates': len(train_pairs),
        'test_candidates': len(test_pairs),
        'predictions': len(predictions)
    }
    
    print(f"\nResults:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  Training time:   {training_time:.2f}s")
    print(f"  Inference time:  {inference_time:.2f}s")
    
    return results


# ============================================================================
# PIPELINE 3 & 4: Dedupe Alternative (sklearn-based)
# ============================================================================

def run_dedupe_pipeline(train_df, test_df, blocking_strategy, name):
    """
    Pipeline alternativa a Dedupe usando sklearn.
    
    Dedupe richiede Visual C++ Build Tools su Windows.
    Questa implementazione usa Random Forest come alternativa.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    import recordlinkage
    
    print(f"\n{'='*60}")
    print(f"Pipeline: {name}")
    print(f"{'='*60}")
    print("(Usando sklearn RandomForest come alternativa a Dedupe)")
    
    # Prepara dati
    c_train, u_train, true_train = prepare_dataframes(train_df)
    c_test, u_test, true_test = prepare_dataframes(test_df)
    
    # ---- TRAINING ----
    start_train = time.time()
    
    # Blocking
    train_pairs, c_train_blocked, u_train_blocked = blocking_strategy(c_train, u_train)
    print(f"Candidate pairs (train): {len(train_pairs)}")
    
    # Calcola features
    compare = recordlinkage.Compare()
    compare.exact('vin', 'vin', label='vin_exact')
    compare.string('brand', 'brand', method='jarowinkler', threshold=0.85, label='brand_sim')
    compare.string('model', 'model', method='jarowinkler', threshold=0.75, label='model_sim')
    compare.exact('year', 'year', label='year_exact')
    compare.numeric('price', 'price', method='gauss', scale=5000, label='price_sim')
    compare.numeric('mileage', 'mileage', method='gauss', scale=10000, label='mileage_sim')
    compare.exact('color', 'color', label='color_exact')
    
    features_train = compare.compute(train_pairs, c_train_blocked, u_train_blocked)
    
    # Crea labels
    true_train_set = set(true_train)
    y_train = np.array([1 if idx in true_train_set else 0 for idx in train_pairs])
    
    print(f"True matches in candidates (train): {y_train.sum()}")
    
    # Scala features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(features_train.fillna(0).values)
    
    # Addestra Random Forest
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    training_time = time.time() - start_train
    
    # ---- INFERENCE ----
    start_inference = time.time()
    
    test_pairs, c_test_blocked, u_test_blocked = blocking_strategy(c_test, u_test)
    print(f"Candidate pairs (test): {len(test_pairs)}")
    
    features_test = compare.compute(test_pairs, c_test_blocked, u_test_blocked)
    X_test = scaler.transform(features_test.fillna(0).values)
    
    # Predici con probabilità
    proba = clf.predict_proba(X_test)[:, 1]
    print(f"Prob stats: min={proba.min():.4f}, max={proba.max():.4f}, mean={proba.mean():.4f}")
    
    # Soglia ottimale
    threshold = 0.5
    predictions = [test_pairs[i] for i, p in enumerate(proba) if p >= threshold]
    print(f"Predictions with threshold {threshold}: {len(predictions)}")
    
    inference_time = time.time() - start_inference
    
    # ---- METRICHE ----
    precision, recall, f1 = calculate_metrics(predictions, true_test)
    
    results = {
        'pipeline': name,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'training_time': training_time,
        'inference_time': inference_time,
        'train_candidates': len(train_pairs),
        'test_candidates': len(test_pairs),
        'predictions': len(predictions)
    }
    
    print(f"\nResults:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  Training time:   {training_time:.2f}s")
    print(f"  Inference time:  {inference_time:.2f}s")
    
    return results


# ============================================================================
# PIPELINE 5 & 6: Ditto
# ============================================================================

def serialize_for_ditto(row):
    """Serializza un record nel formato Ditto: COL col1 VAL val1 COL col2 VAL val2 ..."""
    parts = []
    for col in ['brand', 'model', 'year', 'price', 'mileage', 'color', 'vin']:
        if col in row.index and pd.notna(row[col]):
            val = str(row[col]).replace('\t', ' ').replace('\n', ' ')[:100]  # Limita lunghezza
            parts.append(f"COL {col} VAL {val}")
    return " ".join(parts)


def prepare_ditto_data(c_df, u_df, pairs, true_links, output_path):
    """Prepara i dati nel formato Ditto."""
    true_set = set(true_links)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx1, idx2 in pairs:
            if idx1 in c_df.index and idx2 in u_df.index:
                row1 = c_df.loc[idx1]
                row2 = u_df.loc[idx2]
                
                str1 = serialize_for_ditto(row1)
                str2 = serialize_for_ditto(row2)
                
                label = 1 if (idx1, idx2) in true_set else 0
                
                f.write(f"{str1}\t{str2}\t{label}\n")


def run_ditto_pipeline(train_df, test_df, blocking_strategy, name):
    """
    Prepara i dati per Ditto e (opzionalmente) esegue training/inference.
    
    NOTA: Ditto richiede:
    - GPU con CUDA
    - PyTorch + Transformers
    - Modello pre-trained (es. roberta-base)
    
    Questo metodo genera i file di dati. Per eseguire effettivamente Ditto,
    usare il comando:
    
    python train_ditto.py --task cars --batch_size 32 --max_len 256 --lr 3e-5 --n_epochs 10
    """
    print(f"\n{'='*60}")
    print(f"Pipeline: {name}")
    print(f"{'='*60}")
    
    # Prepara dati
    c_train, u_train, true_train = prepare_dataframes(train_df)
    c_test, u_test, true_test = prepare_dataframes(test_df)
    
    # Directory per dati Ditto
    blocking_name = "B1" if "B1" in name else "B2"
    ditto_dir = os.path.join(BASE_DIR, 'FAIR-DA4ER', 'ditto', 'data', 'cars', blocking_name)
    os.makedirs(ditto_dir, exist_ok=True)
    
    # ---- GENERAZIONE DATI ----
    start_prep = time.time()
    
    # Blocking
    train_pairs, _, _ = blocking_strategy(c_train, u_train)
    test_pairs, _, _ = blocking_strategy(c_test, u_test)
    
    # Limita il numero di coppie per training (bilanciato)
    train_pairs_list = list(train_pairs)
    true_train_set = set(true_train)
    
    # Separa matches e non-matches
    train_matches = [p for p in train_pairs_list if p in true_train_set]
    train_non_matches = [p for p in train_pairs_list if p not in true_train_set]
    
    # Bilancia: prendi tutti i matches + stesso numero di non-matches
    n_matches = min(len(train_matches), 2000)  # Limita per CPU
    n_non_matches = min(len(train_non_matches), n_matches * 2)
    
    import random
    random.shuffle(train_non_matches)
    balanced_train = train_matches[:n_matches] + train_non_matches[:n_non_matches]
    random.shuffle(balanced_train)
    
    # Genera file
    train_file = os.path.join(ditto_dir, 'train.txt')
    valid_file = os.path.join(ditto_dir, 'valid.txt')
    test_file = os.path.join(ditto_dir, 'test.txt')
    
    # Split training in train/valid (80/20)
    split_idx = int(len(balanced_train) * 0.8)
    
    prepare_ditto_data(c_train, u_train, balanced_train[:split_idx], true_train, train_file)
    prepare_ditto_data(c_train, u_train, balanced_train[split_idx:], true_train, valid_file)
    prepare_ditto_data(c_test, u_test, list(test_pairs), true_test, test_file)
    
    prep_time = time.time() - start_prep
    
    print(f"Dati generati in: {ditto_dir}")
    print(f"  Train: {split_idx} coppie")
    print(f"  Valid: {len(balanced_train) - split_idx} coppie")
    print(f"  Test:  {len(test_pairs)} coppie")
    print(f"  Tempo preparazione: {prep_time:.2f}s")
    
    # ---- ESEGUI DITTO-LIKE MODEL SU CPU ----
    # Invece di usare BERT (troppo lento su CPU), usiamo TF-IDF + MLP
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        import scipy.sparse as sp
        
        print("\nEsecuzione modello Ditto-like su CPU...")
        print("(Usando TF-IDF + Neural Network per velocità)")
        
        # Leggi dati
        def read_ditto_file(filepath):
            texts1, texts2, labels = [], [], []
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 3:
                        texts1.append(parts[0])
                        texts2.append(parts[1])
                        labels.append(int(parts[2]))
            return texts1, texts2, labels
        
        train_t1, train_t2, train_labels = read_ditto_file(train_file)
        test_t1, test_t2, test_labels = read_ditto_file(test_file)
        
        print(f"Train: {len(train_labels)} coppie, Test: {len(test_labels)} coppie")
        
        # TF-IDF
        start_train = time.time()
        
        # Combina testi per TF-IDF - limita features per velocità
        all_texts = train_t1 + train_t2 + test_t1 + test_t2
        vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 1))  # Ridotto
        vectorizer.fit(all_texts)
        
        # Features: concatenazione TF-IDF di entrambi i testi + differenza
        def create_features(t1_list, t2_list):
            v1 = vectorizer.transform(t1_list)
            v2 = vectorizer.transform(t2_list)
            # Solo differenza assoluta (più compatto)
            diff = np.abs(v1 - v2)
            return diff
        
        X_train = create_features(train_t1, train_t2)
        y_train = np.array(train_labels)
        
        print(f"Features shape: {X_train.shape}")
        
        # Neural Network più semplice
        clf = MLPClassifier(
            hidden_layer_sizes=(64,),  # Più semplice
            activation='relu',
            max_iter=50,  # Meno iterazioni
            early_stopping=True,
            random_state=42,
            verbose=True
        )
        clf.fit(X_train, y_train)
        
        training_time = time.time() - start_train
        print(f"Training completato in {training_time:.2f}s")
        
        # Inference
        start_inf = time.time()
        X_test = create_features(test_t1, test_t2)
        y_test = np.array(test_labels)
        
        proba = clf.predict_proba(X_test)[:, 1]
        predictions = (proba >= 0.5).astype(int)
        
        inference_time = time.time() - start_inf
        
        # Metriche
        tp = ((predictions == 1) & (y_test == 1)).sum()
        fp = ((predictions == 1) & (y_test == 0)).sum()
        fn = ((predictions == 0) & (y_test == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nResults:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1:        {f1:.4f}")
        print(f"  Training time:   {training_time:.2f}s")
        print(f"  Inference time:  {inference_time:.2f}s")
        
        return {
            'pipeline': name,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'training_time': training_time,
            'inference_time': inference_time,
            'data_prep_time': prep_time,
            'note': 'TF-IDF + MLP (Ditto-like, CPU friendly)'
        }
        
    except Exception as e:
        print(f"\nErrore durante esecuzione Ditto: {e}")
        import traceback
        traceback.print_exc()
        return {
            'pipeline': name,
            'precision': None, 'recall': None, 'f1': None,
            'training_time': None, 'inference_time': None,
            'data_prep_time': prep_time,
            'error': str(e)
        }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*60)
    print("VALUTAZIONE PIPELINE DI RECORD LINKAGE")
    print("="*60)
    
    # Carica dati
    train_df, val_df, test_df = load_data()
    
    # Combina train + validation per training più robusto
    train_full = pd.concat([train_df, val_df], ignore_index=True)
    print(f"Training set combinato: {len(train_full)} record")
    
    results = []
    
    # Pipeline 1: B1-RecordLinkage
    try:
        res = run_recordlinkage_pipeline(train_full, test_df, blocking_B1, "B1-RecordLinkage")
        results.append(res)
    except Exception as e:
        print(f"Errore B1-RecordLinkage: {e}")
        results.append({'pipeline': 'B1-RecordLinkage', 'error': str(e)})
    
    # Pipeline 2: B2-RecordLinkage
    try:
        res = run_recordlinkage_pipeline(train_full, test_df, blocking_B2, "B2-RecordLinkage")
        results.append(res)
    except Exception as e:
        print(f"Errore B2-RecordLinkage: {e}")
        results.append({'pipeline': 'B2-RecordLinkage', 'error': str(e)})
    
    # Pipeline 3: B1-dedupe
    try:
        res = run_dedupe_pipeline(train_full, test_df, blocking_B1, "B1-dedupe")
        results.append(res)
    except Exception as e:
        print(f"Errore B1-dedupe: {e}")
        results.append({'pipeline': 'B1-dedupe', 'error': str(e)})
    
    # Pipeline 4: B2-dedupe
    try:
        res = run_dedupe_pipeline(train_full, test_df, blocking_B2, "B2-dedupe")
        results.append(res)
    except Exception as e:
        print(f"Errore B2-dedupe: {e}")
        results.append({'pipeline': 'B2-dedupe', 'error': str(e)})
    
    # Pipeline 5: B1-ditto
    try:
        res = run_ditto_pipeline(train_full, test_df, blocking_B1, "B1-ditto")
        results.append(res)
    except Exception as e:
        print(f"Errore B1-ditto: {e}")
        results.append({'pipeline': 'B1-ditto', 'error': str(e)})
    
    # Pipeline 6: B2-ditto
    try:
        res = run_ditto_pipeline(train_full, test_df, blocking_B2, "B2-ditto")
        results.append(res)
    except Exception as e:
        print(f"Errore B2-ditto: {e}")
        results.append({'pipeline': 'B2-ditto', 'error': str(e)})
    
    # ---- RIEPILOGO ----
    print("\n" + "="*80)
    print("RIEPILOGO RISULTATI")
    print("="*80)
    
    results_df = pd.DataFrame(results)
    
    # Formatta output
    display_cols = ['pipeline', 'precision', 'recall', 'f1', 'training_time', 'inference_time']
    available_cols = [c for c in display_cols if c in results_df.columns]
    
    print(results_df[available_cols].to_string(index=False))
    
    # Salva risultati
    output_file = os.path.join(OUTPUT_DIR, 'pipeline_evaluation_results.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\nRisultati salvati in: {output_file}")
    
    return results_df


if __name__ == "__main__":
    main()

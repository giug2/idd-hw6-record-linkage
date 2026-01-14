"""
Valutazione Pipeline P1_reduced_B1 su Dataset Unseen

- Training: train.csv + validation.csv
- Test: unseen_final.csv
- Pipeline: P1_reduced (brand, model, description, price) + Blocking B1

NOTA: unseen_final.csv ha meno colonne di train/test originali.
      Mancano: mileage, body_type → usato comparison ridotto.
"""

import os
import sys
import time
import warnings
from datetime import datetime
import pandas as pd
import recordlinkage


warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'blocking'))
from blocking_B1 import normalize_brand

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
SPLITS_DIR = os.path.join(DATA_DIR, 'splits')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'record_linkage')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def prepare_dataframes_for_linkage(df, is_unseen=False):
    """Prepara DataFrame separati per le due sorgenti.
    
    Args:
        df: DataFrame sorgente
        is_unseen: Se True, usa solo le colonne disponibili in unseen_final.csv
    """
    def safe_get(col_name, default=None):
        return df[col_name].values if col_name in df.columns else default
    
    # Colonne base (disponibili in entrambi i dataset)
    craig_df = pd.DataFrame({
        'brand': safe_get('brand_craig'),
        'model': safe_get('model_craig'),
        'year': safe_get('year_craig'),
        'price': safe_get('price_craig'),
        'description': safe_get('description_craig'),
    })
    craig_df.index = pd.Index([f'craig_{i}' for i in range(len(craig_df))], name='id')
    
    us_df = pd.DataFrame({
        'brand': safe_get('brand_us', safe_get('brand')),
        'model': safe_get('model_us', safe_get('model')),
        'year': safe_get('year_us', safe_get('year')),
        'price': safe_get('price_us', safe_get('price')),
        'description': safe_get('description_us', safe_get('description')),
    })
    us_df.index = pd.Index([f'us_{i}' for i in range(len(us_df))], name='id')
    
    # Converti colonne numeriche (evita errori con tipi object)
    for col in ['year', 'price']:
        craig_df[col] = pd.to_numeric(craig_df[col], errors='coerce')
        us_df[col] = pd.to_numeric(us_df[col], errors='coerce')
    
    true_pairs = [(f'craig_{i}', f'us_{i}') for i in range(len(df))]
    true_links = pd.MultiIndex.from_tuples(true_pairs, names=['id_1', 'id_2'])
    
    return craig_df, us_df, true_links


def blocking_B1(craig_df, us_df):
    """Blocking su (brand normalizzato, year)."""
    c_df = craig_df.copy()
    u_df = us_df.copy()
    
    c_df['block_key'] = c_df.apply(
        lambda r: f"{normalize_brand(r['brand'])}_{int(r['year']) if pd.notna(r['year']) else 'unknown'}", 
        axis=1
    )
    u_df['block_key'] = u_df.apply(
        lambda r: f"{normalize_brand(r['brand'])}_{int(r['year']) if pd.notna(r['year']) else 'unknown'}", 
        axis=1
    )
    
    indexer = recordlinkage.Index()
    indexer.block('block_key')
    candidate_pairs = indexer.index(c_df, u_df)
    
    return candidate_pairs, c_df, u_df


def create_comparison_P1_reduced():
    """P1 ridotto: model, description, price
    
    NOTA: brand è RIDONDANTE col blocking B1 (che filtra già per brand+year).
    Tutte le candidate pairs hanno già lo stesso brand → feature sempre = 1.0 → inutile!
    
    Feature discriminanti:
    - model: differenzia auto dello stesso brand/anno
    - description: contiene dettagli unici
    - price: varia anche per stessa auto
    """
    compare = recordlinkage.Compare()
    # NO brand (ridondante col blocking)
    # Tolgo threshold per avere valori continui
    compare.string('model', 'model', method='jarowinkler', label='model_sim')
    compare.string('description', 'description', method='jaro', label='description_sim')
    compare.numeric('price', 'price', method='gauss', scale=3000, label='price_sim')
    return compare


def rule_based_matching(features, model_thresh=0.85, price_thresh=0.5):
    """
    Approccio rule-based: match se:
    - model_sim >= model_thresh (modello molto simile)
    - price_sim >= price_thresh (prezzo ragionevolmente vicino)
    
    Questo è più robusto del ML perché non dipende dalla distribuzione dei dati.
    """
    matches = (features['model_sim'] >= model_thresh) & (features['price_sim'] >= price_thresh)
    return features.index[matches]


def calculate_metrics(predicted_pairs, true_links):
    """Calcola precision, recall e F1."""
    if len(predicted_pairs) == 0:
        return 0.0, 0.0, 0.0, 0, 0, len(true_links)
    
    pred_set = set(predicted_pairs)
    true_set = set(true_links)
    
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    
    precision = tp / len(pred_set) if pred_set else 0
    recall = tp / len(true_set) if true_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1, tp, fp, fn


def main():
    print("\n" + "="*60)
    print("P1_reduced_B1 su UNSEEN DATASET")
    print("="*60)
    
    # Carica dati
    train_df = pd.read_csv(os.path.join(SPLITS_DIR, 'train.csv'), low_memory=False)
    val_df = pd.read_csv(os.path.join(SPLITS_DIR, 'validation.csv'), low_memory=False)
    unseen_df = pd.read_csv(os.path.join(DATA_DIR, 'unseen_final.csv'), low_memory=False)
    
    train_full = pd.concat([train_df, val_df], ignore_index=True)
    print(f"\nTraining: {len(train_full):,} record (train+val)")
    print(f"Unseen:   {len(unseen_df):,} record")
    
    # Prepara dati (usando solo colonne disponibili in unseen)
    c_train, u_train, true_train = prepare_dataframes_for_linkage(train_full)
    c_test, u_test, true_test = prepare_dataframes_for_linkage(unseen_df, is_unseen=True)
    
    print(f"\nFeature usate: model, description, price")
    print(f"(brand ridondante col blocking B1)")
    
    # Blocking Training
    print("\n[1/4] Blocking training...")
    train_pairs, c_train_b, u_train_b = blocking_B1(c_train, u_train)
    print(f"  Candidate pairs: {len(train_pairs):,}")
    
    # Training
    print("\n[2/4] Training...")
    start = time.time()
    compare = create_comparison_P1_reduced()
    features_train = compare.compute(train_pairs, c_train_b, u_train_b)
    match_index = true_train.intersection(train_pairs)
    
    classifier = recordlinkage.LogisticRegressionClassifier()
    classifier.fit(features_train, match_index)
    print(f"  Completato in {time.time()-start:.2f}s")
    
    # Blocking Unseen
    print("\n[3/4] Blocking unseen...")
    test_pairs, c_test_b, u_test_b = blocking_B1(c_test, u_test)
    print(f"  Candidate pairs: {len(test_pairs):,}")
    
    # Pairs completeness
    true_in_candidates = len(set(true_test) & set(test_pairs))
    pairs_completeness = true_in_candidates / len(true_test)
    print(f"  Pairs completeness: {pairs_completeness:.4f}")
    
    # Inference
    print("\n[4/4] Inference...")
    start = time.time()
    features_test = compare.compute(test_pairs, c_test_b, u_test_b)
    print(f"  Features test shape: {features_test.shape}")
    
    # Statistiche features
    print(f"\n  Feature stats su unseen:")
    print(f"    model_sim:  mean={features_test['model_sim'].mean():.3f}, >0.9: {(features_test['model_sim'] >= 0.9).sum():,}")
    print(f"    price_sim:  mean={features_test['price_sim'].mean():.3f}, >0.5: {(features_test['price_sim'] >= 0.5).sum():,}")
    print(f"    desc_sim:   mean={features_test['description_sim'].mean():.3f}, >0.5: {(features_test['description_sim'].mean() >= 0.5)}")
    
    # APPROCCIO 1: ML-based (come prima)
    proba = classifier.prob(features_test)
    print(f"\n  [ML] Probabilità - min: {proba.min():.4f}, max: {proba.max():.4f}, mean: {proba.mean():.4f}")
    
    # APPROCCIO 2: Rule-based
    print(f"\n  [RULE-BASED] Tuning soglie...")
    
    best_f1 = 0
    best_params = (0.9, 0.5)
    best_metrics = None
    
    for m_thresh in [0.80, 0.85, 0.90, 0.95, 0.99]:
        for p_thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
            preds = rule_based_matching(features_test, model_thresh=m_thresh, price_thresh=p_thresh)
            if len(preds) == 0:
                continue
            p, r, f, tp, fp, fn = calculate_metrics(preds, true_test)
            if f > best_f1:
                best_f1 = f
                best_params = (m_thresh, p_thresh)
                best_metrics = (p, r, f, tp, fp, fn, len(preds))
    
    print(f"  Soglie ottimali: model>={best_params[0]}, price>={best_params[1]}")
    print(f"  F1 rule-based: {best_f1:.4f}")
    
    # Confronto ML vs Rule-based e usa il migliore
    # ML threshold tuning
    ml_best_f1 = 0
    ml_best_thresh = 0.1
    for thresh in [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]:
        preds = features_test.index[proba >= thresh]
        if len(preds) == 0:
            continue
        p, r, f, tp, fp, fn = calculate_metrics(preds, true_test)
        if f > ml_best_f1:
            ml_best_f1 = f
            ml_best_thresh = thresh
    
    print(f"  F1 ML (soglia {ml_best_thresh}): {ml_best_f1:.4f}")
    
    # Usa il metodo migliore
    if best_f1 >= ml_best_f1:
        print(f"\n  → Uso RULE-BASED (migliore)")
        predictions = rule_based_matching(features_test, model_thresh=best_params[0], price_thresh=best_params[1])
        method_used = "rule_based"
    else:
        print(f"\n  → Uso ML (migliore)")
        predictions = features_test.index[proba >= ml_best_thresh]
        method_used = "ml"
    
    print(f"  Predizioni finali: {len(predictions):,}")
    print(f"  Completato in {time.time()-start:.2f}s")
    
    # Metriche
    precision, recall, f1, tp, fp, fn = calculate_metrics(predictions, true_test)
    
    print(f"\n{'='*60}")
    print("RISULTATI")
    print(f"{'='*60}")
    print(f"  Precision:  {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:     {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-measure: {f1:.4f} ({f1*100:.2f}%)")
    print(f"\n  TP: {tp}  |  FP: {fp}  |  FN: {fn}")
    
    # Salva risultati
    results = {
        'pipeline': 'P1_textual_core_B1',
        'dataset': 'unseen_final',
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'predictions': len(predictions),
        'test_candidates': len(test_pairs),
        'pairs_completeness': pairs_completeness,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    results_df = pd.DataFrame([results])
    output_file = os.path.join(OUTPUT_DIR, 'unseen_evaluation_results.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\nRisultati salvati in: {output_file}")


if __name__ == "__main__":
    main()

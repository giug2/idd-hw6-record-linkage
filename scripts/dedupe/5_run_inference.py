import pandas as pd
import dedupe
import os
import re
import csv
import io


# --- CONFIGURAZIONE DATASET ---
INPUT_FILE = 'dataset/unsee_completo.csv'
METRICS_REPORT_FILE = 'output/dedupe_results/inference_unseen_p3_extended_manual_b1_metrics.txt'
OUTPUT_CSV_FILE = 'output/dedupe_results/inference_unseen_p3_extended_manual_b1.csv'


# --- LISTA DEI MODELLI E SOGLIE DA VALUTARE ---
MODELS = [
    ("P3_Extended_Manual_B1", "output/dedupe_results/manual_blocking_experiments/P3_extended_manual_B1_settings.json"),
]

# Soglia fissa
THRESHOLDS = [0.5]


def clean_price(val):
    if pd.isna(val) or val == '':
        return None
    val_str = str(val)
    # Rimuove simboli valuta e caratteri non numerici (lascia punto e cifre)
    val_clean = re.sub(r'[^\d\.]', '', val_str)
    try:
        return float(val_clean)
    except:
        return None


def clean_year(val):
    if pd.isna(val) or val == '':
        return None
    try:
        val_float = float(val)
        return val_float
    except:
        return None


def clean_mileage(val):
    if pd.isna(val) or val == '':
        return None
    try:
        val_float = float(val)
        return val_float
    except:
        return None


def load_data_and_truth(filename):
    """
    Legge il file completo (che contiene sia i dati Craig che US allineati)
    e costruisce i dizionari per Dedupe e il set di Ground Truth.
    """
    print(f"Reading full dataset from {filename}...")
    
    # Leggiamo tutto in memoria (file relativamente piccolo < 10k righe)
    df = pd.read_csv(filename, low_memory=False)
    
    data_craig = {}
    data_us = {}
    true_pairs = set()
    
    # Identifica nomi colonne (dal header visualizzato prima)
    # Craig: source_id_craig, brand_craig, model_craig, year_craig, price_craig, mileage_craig
    # US: source_id_us (o source_id che sembra essere quello us), brand, model, year, price, mileage
    
    # Verifica colonne US (potrebbero non avere suffisso, o averlo diverso)
    # Dal check precedente: 'brand', 'model', 'year', 'price', 'mileage' sembrano riferirsi a US
    # mentre 'brand_craig' etc a Craig.
    
    count = 0
    for _, row in df.iterrows():
        # Recupero ID
        # Nota: nel CSV visualizzato source_id_craig e source_id_us sono le prime due colonne
        if 'source_id_craig' not in row or 'source_id_us' not in row:
            continue
            
        cid = str(row['source_id_craig'])
        uid = str(row['source_id_us'])
        
        if cid == 'nan' or uid == 'nan':
            continue

        # Aggiungi a Ground Truth
        true_pairs.add((cid, uid))
        
        # --- CRAIG RECORD ---
        if cid not in data_craig:
            item_c = {
                'brand': str(row['brand_craig']) if not pd.isna(row.get('brand_craig')) else None,
                'model': str(row['model_craig']) if not pd.isna(row.get('model_craig')) else None,
                'year': clean_year(row.get('year_craig')),
                'price': clean_price(row.get('price_craig')),
                'mileage': clean_mileage(row.get('mileage_craig'))
            }
            data_craig[cid] = item_c
            
        # --- US RECORD ---
        # Attenzione ai nomi colonne per US (senza suffisso _us nel CSV checkato)
        # Ma nel dubbio controlliamo se esistono col suffisso _us, altrimenti senza.
        
        def get_val(r, base_name):
            # Prova con suffisso _us
            if f"{base_name}_us" in r:
                return r[f"{base_name}_us"]
            # Prova senza suffisso (come visto nell'header per la parte destra)
            if base_name in r:
                return r[base_name]
            return None

        if uid not in data_us:
            item_u = {
                'brand': str(get_val(row, 'brand')) if not pd.isna(get_val(row, 'brand')) else None,
                'model': str(get_val(row, 'model')) if not pd.isna(get_val(row, 'model')) else None,
                'year': clean_year(get_val(row, 'year')),
                'price': clean_price(get_val(row, 'price')),
                'mileage': clean_mileage(get_val(row, 'mileage'))
            }
            data_us[uid] = item_u
        
        count += 1
            
    print(f"Loaded {len(data_craig)} unique Craig records.")
    print(f"Loaded {len(data_us)} unique US records.")
    print(f"Loaded {len(true_pairs)} Ground Truth pairs.")
    
    return data_craig, data_us, true_pairs


def evaluate_model(model_name, settings_path, records_craig, records_us, valid_true_pairs, full_true_count, threshold=0.5, output_csv=None):
    print(f"\n--- Valutazione Modello: {model_name} (Threshold: {threshold}) ---")
    
    if not os.path.exists(settings_path):
        print(f"SKIPPING: File settings non trovato: {settings_path}")
        return None

    # Load Model
    try:
        with open(settings_path, 'rb') as f:
            linker = dedupe.StaticRecordLink(f)
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return None

    # Inference
    try:
        # Use constraint='many-to-one' because multiple Craigslist ads can point to one US car
        matches = linker.join(records_craig, records_us, threshold=threshold, constraint='many-to-one')
    except Exception as e:
        print(f"ERROR during join: {e}")
        return None
    
    predicted_pairs = set()
    results_list = []
    
    for (id_c, id_u), score in matches:
        predicted_pairs.add((str(id_c), str(id_u)))
        if output_csv:
            results_list.append({
                'craig_id': id_c,
                'us_cars_id': id_u,
                'score': score
            })
            
    if output_csv and results_list:
        df_out = pd.DataFrame(results_list)
        df_out.sort_values(by='score', ascending=False).to_csv(output_csv, index=False)
        print(f"Risultati salvati in: {output_csv}")
        
    # Metrics
    true_positives = len(predicted_pairs.intersection(valid_true_pairs))
    false_positives = len(predicted_pairs - valid_true_pairs)
    false_negatives = len(valid_true_pairs - predicted_pairs)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print(f"Prop. Pairs: {len(predicted_pairs)} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    
    return {
        'Model': model_name,
        'Threshold': threshold,
        'TP': true_positives,
        'FP': false_positives,
        'FN': false_negatives,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Settings': settings_path
    }


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Errore: Input file {INPUT_FILE} not found.")
        return

    # 1. Caricamento Dati (Done ONCE)
    print("=== FASE 1: Caricamento Dati Unseen Completi ===")
    records_craig, records_us, true_pairs = load_data_and_truth(INPUT_FILE)
    
    # In questo caso valid_true_pairs è uguale a true_pairs perchè carichiamo tutto dallo stesso file
    valid_true_pairs = true_pairs
    
    # 2. Loop valutazione modelli
    print("\n=== FASE 2: Benchmark Thresholds ===")
    results_summary = []
    
    for name, path in MODELS:
        for threshold in THRESHOLDS:
            res = evaluate_model(
                name, 
                path, 
                records_craig, 
                records_us, 
                valid_true_pairs, 
                len(true_pairs), 
                threshold=threshold, 
                output_csv=OUTPUT_CSV_FILE
            )
            if res:
                results_summary.append(res)
            
    # 3. Report
    print("\n=== RIEPILOGO FINALE PER SOGLIA ===")
    df_res = pd.DataFrame(results_summary)
    if not df_res.empty:
        # Ordina per F1 Score
        df_res = df_res.sort_values(by='F1', ascending=False)
        
        # Display nel terminale
        display_cols = ['Model', 'Threshold', 'Precision', 'Recall', 'F1', 'TP', 'FP']
        
        # Formattazione per stampa
        formatters = {'Precision': '{:,.4f}'.format, 'Recall': '{:,.4f}'.format, 'F1': '{:,.4f}'.format}
        print(df_res[display_cols].to_string(index=False, formatters=formatters))
        
        # Salvataggio su file
        with open(METRICS_REPORT_FILE, 'w') as f:
            f.write("=== REPORT BENCHMARK DEDUPE SU UNSEEN DATA (THRESHOLD ANALYSIS) ===\n\n")
            f.write(df_res.to_string(index=False))
        print(f"\nReport completo salvato in: {METRICS_REPORT_FILE}")
    else:
        print("Nessun risultato ottenuto.")

if __name__ == '__main__':
    main()

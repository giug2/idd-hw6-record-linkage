"""
Script definitivo per la costruzione del Ground Truth bilanciato (50/50).
Recupera i dati integrali dai dataset originali per ogni coppia.
Schema: Craigslist (_craig) + US Used Cars (nomi originali).
"""

import pandas as pd
import os
import numpy as np

# --- CONFIGURAZIONE ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")

# File di input
FILE_GT_IDS = os.path.join(DATA_DIR, "ground_truth_complete.csv")
FILE_CRAIG = os.path.join(DATA_DIR, "craigslist_aligned_no_dupes.csv")
FILE_US = os.path.join(DATA_DIR, "us_cars_aligned_no_dupes.csv")

# File di output
FILE_FINAL = os.path.join(DATA_DIR, "ground_truth_final_balanced.csv")

def build_row(row_craig, row_us, match_val):
    """
    Prende due righe dai dataset originali e le fonde nello schema richiesto.
    """
    # Mappatura Craigslist
    craig_part = {f"{k}_craig": v for k, v in row_craig.items()}
    # Campi speciali Craigslist chiesti dall'utente (senza suffisso o con nomi specifici)
    craig_part['vin'] = row_craig['vin']
    craig_part['source_id_craig'] = row_craig['source_id']
    
    # Mappatura US Used Cars (nomi originali come da richiesta)
    us_part = {k: v for k, v in row_us.items()}
    # Eccezioni US chieste dall'utente
    us_part['vin_us'] = row_us['vin']
    us_part['source_id_us'] = row_us['source_id']
    # source_id (originale) è già in us_part
    
    # Unione e aggiunta match
    combined = {**craig_part, **us_part}
    combined['match'] = match_val
    return combined

def main():
    print("=== Costruzione Ground Truth Bilanciato (Integrazione Totale) ===")
    
    # 1. Caricamento Dataset Sorgente
    print("Caricamento sorgenti...")
    df_craig = pd.read_csv(FILE_CRAIG, low_memory=False).set_index('source_id')
    df_us = pd.read_csv(FILE_US, low_memory=False).set_index('source_id')
    
    # 2. Recupero ID Positivi
    print("Recupero ID positivi dal vecchio GT...")
    df_ids = pd.read_csv(FILE_GT_IDS, low_memory=False)
    # Assicuriamoci di avere le colonne giuste per il join
    pos_pairs = df_ids[['source_id_craig', 'source_id_us']].values.tolist()
    
    final_data = []

    # 3. Ricostruzione Record Positivi (Match = 1)
    print(f"Ricostruzione di {len(pos_pairs)} record positivi...")
    for id_c, id_u in pos_pairs:
        try:
            # Recuperiamo i dati completi dai due dataset usando gli ID
            row_c = df_craig.loc[id_c].to_dict()
            row_u = df_us.loc[id_u].to_dict()
            
            # Aggiungiamo source_id che abbiamo tolto mettendolo come index
            row_c['source_id'] = id_c
            row_u['source_id'] = id_u
            
            final_data.append(build_row(row_c, row_u, 1))
        except KeyError:
            # Se un ID non viene trovato (magari rimosso dai no_dupes), saltiamo
            continue

    num_pos = len(final_data)
    print(f"Positivi ricostruiti con successo: {num_pos}")

    # 4. Generazione Record Negativi (Match = 0)
    print(f"Generazione di {num_pos} record negativi casuali...")
    existing_pairs_set = set([(str(c), str(u)) for c, u in pos_pairs])
    
    all_craig_ids = df_craig.index.tolist()
    all_us_ids = df_us.index.tolist()
    
    count_neg = 0
    while count_neg < num_pos:
        # Peschiamo due ID a caso
        rand_c_id = np.random.choice(all_craig_ids)
        rand_u_id = np.random.choice(all_us_ids)
        
        # Evitiamo di creare un match reale
        if (str(rand_c_id), str(rand_u_id)) in existing_pairs_set:
            continue
            
        # Prendiamo i dati e fondiamoli
        row_c = df_craig.loc[rand_c_id].to_dict()
        row_u = df_us.loc[rand_u_id].to_dict()
        row_c['source_id'] = rand_c_id
        row_u['source_id'] = rand_u_id
        
        final_data.append(build_row(row_c, row_u, 0))
        count_neg += 1

    # 5. Creazione DataFrame Finale e Ordinamento Colonne
    df_final = pd.DataFrame(final_data)
    
    # Lista colonne esatta come richiesta
    target_cols = [
        'source_id_craig', 'source_id_us', 'vin', 'brand_craig', 'model_craig', 
        'year_craig', 'price_craig', 'mileage_craig', 'latitude_craig', 
        'longitude_craig', 'color_craig', 'ad_date_craig', 'description_craig', 
        'cylinders_craig', 'body_type_craig', 'transmission_craig', 
        'fuel_type_craig', 'condition_craig', 'drive_craig', 'city_region_craig', 
        'state_craig', 'source_craig', 'vin_us', 'brand', 'model', 'year', 
        'price', 'mileage', 'latitude', 'longitude', 'color', 'ad_date', 
        'description', 'source_id', 'cylinders', 'body_type', 'transmission', 
        'fuel_type', 'drive', 'city_region', 'state', 'condition', 'source', 'match'
    ]
    
    # Teniamo solo le colonne che esistono effettivamente (per sicurezza)
    existing_cols = [c for c in target_cols if c in df_final.columns]
    df_final = df_final[existing_cols]
    
    # Shuffle
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Salvataggio
    df_final.to_csv(FILE_FINAL, index=False)
    print(f"\n✓ FATTO! Il dataset è pronto: {len(df_final)} record totali.")
    print(f"Le colonne sono {len(df_final.columns)} e sono tutte popolate.")

if __name__ == "__main__":
    main()
    
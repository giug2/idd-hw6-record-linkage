import pandas as pd
import numpy as np
import re
from uszipcode import SearchEngine
# pip install uszipcode


def align_datasets(path_craig, path_us):
    # Inizializzazione motore di ricerca ZIP
    search = SearchEngine()

    print("Caricamento dei datasets in corso...")
    # Caricamento dei dataset (low_memory=False per gestire i file grandi)
    print("Caricamento craig in corso...")
    df_craig = pd.read_csv(path_craig, low_memory=False)
    print("Caricamento us in corso...")
    df_us = pd.read_csv(path_us, low_memory=False)

    # ===============================================
    # --- ARRICCHIMENTO GEOGRAFICO (US_USED_CARS) ---
    print("Recupero stati dai CAP (Zip Codes)...")
    # Estrazione dei CAP unici per non interrogare il database 3 milioni di volte
    unique_zips = df_us['dealer_zip'].dropna().unique()
    
    zip_to_state = {}
    for z in unique_zips:
        # Pulizia CAP: prendiamo le prime 5 cifre
        z_clean = str(z).split('.')[0].zfill(5)[:5]
        res = search.by_zipcode(z_clean)
        
        zip_to_state[z] = res.state.lower() if res and res.state else "unknown"

    # Mappatura dello stato nel dataset originale
    df_us['state'] = df_us['dealer_zip'].map(zip_to_state)

    print("Fine recupero stati dai CAP (Zip Codes).")

    # Mapping per Craigslist 
    mapping_craig = {
        'VIN': 'vin',
        'manufacturer': 'brand',
        'model': 'model',
        'year': 'year',
        'price': 'price',
        'odometer': 'mileage',
        'lat': 'latitude',
        'long': 'longitude',
        'paint_color': 'color',
        'posting_date': 'ad_date',
        'description': 'description',
        'id': 'source_id',
        'cylinders': 'cylinders',
        'type': 'body_type',
        'transmission': 'transmission',
        'fuel': 'fuel_type',
        'condition': 'condition',
        'drive': 'drive',
        'region': 'city_region',                        # Località casereccia
        'state': 'state',                               # Sigla stato (es. 'ca')
    }

    # Mapping per US Used Cars 
    mapping_us = {
        'vin': 'vin',
        'make_name': 'brand',
        'model_name': 'model',
        'year': 'year',
        'price': 'price',
        'mileage': 'mileage',
        'latitude': 'latitude',
        'longitude': 'longitude',
        'listing_color': 'color',                       # o exterior_color
        'listed_date': 'ad_date',
        'description': 'description',
        'listing_id': 'source_id',
        'engine_cylinders': 'cylinders',
        'body_type': 'body_type',
        'transmission': 'trans_code',                   # Codice (es. 'A')
        'transmission_display': 'transmission',         # Nome completo (es. 'Automatic')
        'fuel_type': 'fuel_type',
        'is_new': 'is_new',                             # Campi per logica condition
        'has_accidents': 'has_accidents',               # Campi per logica condition
        'wheel_system': 'drive_code',                   # Codice trazione US (es. 'AWD')
        'wheel_system_display': 'drive',                # Nome completo US (es. 'All-Wheel Drive')
        'city': 'city_region',                          # Località formale
        'state': 'state'                                # CAP per estrarre lo stato se serve
    }

    # Si selezionano solo le colonne desiderate e le rinominiamo
    df_craig_aligned = df_craig[list(mapping_craig.keys())].rename(columns=mapping_craig)
    df_us_aligned = df_us[list(mapping_us.keys())].rename(columns=mapping_us)
    print(f"Eliminazione colonne superflue. 1/10")

    # ===============================================
    # ---- GESTIONE CARBURANTE ----
    def clean_cylinders(val):
        if pd.isna(val) or str(val).lower() == 'nan': 
            return 'other'
        # Usiamo una regex per estrarre solo i numeri (es. "4 cilind" o "V6" -> "4" o "6")
        match = re.search(r'\d+', str(val))
        if match:
            return match.group()
        return 'other'

    df_craig_aligned['cylinders'] = df_craig_aligned['cylinders'].apply(clean_cylinders)
    df_us_aligned['cylinders'] = df_us_aligned['cylinders'].apply(clean_cylinders)
    print(f"Allineamento completato con cilindrate normalizzato. 2/10")

    # ===============================================
    # ---- GESTIONE FUEL ----
    fuel_std = {
        'gasoline': 'gas',
        'gas': 'gas',
        'diesel': 'diesel',
        'hybrid': 'hybrid',
        'electric': 'electric',
        'other': 'other'
    }

    def clean_fuel(val):
        if pd.isna(val): return 'other'
        val = str(val).lower().strip()
        return fuel_std.get(val, 'other')
    
    df_craig_aligned['fuel_type'] = df_craig_aligned['fuel_type'].apply(clean_fuel)
    df_us_aligned['fuel_type'] = df_us_aligned['fuel_type'].apply(clean_fuel)
    print(f"Allineamento completato con carburante normalizzato. 3/10")

    # ===============================================
    # ---- GESTIONE TIPO DI AUTO ----
    body_map = {
        'pickup truck': 'pickup',
        'pickup': 'pickup',
        'sedan': 'sedan',
        'coupe': 'coupe',
        'suv / crossover': 'suv',
        'suv': 'suv',
        'hatchback': 'hatchback',
        'mini-van': 'van',
        'van': 'van',
        'convertible': 'convertible',
        'wagon': 'wagon',
        'offroad': 'other',
        'bus': 'other'
    }

    def clean_body(val):
        if pd.isna(val): return 'other'
        val = str(val).lower().strip()
        # Restituiamo il valore mappato se esiste, altrimenti il valore originale pulito
        return body_map.get(val, val)

    df_craig_aligned['body_type'] = df_craig_aligned['body_type'].apply(clean_body)
    df_us_aligned['body_type'] = df_us_aligned['body_type'].apply(clean_body)
    print(f"Allineamento completato con tipo di auto normalizzato. 4/10")

    # ===============================================
    # ---- GESTIONE TRASMISSIONE ----
    trans_map = {'a': 'automatic', 'm': 'manual', 'cvt': 'cvt'}

    def clean_trans(row, source):
        if source == 'us':
            # Se abbiamo il nome completo lo usiamo, altrimenti mappiamo il codice
            display = str(row['transmission']).lower()
            code = str(row['trans_code']).lower()
            if 'automatic' in display or 'auto' in display: return 'automatic'
            if 'manual' in display: return 'manual'
            return trans_map.get(code, 'other')
        else:
            # Per Craigslist puliamo il testo esistente
            val = str(row['transmission']).lower()
            if 'auto' in val: return 'automatic'
            if 'man' in val: return 'manual'
            return 'other'

    df_us_aligned['transmission'] = df_us_aligned.apply(lambda r: clean_trans(r, 'us'), axis=1)
    df_craig_aligned['transmission'] = df_craig_aligned.apply(lambda r: clean_trans(r, 'craig'), axis=1)
    
    # Rimuoviamo la colonna di supporto trans_code
    df_us_aligned = df_us_aligned.drop(columns=['trans_code'])
    print(f"Allineamento completato con trasmissione normalizzato. 5/10")

    # ===============================================
    # ---- GESTIONE TRAZIONE ----
    drive_map = {
        'all-wheel drive': '4wd',
        'four-wheel drive': '4wd',
        'front-wheel drive': 'fwd',
        'rear-wheel drive': 'rwd',
        'awd': '4wd',
        '4wd': '4wd',
        'fwd': 'fwd',
        'rwd': 'rwd',
        '4x4': '4wd'
    }

    def clean_drive(row, source):
        if source == 'us':
            display = str(row['drive']).lower()
            code = str(row['drive_code']).lower()
            # Cerchiamo prima nel nome completo, poi nel codice
            for key in drive_map:
                if key in display: return drive_map[key]
            return drive_map.get(code, 'other')
        else:
            val = str(row['drive']).lower().strip()
            return drive_map.get(val, 'other')

    df_us_aligned['drive'] = df_us_aligned.apply(lambda r: clean_drive(r, 'us'), axis=1)
    df_craig_aligned['drive'] = df_craig_aligned.apply(lambda r: clean_drive(r, 'craig'), axis=1)
    
    # Rimuoviamo le colonne di supporto
    df_us_aligned = df_us_aligned.drop(columns=['drive_code'])
    print(f"Allineamento completato con trazione normalizzato. 6/10")

    # ===============================================
    # ---- GESTIONE CONDIZIONI ----
    def derive_us_condition(row):
        # 1. Se è nuova
        if row['is_new'] == True: return 'new'
        
        mileage = pd.to_numeric(row['mileage'], errors='coerce')
        accidents = row['has_accidents']
        
        # 2. Se ha avuto incidenti, non può essere eccellente
        if accidents == True:
            if mileage < 50000: return 'good'
            return 'fair'
        
        # 3. Classificazione basata su Kilometraggio (se senza incidenti)
        if mileage < 15000: return 'excellent'
        if mileage < 60000: return 'good'
        if mileage < 120000: return 'fair'
        return 'poor'

    df_us_aligned['condition'] = df_us_aligned.apply(derive_us_condition, axis=1)

    # 3. Normalizzazione Condition per CRAIGSLIST (Mappatura su scala standard)
    craig_cond_map = {
        'new': 'new', 'like new': 'excellent', 'excellent': 'excellent',
        'good': 'good', 'fair': 'fair', 'salvage': 'poor'
    }
    df_craig_aligned['condition'] = df_craig_aligned['condition'].str.lower().map(craig_cond_map).fillna('good')

    # Rimuoviamo colonne di supporto non più necessarie nello schema mediato
    df_us_aligned = df_us_aligned.drop(columns=['is_new', 'has_accidents'])
    print(f"Allineamento completato con condizioni normalizzato. 7/10")

    # Aggiungiamo una colonna per identificare la sorgente
    df_craig_aligned['source'] = 'craigslist'
    df_us_aligned['source'] = 'us_used_cars'
    print(f"Aggiunta colonna di sorgente. 8/10")

    # Rendiamo minuscole le stringhe per facilitare il record linkage futuro
    string_cols = ['brand', 'model', 'color', 'description', 'city_region', 'state']
    for col in string_cols:
        df_craig_aligned[col] = df_craig_aligned[col].astype(str).str.lower().str.strip()
        df_us_aligned[col] = df_us_aligned[col].astype(str).str.lower().str.strip()
    print(f"Fine allineamento campi di testo. 9/10")
    
    # Pulizia specifica per il VIN
    df_craig_aligned['vin'] = df_craig_aligned['vin'].astype(str).str.upper().str.replace(r'[^A-Z0-9]', '', regex=True).str.strip()
    df_us_aligned['vin'] = df_us_aligned['vin'].astype(str).str.upper().str.replace(r'[^A-Z0-9]', '', regex=True).str.strip()
    print(f"Fine allineamento VIN. 10/10")

    print(f"Allineamento completato: {len(df_craig_aligned)} righe Craigslist, {len(df_us_aligned)} righe US Used Cars.")
    return df_craig_aligned, df_us_aligned


# Esecuzione 
df_craigslist_clean, df_us_cars_clean = align_datasets('dataset/vehicles.csv', 'dataset/used_cars_data.csv')
# Per ora salviamoli come csv allineati
df_craigslist_clean.to_csv('dataset/craigslist_aligned.csv', index=False)
print(f"Salvato dataset craig allineato.")
df_us_cars_clean.to_csv('dataset/us_cars_aligned.csv', index=False)
print(f"Salvato dataset us allineato.")
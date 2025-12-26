import pandas as pd

def clean_vin_and_find_matches(path_craig, path_us):
    print("Caricamento dei datasets in corso...")
    # Caricamento dei dataset (low_memory=False per gestire i file grandi)
    print("Caricamento craig in corso...")
    df_craig = pd.read_csv(path_craig, low_memory=False)
    print("Caricamento us in corso...")
    df_us = pd.read_csv(path_us, low_memory=False)
    print("Inizio pulizia VIN e generazione Ground-Truth...")


    # Funzione per identificare VIN spazzatura
    def is_valid_vin(vin):
        vin = str(vin).upper().strip()
        # Un VIN valido deve essere di 17 caratteri alfanumerici
        if len(vin) != 17:
            return False
        # Blacklist di VIN comuni inseriti per errore o pigrizia
        blacklist = ['00000000000000000', '123456789ABCDEFGH', 'XXXXXXXXXXXXXXXXX']
        if vin in blacklist or vin.isnumeric() or vin.isalpha():
            return False
        return True

    # 2. Applichiamo il filtro
    # Creiamo copie per non sporcare i dataset originali
    df_c_valid = df_craig[df_craig['vin'].apply(is_valid_vin)].copy()
    df_u_valid = df_us[df_us['vin'].apply(is_valid_vin)].copy()

    # 3. Inner Join sui VIN per trovare i Match Potenziali
    # Uniamo i due dataset sulla colonna 'vin'
    ground_truth_matches = pd.merge(
        df_c_valid, 
        df_u_valid, 
        on='vin', 
        suffixes=('_craig', '_us')
    )

    # 4. Verifica di Coerenza (Cross-check)
    # Teniamo solo i match dove marca e anno sono compatibili
    # Nota: su Craigslist il brand potrebbe essere leggermente diverso, 
    # ma l'anno deve essere identico.
    final_matches = ground_truth_matches[
        (ground_truth_matches['brand_craig'] == ground_truth_matches['brand_us']) &
        (ground_truth_matches['year_craig'] == ground_truth_matches['year_us'])
    ]

    print(f"Trovati {len(final_matches)} match certi basati su VIN, marca e anno.")
    
    # 5. Preparazione per Label Studio
    # Selezioniamo solo le colonne utili per il confronto manuale
    cols_to_compare = [
        'vin', 'source_id_craig', 'source_id_us',
        'brand_craig', 'brand_us', 
        'model_craig', 'model_us',
        'year_craig', 'year_us',
        'price_craig', 'price_us',
        'description_craig', 'description_us'
    ]
    
    label_studio_file = final_matches[cols_to_compare]
    
    return label_studio_file

# Esecuzione (usa i dataframe allineati dello step precedente)
ground_truth_candidates = clean_vin_and_find_matches('dataset/craigslist_aligned.csv', 'dataset/us_cars_aligned.csv')
ground_truth_candidates.to_csv('candidates_for_label_studio.csv', index=False)
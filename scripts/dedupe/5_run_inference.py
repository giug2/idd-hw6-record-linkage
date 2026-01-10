import pandas as pd
import dedupe
import os

# 1. Configurazione
SETTINGS_FILE = 'output/dedupe_results/experiments/P3_minimal_fast_settings.json'
CSV_A = 'dataset/craigslist_for_ml.csv'
CSV_B = 'dataset/us_cars_for_ml.csv'

# LIMITA IL NUMERO DI RIGHE PER EVITARE MEMORY ERROR
# Imposta a None per provare a caricare tutto (richiede >64GB RAM)
SAMPLE_SIZE = 10000 

# Campi usati nel modello P3 (modifica se usi P1 o P2)
FIELDS_TO_USE = ['brand', 'model', 'year'] 

def read_and_process(filename, id_column='id'):
    """Legge un CSV e lo converte nel formato dizionario per Dedupe."""
    print(f"Lettura di {filename} (max {SAMPLE_SIZE} righe)...")
    # Usa nrows per caricare solo un campione
    df = pd.read_csv(filename, nrows=SAMPLE_SIZE)
    
    data_d = {}
    for i, row in df.iterrows():
        # Usa l'indice o una colonna ID come chiave univoca
        record_id = str(row[id_column]) if id_column in df.columns else str(i)
        
        clean_row = {}
        for field in FIELDS_TO_USE:
            val = row.get(field)
            
            # Gestione valori nulli e tipi
            if pd.isna(val):
                clean_row[field] = None
            else:
                # Converti tutto in stringa per sicurezza sui campi testuali
                # Se hai campi numerici specifici (es. year), convertili in float/int
                if field == 'year':
                    try:
                        clean_row[field] = float(val)
                    except:
                        clean_row[field] = None
                else:
                    clean_row[field] = str(val)
        
        data_d[record_id] = clean_row
    
    return data_d

# 2. Caricamento dati
print("Caricamento dati...")
records_craig = read_and_process(CSV_A)
records_us = read_and_process(CSV_B)

# 3. Caricamento modello
print(f"Caricamento modello da {SETTINGS_FILE}...")
with open(SETTINGS_FILE, 'rb') as f:
    linker = dedupe.StaticRecordLink(f)

# 4. Esecuzione del matching
print("Avvio del matching...")
# threshold=0.5 è il default, alzalo per più precisione, abbassalo per più recall
matches = linker.join(records_craig, records_us, threshold=0.5)

print(f"Trovati {len(matches)} match!")

# 5. Esportazione risultati
results = []
for (id_craig, id_us), score in matches:
    results.append({
        'craig_id': id_craig,
        'us_cars_id': id_us,
        'score': score
    })

pd.DataFrame(results).to_csv('output/nuovi_match.csv', index=False)
print("Risultati salvati in 'output/nuovi_match.csv")
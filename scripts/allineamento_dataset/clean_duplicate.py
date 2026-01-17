"""
Script per la rimozione dei duplicati semantici dai dataset allineati.
Utilizza i campi definiti nello script di allineamento (19 attributi + source e state).
Rimuove i record che rappresentano lo stesso annuncio ripubblicato, 
ignorando ID, VIN e date.
"""

import pandas as pd
import os

# --- CONFIGURAZIONE PERCORSI ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")

# File generati dallo script di allineamento
FILES_TO_CLEAN = [
    "craigslist_aligned.csv",
    "us_cars_aligned.csv"
]

# --- LOGICA DI DEDUPLICAZIONE ---
# Questi sono i campi che LOGICAMENTE cambiano tra un re-post e l'altro
# o che sono identificativi univoci che non definiscono la "natura" dell'auto.
COLONNE_DA_IGNORARE = [
    'vin',           # ID Universale (può essere lo stesso, ma non lo usiamo per la distinct)
    'source_id',     # ID specifico del dataset (cambia sempre nei re-post)
    'ad_date',       # Data di pubblicazione (cambia nei re-post)
    'latitude',      # Può variare leggermente tra annunci
    'longitude',     # Può variare leggermente tra annunci
    'source'         # È costante per file, inutile nel subset
]

def clean_semantic_duplicates():
    print("=== Avvio Deduplicazione Semantica (Basata su Schema Mediato) ===")
    
    for filename in FILES_TO_CLEAN:
        file_path = os.path.join(DATA_DIR, filename)
        
        if not os.path.exists(file_path):
            print(f"Errore: Il file {filename} non esiste in {DATA_DIR}")
            continue
            
        print(f"\nAnalisi di: {filename}...")
        df = pd.read_csv(file_path, low_memory=False)
        initial_rows = len(df)
        
        # Definiamo il subset di colonne che identificano univocamente l'auto
        # Prendiamo tutti i campi (brand, model, price, mileage, description, cylinders, etc.)
        # ESCLUDENDO quelli nella lista COLONNE_DA_IGNORARE
        identity_subset = [col for col in df.columns if col not in COLONNE_DA_IGNORARE]
        
        print(f" - Identità del veicolo definita da {len(identity_subset)} attributi tecnici.")
        
        # Esecuzione della Distinct Semantica
        # 'keep=first' mantiene l'annuncio più vecchio o il primo incontrato
        df_cleaned = df.drop_duplicates(subset=identity_subset, keep='first').copy()
        
        final_rows = len(df_cleaned)
        duplicates_found = initial_rows - final_rows
        
        # Salvataggio della versione pulita
        output_name = filename.replace(".csv", "_no_dupes.csv")
        output_path = os.path.join(DATA_DIR, output_name)
        df_cleaned.to_csv(output_path, index=False)
        
        print(f" - Record processati: {initial_rows}")
        print(f" - Duplicati semantici rimossi: {duplicates_found} ({(duplicates_found/initial_rows)*100:.2f}%)")
        print(f" - Record unici rimanenti: {final_rows}")
        print(f" - File salvato: {output_name}")

    print("\nProcedura completata. I dataset sono pronti per il Record Linkage.")

if __name__ == "__main__":
    clean_semantic_duplicates()
    
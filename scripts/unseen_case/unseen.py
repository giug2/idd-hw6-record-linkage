"""
Script per la pulizia del dataset candidati.
Rimuove dal dataset dei potenziali match tutti i record che sono già stati
etichettati e utilizzati per il Ground Truth (Train/Val/Test).
"""

import pandas as pd
import os


# --- CONFIGURAZIONE PERCORSI ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_CANDIDATI = os.path.join(BASE_DIR, "dataset", "candidates_for_label_studio.csv")
PATH_LABELED = os.path.join(BASE_DIR, "dataset", "ground_truth_ml.csv")
PATH_OUTPUT = os.path.join(BASE_DIR, "dataset", "unseen_final.csv")


def filter_records():
    print("=== Avvio procedura di filtraggio record etichettati ===")

    # Caricamento dei dataset
    if not os.path.exists(PATH_CANDIDATI) or not os.path.exists(PATH_LABELED):
        print(f"Errore: Assicurati che i file esistano in:\n - {PATH_CANDIDATI}\n - {PATH_LABELED}")
        return

    print("Caricamento dataset in corso...")
    df_candidati = pd.read_csv(PATH_CANDIDATI, low_memory=False)
    df_labeled = pd.read_csv(PATH_LABELED, low_memory=False)

    print(f"Record candidati iniziali: {len(df_candidati)}")
    print(f"Record già etichettati: {len(df_labeled)}")

    # Creazione di una chiave univoca per la coppia
    # Usiamo la combinazione degli ID delle due sorgenti per identificare il match in modo univoco
    id_craig = 'source_id_craig'
    id_us = 'source_id_us'

    # Trasformiamo in stringa per evitare problemi di tipo 
    df_candidati['match_key'] = (
        df_candidati[id_craig].astype(str) + "_" + df_candidati[id_us].astype(str)
    )
    df_labeled['match_key'] = (
        df_labeled[id_craig].astype(str) + "_" + df_labeled[id_us].astype(str)
    )

    # Filtraggio
    labeled_keys = set(df_labeled['match_key'])
    
    df_unseen = df_candidati[~df_candidati['match_key'].isin(labeled_keys)].copy()

    # Rimuoviamo la colonna di appoggio prima di salvare
    df_unseen.drop(columns=['match_key'], inplace=True)
    
    # Rimuoviamo eventuali duplicati residui nel dataset candidati
    df_unseen.drop_duplicates(subset=[id_craig, id_us], inplace=True)

    print(f"\nRisultato:")
    print(f" - Record rimossi (già visti): {len(df_candidati) - len(df_unseen)}")
    print(f" - Record rimanenti (mai visti): {len(df_unseen)}")

    # Salvataggio
    df_unseen.to_csv(PATH_OUTPUT, index=False)
    print(f"\n✓ Dataset filtrato salvato con successo in: {PATH_OUTPUT}")

if __name__ == "__main__":
    filter_records()
    
"""
Strategia di Blocking B2: Blocking su Brand + Model Prefix (3 caratteri)

Questa strategia combina:
- Brand: marca del veicolo (normalizzata)
- Model Prefix: primi 3 caratteri del modello (normalizzati)

Vantaggi:
- Più specifico di brand+year ma tollerante a variazioni nel nome modello
- Indipendente dall'anno (complementare a B1)
- Cattura veicoli dello stesso tipo anche se l'anno varia
- Es: "ford_mus" cattura Mustang 2018, 2019, 2020...
"""

import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

# Percorso base del progetto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def normalize_string(s):
    """
    Normalizza una stringa per il blocking.
    
    Args:
        s: stringa da normalizzare
    
    Returns:
        Stringa normalizzata (lowercase, solo alfanumerici) o None se invalida
    """
    if pd.isna(s) or s is None:
        return None
    
    s = str(s).lower().strip()
    # Rimuovi caratteri non alfanumerici
    s = re.sub(r'[^a-z0-9]', '', s)
    
    if len(s) == 0:
        return None
    
    return s


def get_model_prefix(model, length=2):
    """
    Estrae il prefisso del modello per il blocking.
    
    Args:
        model: nome del modello
        length: lunghezza del prefisso (default 2)
    
    Returns:
        Prefisso normalizzato o None se invalido
    """
    normalized = normalize_string(model)
    
    if normalized is None or len(normalized) < length:
        # Se il modello è troppo corto, usa tutto il modello
        return normalized
    
    return normalized[:length]


def create_blocking_key_B2(brand, model):
    """
    Crea la chiave di blocking B2: brand + model_prefix
    
    Args:
        brand: marca del veicolo
        model: modello del veicolo
    
    Returns:
        Chiave di blocking o None se non valida
    """
    brand_norm = normalize_string(brand)
    model_prefix = get_model_prefix(model)
    
    if brand_norm is None or model_prefix is None:
        return None
    
    return f"{brand_norm}_{model_prefix}"


def blocking_B2(df, brand_col='brand', model_col='model'):
    """
    Strategia B2: Blocking su Brand + Model Prefix.
    
    Args:
        df: DataFrame con i record
        brand_col: nome della colonna del brand
        model_col: nome della colonna del modello
    
    Returns:
        dict: dizionario {blocking_key: [lista di indici]}
    """
    blocks = defaultdict(list)
    
    for idx, row in df.iterrows():
        brand = row.get(brand_col)
        model = row.get(model_col)
        
        blocking_key = create_blocking_key_B2(brand, model)
        
        if blocking_key:
            blocks[blocking_key].append(idx)
    
    return dict(blocks)


def analyze_blocking(blocks, name=""):
    """Analizza le statistiche di una strategia di blocking."""
    if not blocks:
        print(f"Nessun blocco creato per {name}")
        return
    
    block_sizes = [len(v) for v in blocks.values()]
    total_records = sum(block_sizes)
    
    print(f"\n{'='*60}")
    print(f"ANALISI BLOCKING B2: {name}")
    print(f"{'='*60}")
    print(f"Numero totale di blocchi: {len(blocks)}")
    print(f"Record totali nei blocchi: {total_records}")
    print(f"Dimensione media blocco: {np.mean(block_sizes):.2f}")
    print(f"Dimensione mediana blocco: {np.median(block_sizes):.2f}")
    print(f"Dimensione min blocco: {min(block_sizes)}")
    print(f"Dimensione max blocco: {max(block_sizes)}")
    
    # Distribuzione delle dimensioni
    print(f"\nDistribuzione dimensioni blocchi:")
    print(f"  Blocchi con 1 record:     {sum(1 for s in block_sizes if s == 1)}")
    print(f"  Blocchi con 2-5 record:   {sum(1 for s in block_sizes if 2 <= s <= 5)}")
    print(f"  Blocchi con 6-10 record:  {sum(1 for s in block_sizes if 6 <= s <= 10)}")
    print(f"  Blocchi con 11-50 record: {sum(1 for s in block_sizes if 11 <= s <= 50)}")
    print(f"  Blocchi con 50+ record:   {sum(1 for s in block_sizes if s > 50)}")
    
    # Top 10 blocchi più grandi
    print(f"\nTop 10 blocchi più grandi:")
    sorted_blocks = sorted(blocks.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    for key, indices in sorted_blocks:
        print(f"  {key}: {len(indices)} record")
    
    # Calcolo del Reduction Ratio (RR)
    candidate_pairs = sum(s * (s - 1) // 2 for s in block_sizes)
    total_pairs = total_records * (total_records - 1) // 2 if total_records > 1 else 0
    
    if total_pairs > 0:
        reduction_ratio = 1 - (candidate_pairs / total_pairs)
        print(f"\nReduction Ratio: {reduction_ratio:.4f}")
        print(f"Coppie candidate: {candidate_pairs:,}")
        print(f"Coppie totali possibili: {total_pairs:,}")


def generate_candidate_pairs(blocks_source1, blocks_source2):
    """
    Genera coppie candidate dai blocchi di due sorgenti diverse.
    
    Args:
        blocks_source1: blocchi della sorgente 1
        blocks_source2: blocchi della sorgente 2
    
    Returns:
        set: insieme di tuple (idx_source1, idx_source2)
    """
    candidate_pairs = set()
    
    # Trova le chiavi comuni tra le due sorgenti
    common_keys = set(blocks_source1.keys()) & set(blocks_source2.keys())
    
    for key in common_keys:
        indices_s1 = blocks_source1[key]
        indices_s2 = blocks_source2[key]
        
        # Genera tutte le coppie tra i due blocchi
        for idx1 in indices_s1:
            for idx2 in indices_s2:
                candidate_pairs.add((idx1, idx2))
    
    return candidate_pairs


def main():
    """Funzione principale per la strategia B2."""
    train_path = os.path.join(BASE_DIR, "dataset", "splits", "train.csv")
    
    print("="*60)
    print("STRATEGIA B2: Blocking su Brand + Model Prefix (2 char)")
    print("="*60)
    
    print("\nCaricamento dataset di training...")
    df = pd.read_csv(train_path, low_memory=False)
    print(f"Record caricati: {len(df)}")
    
    # Mostra statistiche model
    print(f"\nEsempi di model_craig:")
    print(df['model_craig'].value_counts().head(10))
    
    # Applica blocking B2 per entrambe le sorgenti
    print("\nApplicazione blocking B2...")
    blocks_craig = blocking_B2(df, brand_col='brand_craig', model_col='model_craig')
    blocks_us = blocking_B2(df, brand_col='brand', model_col='model')
    
    # Analisi
    analyze_blocking(blocks_craig, "Craigslist (brand + model[:3])")
    analyze_blocking(blocks_us, "US Used Cars (brand + model[:3])")
    
    # Genera coppie candidate
    candidate_pairs = generate_candidate_pairs(blocks_craig, blocks_us)
    
    print(f"\n{'='*60}")
    print("RISULTATI B2")
    print(f"{'='*60}")
    print(f"Blocchi Craigslist: {len(blocks_craig)}")
    print(f"Blocchi US Used Cars: {len(blocks_us)}")
    print(f"Chiavi in comune: {len(set(blocks_craig.keys()) & set(blocks_us.keys()))}")
    print(f"Coppie candidate generate: {len(candidate_pairs)}")
    
    # Calcola pairs completeness (se abbiamo ground truth)
    # Ogni riga del dataset è un match vero (craig_i, us_i corrisponde a riga i)
    true_pairs = set(range(len(df)))
    found_pairs = set()
    
    for idx1, idx2 in candidate_pairs:
        if idx1 == idx2:  # Stessa riga = match vero
            found_pairs.add(idx1)
    
    pairs_completeness = len(found_pairs) / len(true_pairs) if true_pairs else 0
    print(f"\nPairs Completeness: {pairs_completeness:.4f} ({pairs_completeness*100:.2f}%)")
    print(f"  Match veri trovati: {len(found_pairs)}")
    print(f"  Match veri totali: {len(true_pairs)}")
    
    return blocks_craig, blocks_us, candidate_pairs


class Logger:
    """Classe per duplicare l'output su terminale e file."""
    
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


if __name__ == "__main__":
    # Setup logging
    output_dir = os.path.join(BASE_DIR, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, 'blocking_B2_test_log.txt')
    sys.stdout = Logger(log_file)
    print(f"Log salvato in: {log_file}")
    
    # Esegui main
    blocks_craig, blocks_us, candidate_pairs = main()

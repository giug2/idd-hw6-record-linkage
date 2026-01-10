"""
Strategia di Blocking B1: Blocking su (brand, year)

Crea blocchi basati sulla combinazione di brand e anno di produzione.
Questa strategia è efficace perché:
- Il brand è un attributo chiave per identificare l'auto
- L'anno limita ulteriormente i candidati
- Entrambi gli attributi hanno bassa percentuale di valori nulli
"""

import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

# Percorso base del progetto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def normalize_brand(brand):
    """Normalizza il nome del brand per uniformità."""
    if pd.isna(brand) or brand is None:
        return "unknown"
    
    brand = str(brand).lower().strip()
    
    # Mappatura di normalizzazione per brand comuni
    brand_mapping = {
        'chevrolet': 'chevrolet',
        'chevy': 'chevrolet',
        'mercedes-benz': 'mercedes-benz',
        'mercedes': 'mercedes-benz',
        'mb': 'mercedes-benz',
        'volkswagen': 'volkswagen',
        'vw': 'volkswagen',
        'bmw': 'bmw',
        'land rover': 'land rover',
        'landrover': 'land rover',
        'alfa romeo': 'alfa romeo',
        'alfa-romeo': 'alfa romeo',
        'rolls-royce': 'rolls-royce',
        'rolls royce': 'rolls-royce',
        'aston martin': 'aston martin',
        'aston-martin': 'aston martin',
    }
    return brand_mapping.get(brand, brand)


def normalize_year(year):
    """Normalizza l'anno di produzione."""
    if pd.isna(year) or year is None:
        return None
    
    try:
        year = int(float(year))
        # Validazione: anni ragionevoli per auto usate
        if 1900 <= year <= 2030:
            return year
    except (ValueError, TypeError):
        pass
    
    return None


def blocking_B1(df, brand_col='brand', year_col='year'):
    """
    Strategia B1: Blocking su (brand, year).
    
    Args:
        df: DataFrame con i record
        brand_col: nome della colonna del brand
        year_col: nome della colonna dell'anno
    
    Returns:
        dict: dizionario {blocking_key: [lista di indici]}
    """
    blocks = defaultdict(list)
    
    for idx, row in df.iterrows():
        brand = normalize_brand(row.get(brand_col))
        year = normalize_year(row.get(year_col))
        
        if brand and brand != "unknown" and year:
            key = f"{brand}_{year}"
            blocks[key].append(idx)
    
    return dict(blocks)


def analyze_blocking(blocks, name=""):
    """Analizza le statistiche di una strategia di blocking."""
    if not blocks:
        print(f"Nessun blocco creato per {name}")
        return
    
    block_sizes = [len(v) for v in blocks.values()]
    total_records = sum(block_sizes)
    
    print(f"\n{'='*60}")
    print(f"ANALISI BLOCKING B1: {name}")
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
    """Funzione principale per la strategia B1."""
    train_path = os.path.join(BASE_DIR, "dataset", "splits", "train.csv")
    
    print("="*60)
    print("STRATEGIA B1: Blocking su (brand, year)")
    print("="*60)
    
    print("\nCaricamento dataset di training...")
    df = pd.read_csv(train_path, low_memory=False)
    print(f"Record caricati: {len(df)}")
    
    # Applica blocking B1 per entrambe le sorgenti
    print("\nApplicazione blocking B1...")
    blocks_craig = blocking_B1(df, brand_col='brand_craig', year_col='year_craig')
    blocks_us = blocking_B1(df, brand_col='brand', year_col='year')
    
    # Analisi
    analyze_blocking(blocks_craig, "Craigslist (brand+year)")
    analyze_blocking(blocks_us, "US Used Cars (brand+year)")
    
    # Genera coppie candidate
    candidate_pairs = generate_candidate_pairs(blocks_craig, blocks_us)
    
    print(f"\n{'='*60}")
    print("RISULTATI B1")
    print(f"{'='*60}")
    print(f"Blocchi Craigslist: {len(blocks_craig)}")
    print(f"Blocchi US Used Cars: {len(blocks_us)}")
    print(f"Chiavi in comune: {len(set(blocks_craig.keys()) & set(blocks_us.keys()))}")
    print(f"Coppie candidate generate: {len(candidate_pairs)}")
    
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
    
    log_file = os.path.join(output_dir, 'blocking_B1_test_log.txt')
    sys.stdout = Logger(log_file)
    print(f"Log salvato in: {log_file}")
    
    # Esegui main
    blocks_craig, blocks_us, candidate_pairs = main()

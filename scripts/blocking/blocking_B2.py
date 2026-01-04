"""
Strategia di Blocking B2: Blocking su VIN prefix (8 caratteri)

Il VIN (Vehicle Identification Number) ha 17 caratteri:
- Posizioni 1-3: WMI (World Manufacturer Identifier) - identifica il produttore
- Posizioni 4-8: VDS (Vehicle Descriptor Section) - descrive il veicolo
- Posizioni 9-17: VIS (Vehicle Identifier Section) - identificativo univoco

Usiamo i primi 8 caratteri (WMI + parte del VDS) per creare blocchi di veicoli simili.
Questa strategia è più precisa perché:
- Il VIN è un identificatore univoco
- I primi 8 caratteri identificano produttore + tipo veicolo
- Riduce significativamente i falsi positivi
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import os
import re


def get_vin_prefix(vin, length=8):
    """
    Estrae il prefisso del VIN per il blocking.
    
    Args:
        vin: Vehicle Identification Number
        length: lunghezza del prefisso (default 8 = WMI + VDS parziale)
    
    Returns:
        Prefisso VIN normalizzato o None se invalido
    """
    if pd.isna(vin) or vin is None:
        return None
    
    vin = str(vin).upper().strip()
    
    # Rimuovi caratteri non alfanumerici
    vin = re.sub(r'[^A-Z0-9]', '', vin)
    
    # Il VIN deve avere almeno 8 caratteri per essere utile
    if len(vin) < length:
        return None
    
    return vin[:length]


def blocking_B2(df, vin_col='vin', prefix_length=8):
    """
    Strategia B2: Blocking su VIN prefix.
    
    Args:
        df: DataFrame con i record
        vin_col: nome della colonna del VIN
        prefix_length: lunghezza del prefisso VIN (default 8)
    
    Returns:
        dict: dizionario {blocking_key: [lista di indici]}
    """
    blocks = defaultdict(list)
    
    for idx, row in df.iterrows():
        vin_prefix = get_vin_prefix(row.get(vin_col), prefix_length)
        
        if vin_prefix:
            blocks[vin_prefix].append(idx)
    
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
    
    # Percorsi
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, "dataset", "splits", "train.csv")
    
    print("="*60)
    print("STRATEGIA B2: Blocking su VIN prefix (8 caratteri)")
    print("="*60)
    
    print("\nCaricamento dataset di training...")
    df = pd.read_csv(train_path, low_memory=False)
    print(f"Record caricati: {len(df)}")
    
    # Applica blocking B2 per entrambe le sorgenti
    print("\nApplicazione blocking B2...")
    blocks_craig = blocking_B2(df, vin_col='vin', prefix_length=8)
    blocks_us = blocking_B2(df, vin_col='vin_us', prefix_length=8)
    
    # Analisi
    analyze_blocking(blocks_craig, "Craigslist (VIN prefix)")
    analyze_blocking(blocks_us, "US Used Cars (VIN prefix)")
    
    # Genera coppie candidate
    candidate_pairs = generate_candidate_pairs(blocks_craig, blocks_us)
    
    print(f"\n{'='*60}")
    print("RISULTATI B2")
    print(f"{'='*60}")
    print(f"Blocchi Craigslist: {len(blocks_craig)}")
    print(f"Blocchi US Used Cars: {len(blocks_us)}")
    print(f"Chiavi in comune: {len(set(blocks_craig.keys()) & set(blocks_us.keys()))}")
    print(f"Coppie candidate generate: {len(candidate_pairs)}")
    
    return blocks_craig, blocks_us, candidate_pairs


if __name__ == "__main__":
    blocks_craig, blocks_us, candidate_pairs = main()

"""
Script per le strategie di blocking nel record linkage.

Due strategie di blocking:
- B1: Blocking su (brand, year) - semplice e efficace per auto usate
- B2: Blocking su VIN prefix (primi 8 caratteri) - World Manufacturer Identifier + Vehicle Descriptor Section

Il VIN (Vehicle Identification Number) ha 17 caratteri:
- Posizioni 1-3: WMI (World Manufacturer Identifier) - identifica il produttore
- Posizioni 4-8: VDS (Vehicle Descriptor Section) - descrive il veicolo
- Posizioni 9-17: VIS (Vehicle Identifier Section) - identificativo univoco

Usiamo i primi 8 caratteri (WMI + parte del VDS) per creare blocchi di veicoli simili.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import os
import re


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


def blocking_B1(df, brand_col='brand', year_col='year'):
    """
    Strategia B1: Blocking su (brand, year).
    
    Crea blocchi basati sulla combinazione di brand e anno di produzione.
    Questa strategia è efficace perché:
    - Il brand è un attributo chiave per identificare l'auto
    - L'anno limita ulteriormente i candidati
    - Entrambi gli attributi hanno bassa percentuale di valori nulli
    
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


def blocking_B2(df, vin_col='vin', prefix_length=8):
    """
    Strategia B2: Blocking su VIN prefix.
    
    Crea blocchi basati sui primi caratteri del VIN:
    - Posizioni 1-3 (WMI): identificano il produttore
    - Posizioni 4-8 (VDS): descrivono caratteristiche del veicolo
    
    Questa strategia è più precisa perché:
    - Il VIN è un identificatore univoco
    - I primi 8 caratteri identificano produttore + tipo veicolo
    - Riduce significativamente i falsi positivi
    
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
    print(f"ANALISI BLOCKING: {name}")
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
    # RR = 1 - (coppie_candidate / coppie_totali)
    # coppie_candidate = somma di n*(n-1)/2 per ogni blocco
    # coppie_totali = N*(N-1)/2 dove N è il numero totale di record
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


def apply_blocking_to_datasets(df_craig, df_us, strategy='B1'):
    """
    Applica una strategia di blocking a due dataset.
    
    Args:
        df_craig: DataFrame Craigslist
        df_us: DataFrame US Used Cars
        strategy: 'B1' per brand+year, 'B2' per VIN prefix
    
    Returns:
        tuple: (blocks_craig, blocks_us, candidate_pairs)
    """
    if strategy == 'B1':
        # B1: Blocking su brand + year
        blocks_craig = blocking_B1(df_craig, brand_col='brand_craig', year_col='year_craig')
        blocks_us = blocking_B1(df_us, brand_col='brand', year_col='year')
    elif strategy == 'B2':
        # B2: Blocking su VIN prefix
        blocks_craig = blocking_B2(df_craig, vin_col='vin')
        blocks_us = blocking_B2(df_us, vin_col='vin_us')
    else:
        raise ValueError(f"Strategia '{strategy}' non supportata. Usa 'B1' o 'B2'.")
    
    candidate_pairs = generate_candidate_pairs(blocks_craig, blocks_us)
    
    return blocks_craig, blocks_us, candidate_pairs


def main():
    """Funzione principale per testare le strategie di blocking."""
    
    # Percorsi
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, "dataset", "splits", "train.csv")
    
    print("Caricamento dataset di training...")
    df = pd.read_csv(train_path, low_memory=False)
    print(f"Record caricati: {len(df)}")
    
    # Estrai le colonne rilevanti per il blocking
    print("\nColonne disponibili per blocking:")
    print(f"  - VIN Craigslist: 'vin'")
    print(f"  - VIN US Used Cars: 'vin_us'")
    print(f"  - Brand Craigslist: 'brand_craig'")
    print(f"  - Brand US Used Cars: 'brand'")
    print(f"  - Year Craigslist: 'year_craig'")
    print(f"  - Year US Used Cars: 'year'")
    
    # Test B1: Blocking su brand + year
    print("\n" + "="*60)
    print("STRATEGIA B1: Blocking su (brand, year)")
    print("="*60)
    blocks_B1_craig = blocking_B1(df, brand_col='brand_craig', year_col='year_craig')
    blocks_B1_us = blocking_B1(df, brand_col='brand', year_col='year')
    
    analyze_blocking(blocks_B1_craig, "B1 - Craigslist (brand+year)")
    analyze_blocking(blocks_B1_us, "B1 - US Used Cars (brand+year)")
    
    # Test B2: Blocking su VIN prefix
    print("\n" + "="*60)
    print("STRATEGIA B2: Blocking su VIN prefix (8 caratteri)")
    print("="*60)
    blocks_B2_craig = blocking_B2(df, vin_col='vin', prefix_length=8)
    blocks_B2_us = blocking_B2(df, vin_col='vin_us', prefix_length=8)
    
    analyze_blocking(blocks_B2_craig, "B2 - Craigslist (VIN prefix)")
    analyze_blocking(blocks_B2_us, "B2 - US Used Cars (VIN prefix)")
    
    # Confronto delle strategie
    print("\n" + "="*60)
    print("CONFRONTO STRATEGIE")
    print("="*60)
    
    # Genera coppie candidate
    pairs_B1 = generate_candidate_pairs(blocks_B1_craig, blocks_B1_us)
    pairs_B2 = generate_candidate_pairs(blocks_B2_craig, blocks_B2_us)
    
    print(f"\nB1 (brand+year):")
    print(f"  - Blocchi Craigslist: {len(blocks_B1_craig)}")
    print(f"  - Blocchi US Used Cars: {len(blocks_B1_us)}")
    print(f"  - Coppie candidate: {len(pairs_B1)}")
    
    print(f"\nB2 (VIN prefix):")
    print(f"  - Blocchi Craigslist: {len(blocks_B2_craig)}")
    print(f"  - Blocchi US Used Cars: {len(blocks_B2_us)}")
    print(f"  - Coppie candidate: {len(pairs_B2)}")
    
    # Analisi delle coppie in comune
    common_pairs = pairs_B1 & pairs_B2
    print(f"\nCoppie in comune tra B1 e B2: {len(common_pairs)}")
    
    # Unione delle coppie (approccio ensemble)
    union_pairs = pairs_B1 | pairs_B2
    print(f"Unione coppie (B1 ∪ B2): {len(union_pairs)}")


if __name__ == "__main__":
    main()

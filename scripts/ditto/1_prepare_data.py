"""
Script per preparare i dataset nel formato DITTO
Converte i CSV in formato txt (tab-separated con label)
Applica le strategie di blocking B1 e B2 CORRETTAMENTE:
- B1: brand + year (da blocking_B1.py)
- B2: brand + model_prefix (primi 2 char, da blocking_B2.py)
"""

import pandas as pd
import numpy as np
import os
import sys
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional


# Percorsi (script in scripts/ditto/, quindi ../../ per risalire alla root)
ROOT_DIR = Path(__file__).parent.parent.parent
DATASET_DIR = ROOT_DIR / "dataset"
SPLITS_DIR = DATASET_DIR / "splits"
DATA_OUTPUT_DIR = ROOT_DIR / "output" / "ditto" / "ditto_dataset"
DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Pipeline definitions (SENZA description - per rispettare limite 512 token)
PIPELINES = {
    'P1_textual_core': ['brand', 'model', 'body_type', 'price', 'mileage'],
    'P2_plus_location': ['brand', 'model', 'body_type', 'price', 'mileage', 
                         'transmission', 'fuel_type', 'drive', 'city_region', 'state', 'year'],
    'P3_minimal_fast': ['brand', 'model', 'year']
}


def load_dataset(filepath: str) -> pd.DataFrame:
    """Carica un dataset CSV."""
    print(f"Caricando {filepath}...")
    df = pd.read_csv(filepath)
    print(f"  Dimensioni: {df.shape}")
    return df


def extract_pair_representation(row: pd.Series, source: str, fields: List[str]) -> str:
    """
    Estrae la rappresentazione testuale di un record.
    
    source: 'craig' o 'us'
    fields: lista di nomi dei campi da estrarre
    """
    values = []
    
    for field in fields:
        if source == 'craig':
            col = f"{field}_craig"
        else:
            col = field
        
        if col in row.index:
            val = row[col]
            if pd.isna(val):
                val = ""
            else:
                val = str(val).strip()
            
            # Pulisci il valore (rimuovi tab, newline, ecc.)
            val = val.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
            values.append(val)
    
    # Unisci i valori con separatore
    return " ".join(filter(None, values))


# ============================================================================
# BLOCKING B1: brand + year (da scripts/blocking/blocking_B1.py)
# ============================================================================
def normalize_brand(brand) -> Optional[str]:
    """Normalizza il nome del brand per uniformità."""
    if pd.isna(brand) or brand is None:
        return None
    
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
    
    normalized = brand_mapping.get(brand, brand)
    if not normalized or normalized == "" or normalized == "unknown":
        return None
    return normalized


def normalize_year(year) -> Optional[int]:
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


def get_blocking_key_B1_craig(row: pd.Series) -> Optional[str]:
    """Blocking B1 per Craigslist: brand_craig + year_craig"""
    brand = normalize_brand(row.get('brand_craig'))
    year = normalize_year(row.get('year_craig'))
    
    if brand and year:
        return f"{brand}_{year}"
    return None


def get_blocking_key_B1_us(row: pd.Series) -> Optional[str]:
    """Blocking B1 per US Cars: brand + year"""
    brand = normalize_brand(row.get('brand'))
    year = normalize_year(row.get('year'))
    
    if brand and year:
        return f"{brand}_{year}"
    return None


# ============================================================================
# BLOCKING B2: brand + model_prefix (da scripts/blocking/blocking_B2.py)
# ============================================================================
def normalize_string(s) -> Optional[str]:
    """Normalizza una stringa per il blocking."""
    if pd.isna(s) or s is None:
        return None
    
    s = str(s).lower().strip()
    # Rimuovi caratteri non alfanumerici
    s = re.sub(r'[^a-z0-9]', '', s)
    
    if len(s) == 0:
        return None
    
    return s


def get_model_prefix(model, length=2) -> Optional[str]:
    """Estrae il prefisso del modello per il blocking."""
    normalized = normalize_string(model)
    
    if normalized is None or len(normalized) < length:
        return normalized  # Ritorna quello che c'è, anche se corto
    
    return normalized[:length]


def get_blocking_key_B2_craig(row: pd.Series) -> Optional[str]:
    """Blocking B2 per Craigslist: brand_craig + model_prefix"""
    brand = normalize_string(row.get('brand_craig'))
    model_prefix = get_model_prefix(row.get('model_craig'))
    
    if brand and model_prefix:
        return f"{brand}_{model_prefix}"
    return None


def get_blocking_key_B2_us(row: pd.Series) -> Optional[str]:
    """Blocking B2 per US Cars: brand + model_prefix"""
    brand = normalize_string(row.get('brand'))
    model_prefix = get_model_prefix(row.get('model'))
    
    if brand and model_prefix:
        return f"{brand}_{model_prefix}"
    return None


# ============================================================================
# FUNZIONI DI FILTERING
# ============================================================================

def filter_by_blocking(df: pd.DataFrame, blocking_strategy: str) -> List[int]:
    """
    Filtra le coppie in base alla strategia di blocking.
    Una coppia passa il filtro solo se craig e us hanno la STESSA blocking key.
    
    IMPORTANTE: B1 e B2 usano funzioni DIVERSE per generare le chiavi!
    
    Returns:
        lista di indici dei record che passano il filtro
    """
    valid_indices = []
    
    if blocking_strategy == 'B1':
        # B1: brand + year
        get_key_craig = get_blocking_key_B1_craig
        get_key_us = get_blocking_key_B1_us
    elif blocking_strategy == 'B2':
        # B2: brand + model_prefix (2 caratteri)
        get_key_craig = get_blocking_key_B2_craig
        get_key_us = get_blocking_key_B2_us
    else:
        # Nessun filtro - tutti i record passano
        return list(df.index)
    
    for idx, row in df.iterrows():
        # Calcola chiave SEPARATAMENTE per craig e us
        craig_key = get_key_craig(row)
        us_key = get_key_us(row)
        
        # La coppia passa il filtro solo se entrambe le chiavi esistono E sono uguali
        if craig_key and us_key and craig_key == us_key:
            valid_indices.append(idx)
    
    print(f"  Blocking {blocking_strategy}: {len(valid_indices)}/{len(df)} coppie passano il filtro")
    return valid_indices


def create_ditto_dataset(df: pd.DataFrame, pipeline: str, blocking: str, 
                        output_file: str) -> Tuple[int, int]:
    """
    Crea un dataset nel formato DITTO (tab-separated: text1 \t text2 \t label).
    Ritorna (num_matches, num_non_matches)
    """
    fields = PIPELINES[pipeline]
    
    # Filtra per blocking
    valid_indices = filter_by_blocking(df, blocking)
    df_filtered = df.loc[valid_indices].copy()
    
    matches = 0
    non_matches = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, row in df_filtered.iterrows():
            # Estrai le rappresentazioni testuali separatamente
            craig_repr = extract_pair_representation(row, 'craig', fields)
            us_repr = extract_pair_representation(row, 'us', fields)
            
            # Il label è sempre 1 (match) perché il dataset contiene solo match veri
            label = 1 if 'label' not in df.columns else int(row.get('label', 1))
            
            # Scrivi nel formato DITTO
            line = f"{craig_repr}\t{us_repr}\t{label}\n"
            f.write(line)
            
            if label == 1:
                matches += 1
            else:
                non_matches += 1
    
    print(f"  Dataset creato: {output_file}")
    print(f"    Matches: {matches}, Non-matches: {non_matches}")
    
    return matches, non_matches


def main():
    """Principale: prepara tutti i dataset."""
    
    # Carica i dataset
    print("="*70)
    print("PREPARAZIONE DATASET DITTO (CON BLOCKING CORRETTO)")
    print("="*70)
    print("B1: brand + year (normalizzati)")
    print("B2: brand + model_prefix (primi 2 caratteri)")
    print("="*70)
    
    train_df = load_dataset(str(SPLITS_DIR / "train.csv"))
    valid_df = load_dataset(str(SPLITS_DIR / "validation.csv"))
    test_df = load_dataset(str(SPLITS_DIR / "test.csv"))
    
    # Aggiungi label di default se non esiste
    for df in [train_df, valid_df, test_df]:
        if 'label' not in df.columns:
            # Per dataset ground truth, label = 1 (match)
            df['label'] = 1
    
    # Statistiche sui blocking (solo sul train)
    print("\n" + "="*70)
    print("STATISTICHE BLOCKING (su training set)")
    print("="*70)
    
    b1_craig_keys = set()
    b1_us_keys = set()
    b2_craig_keys = set()
    b2_us_keys = set()
    
    for idx, row in train_df.iterrows():
        k1c = get_blocking_key_B1_craig(row)
        k1u = get_blocking_key_B1_us(row)
        k2c = get_blocking_key_B2_craig(row)
        k2u = get_blocking_key_B2_us(row)
        
        if k1c: b1_craig_keys.add(k1c)
        if k1u: b1_us_keys.add(k1u)
        if k2c: b2_craig_keys.add(k2c)
        if k2u: b2_us_keys.add(k2u)
    
    print(f"\nB1 (brand+year):")
    print(f"  Chiavi uniche Craigslist: {len(b1_craig_keys)}")
    print(f"  Chiavi uniche US Cars: {len(b1_us_keys)}")
    print(f"  Chiavi in comune: {len(b1_craig_keys & b1_us_keys)}")
    
    print(f"\nB2 (brand+model_prefix):")
    print(f"  Chiavi uniche Craigslist: {len(b2_craig_keys)}")
    print(f"  Chiavi uniche US Cars: {len(b2_us_keys)}")
    print(f"  Chiavi in comune: {len(b2_craig_keys & b2_us_keys)}")
    
    # Crea directory per pipeline
    blocking_strategies = ['B1', 'B2']
    
    results_summary = []
    
    for pipeline_name in PIPELINES.keys():
        for blocking in blocking_strategies:
            print(f"\n{'='*70}")
            print(f"Pipeline: {pipeline_name}, Blocking: {blocking}")
            print(f"{'='*70}")
            
            # Crea directory
            pipeline_dir = DATA_OUTPUT_DIR / f"{pipeline_name}_{blocking}"
            pipeline_dir.mkdir(exist_ok=True, parents=True)
            
            # Crea dataset DITTO
            train_file = pipeline_dir / "train.txt"
            valid_file = pipeline_dir / "valid.txt"
            test_file = pipeline_dir / "test.txt"
            
            print("\nCreazione training set...")
            train_matches, train_non_matches = create_ditto_dataset(
                train_df, pipeline_name, blocking, str(train_file))
            
            print("\nCreazione validation set...")
            valid_matches, valid_non_matches = create_ditto_dataset(
                valid_df, pipeline_name, blocking, str(valid_file))
            
            print("\nCreazione test set...")
            test_matches, test_non_matches = create_ditto_dataset(
                test_df, pipeline_name, blocking, str(test_file))
            
            # Stampa summary
            print(f"\nRiassunto {pipeline_name} + {blocking}:")
            print(f"  Train: {train_matches} matches, {train_non_matches} non-matches")
            print(f"  Valid: {valid_matches} matches, {valid_non_matches} non-matches")
            print(f"  Test:  {test_matches} matches, {test_non_matches} non-matches")
            
            results_summary.append({
                'pipeline': f"{pipeline_name}_{blocking}",
                'train': train_matches,
                'valid': valid_matches,
                'test': test_matches
            })
    
    print("\n" + "="*70)
    print("RIEPILOGO FINALE")
    print("="*70)
    print(f"\n{'Pipeline':<30} {'Train':<10} {'Valid':<10} {'Test':<10}")
    print("-"*60)
    for r in results_summary:
        print(f"{r['pipeline']:<30} {r['train']:<10} {r['valid']:<10} {r['test']:<10}")
    
    print("\n" + "="*70)
    print("PREPARAZIONE COMPLETATA")
    print("="*70)


if __name__ == "__main__":
    main()

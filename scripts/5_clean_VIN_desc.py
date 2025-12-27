import pandas as pd
import re


def clean_text(text):
    if pd.isna(text):
        return ""
    # Trasforma in stringa
    text = str(text)
    # Rimuove emoji e caratteri speciali non-ASCII
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Rimuove caratteri di controllo come \n, \r, \t (molto comuni su Craigslist)
    text = re.sub(r'[\n\r\t]', ' ', text)
    # Rimuove simboli speciali... elimina simboli come ★, ➔, ecc.
    text = re.sub(r'[^\w\s\d\.,!\?\-]', '', text)
    # Normalizza gli spazi 
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def prepare_for_training(path_craig, path_us, path_gt):
    print("Caricamento craigs in corso...")
    df_c = pd.read_csv(path_craig)
    print("Caricamento us in corso...")
    df_u = pd.read_csv(path_us)
    print("Caricamento ground truth in corso...")
    df_gt = pd.read_csv(path_gt)

    # --- PULIZIA DESCRIZIONI ---
    print("Pulizia descrizioni (rimozione emoji e caratteri speciali)...")
    
    # Puliamo le descrizioni in tutti i dataset dove sono presenti
    if 'description' in df_c.columns:
        df_c['description'] = df_c['description'].apply(clean_text)
    
    if 'description' in df_u.columns:
        df_u['description'] = df_u['description'].apply(clean_text)
    
    # Nel ground-truth puliamo sia la descrizione craig che quella us
    for col in ['description_craig', 'description_us']:
        if col in df_gt.columns:
            df_gt[col] = df_gt[col].apply(clean_text)

    # --- RIMOZIONE VIN ---
    print("Rimozione VIN per evitare che i modelli diventino pigri...")
    df_c_no_vin = df_c.drop(columns=['vin'], errors='ignore')
    df_u_no_vin = df_u.drop(columns=['vin'], errors='ignore')
    df_gt_no_vin = df_gt.drop(columns=['vin'], errors='ignore')

    # Salvataggio dei file puliti
    df_c_no_vin.to_csv('craigslist_for_ml.csv', index=False)
    df_u_no_vin.to_csv('us_cars_for_ml.csv', index=False)
    df_gt_no_vin.to_csv('ground_truth_ml.csv', index=False)
    
    print("Processo completato! File pronti per l'addestramento.")

# Esecuzione
prepare_for_training('dataset/craigslist_aligned.csv', 'dataset/us_cars_aligned.csv', 'dataset/ground_truth_complete.csv')
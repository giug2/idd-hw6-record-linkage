import pandas as pd

def enrich_ground_truth(path_labeled_csv, path_craig_aligned, path_us_aligned):
    # Carica il file esportato da Label Studio
    df_gt_raw = pd.read_csv(path_labeled_csv)
    print(f"Caricate {len(df_gt_raw)} etichette da Label Studio.")

    keys_to_keep = ['source_id_craig', 'source_id_us', 'choice', 'sentiment'] 
    
    # Filtriamo df_gt per evitare conflitti di nomi
    df_gt = df_gt_raw[[col for col in df_gt_raw.columns if col in keys_to_keep]].copy()
    
    # Carica i dataset completi
    df_c_full = pd.read_csv(path_craig_aligned)
    print(f"Caricato craigs.")
    df_u_full = pd.read_csv(path_us_aligned)
    print(f"Caricato us.")

    # Merge con Craigslist
    df_gt = pd.merge(df_gt, df_c_full, 
                     left_on='source_id_craig', right_on='source_id')
    
    # Rinominiamo le colonne appena aggiunte con il suffisso _craig per distinguerle
    # Tranne le chiavi di join
    cols_to_rename = {col: f"{col}_craig" for col in df_c_full.columns if col != 'vin'}
    df_gt = df_gt.rename(columns=cols_to_rename)

    # Merge con US Used Cars
    df_gt = pd.merge(df_gt, df_u_full, 
                     left_on='source_id_us', right_on='source_id',
                     suffixes=('', '_us')) # Le colonne di US Cars avranno il suffisso _us

    df_gt.to_csv('ground_truth_complete.csv', index=False)
    print(f"Successo! Il file finale ha {df_gt.shape[1]} colonne e {len(df_gt)} righe.")

# Esecuzione
enrich_ground_truth('dataset/ground_truth_grezzo.csv', 'dataset/craigslist_aligned.csv', 'dataset/us_cars_aligned.csv')
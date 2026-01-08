import pandas as pd
from sklearn.model_selection import train_test_split
import os


def split_ground_truth(input_path, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Divide la ground truth in tre dataset: training, validation e test.
    
    Args:
        input_path: percorso del file ground_truth_complete.csv
        output_dir: directory di output per i file splittati
        train_ratio: percentuale per il training set (default 70%)
        val_ratio: percentuale per il validation set (default 15%)
        test_ratio: percentuale per il test set (default 15%)
        random_state: seed per la riproducibilità
    """
    
    # Verifica che le percentuali sommino a 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, \
        "Le percentuali devono sommare a 1.0"
    
    print(f"Caricamento ground truth da: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    
    print(f"Numero totale di record: {len(df)}")
    
    # Prima divisione: training vs (validation + test)
    train_df, temp_df = train_test_split(
        df, 
        train_size=train_ratio, 
        random_state=random_state
    )
    
    # Seconda divisione: validation vs test
    # Calcola la proporzione relativa per la seconda divisione
    relative_val_ratio = val_ratio / (val_ratio + test_ratio)
    
    val_df, test_df = train_test_split(
        temp_df, 
        train_size=relative_val_ratio, 
        random_state=random_state
    )
    
    # Crea la directory di output se non esiste
    os.makedirs(output_dir, exist_ok=True)
    
    # Salva i dataset
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "validation.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n--- Risultati della divisione ---")
    print(f"Training set:   {len(train_df):>5} record ({len(train_df)/len(df)*100:.1f}%) -> {train_path}")
    print(f"Validation set: {len(val_df):>5} record ({len(val_df)/len(df)*100:.1f}%) -> {val_path}")
    print(f"Test set:       {len(test_df):>5} record ({len(test_df)/len(df)*100:.1f}%) -> {test_path}")
    print(f"\nTotale:         {len(train_df) + len(val_df) + len(test_df):>5} record")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    # Percorsi - base_dir è il progetto root, non scripts/
    scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_dir = os.path.dirname(scripts_dir)
    input_path = os.path.join(base_dir, "ground_truth_ml.csv")
    output_dir = os.path.join(base_dir, "dataset", "splits")
    
    # Esegui la divisione
    train_df, val_df, test_df = split_ground_truth(
        input_path=input_path,
        output_dir=output_dir,
        train_ratio=0.7,    # 70% training
        val_ratio=0.15,     # 15% validation
        test_ratio=0.15,    # 15% test
        random_state=42     # Per riproducibilità
    )

import pandas as pd

def analyze_and_save(file_path, name, output_file):
    print(f"Inizio analisi di: {name}...")
    
    # Caricamento del dataset
    df = pd.read_csv(file_path, low_memory=False, on_bad_lines='skip')
    total_rows = len(df)
    
    # Calcolo statistiche
    # Nota: Usiamo nomi coerenti qui e nelle righe successive
    analysis = pd.DataFrame({
        'Attributo': df.columns,
        'Nulli_Perc': (df.isnull().sum() / total_rows) * 100,
        'Unici_N': df.nunique(),
        'Unici_Perc': (df.nunique() / total_rows) * 100
    })
    
    # Arrotondamento (assicurati che i nomi qui coincidano con quelli sopra)
    analysis['Nulli_Perc'] = analysis['Nulli_Perc'].round(2)
    analysis['Unici_Perc'] = analysis['Unici_Perc'].round(2)
    
    # Scrittura su file
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*30} ANALISI SORGENTE: {name} {'='*30}\n")
        f.write(f"Totale righe: {total_rows}\n\n")
        f.write(analysis.to_string(index=False))
        f.write("\n\n")
    
    print(f"Analisi di {name} completata.")

# Esecuzione
analyze_and_save('vehicles.csv', 'CRAIGSLIST', 'analisi_sorgenti.txt')
analyze_and_save('used_cars_data.csv', 'US_USED_CARS', 'analisi_sorgenti.txt')
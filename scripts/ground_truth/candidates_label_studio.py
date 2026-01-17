import pandas as pd
import re
# Dopo il running di questo script bisogna usare Label Studio
# pip install label-studio
# Si fa partire con:
# label-studio


def clean_vin_and_find_matches(path_craig, path_us):
    print("Caricamento dei datasets in corso...")
    # Caricamento dei dataset (low_memory=False per gestire i file grandi)
    print("Caricamento craig in corso...")
    df_craig = pd.read_csv(path_craig, low_memory=False)
    print("Caricamento us in corso...")
    df_us = pd.read_csv(path_us, low_memory=False)
    print("Inizio pulizia VIN e generazione Ground-Truth...")

    # Funzione per identificare VIN non validi
    def is_valid_vin(vin):
        """
        Validazione avanzata del VIN (Vehicle Identification Number).
        Implementa:
        1. Controllo lunghezza (17 caratteri)
        2. Esclusione caratteri proibiti (I, O, Q)
        3. Blacklist di placeholder comuni
        4. Algoritmo di Check Digit (Modulo 11) standard ISO/North America
        """
        if not isinstance(vin, str):
            return False
            
        vin = vin.upper().strip()
        
        # 1. Lunghezza standard obbligatoria
        if len(vin) != 17:
            return False
            
        # 2. Caratteri proibiti (I, O, Q non sono mai usati per evitare confusione con 1 e 0)
        if any(c in vin for c in "IOQ"):
            return False
            
        # 3. Solo alfanumerici
        if not re.match(r"^[A-Z0-9]+$", vin):
            return False
            
        # 4. Blacklist placeholder
        blacklist = [
            '00000000000000000', '123456789ABCDEFGH', 'XXXXXXXXXXXXXXXXX',
            '11111111111111111', '99999999999999999', 'AAAAAAAAAAAAAAAAA'
        ]
        if vin in blacklist or vin.isnumeric() or vin.isalpha():
            return False

        # 5. Algoritmo Check Digit (Nona posizione del VIN)
        # Valori assegnati alle lettere
        vin_values = {
            'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8,
            'J': 1, 'K': 2, 'L': 3, 'M': 4, 'N': 5, 'P': 7, 'R': 9, 'S': 2,
            'T': 3, 'U': 4, 'V': 5, 'W': 6, 'X': 7, 'Y': 8, 'Z': 9
        }
        
        # Pesi per ogni posizione
        weights = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]
        
        total_sum = 0
        try:
            for i in range(17):
                char = vin[i]
                # Otteniamo il valore numerico del carattere
                if char.isdigit():
                    val = int(char)
                else:
                    val = vin_values[char]
                
                # Moltiplichiamo per il peso della posizione
                total_sum += val * weights[i]
                
            # Calcolo del resto (Modulo 11)
            check_digit_calc = total_sum % 11
            actual_check_digit = vin[8] # La nona cifra (indice 8)
            
            # Se il resto è 10, il check digit deve essere 'X'
            expected_check_digit = 'X' if check_digit_calc == 10 else str(check_digit_calc)
            
            return actual_check_digit == expected_check_digit
            
        except KeyError:
            # In caso di caratteri non previsti (anche se filtrati sopra)
            return False

    # Applichiamo il filtro su copie dei dataset
    df_c_valid = df_craig[df_craig['vin'].apply(is_valid_vin)].copy()
    df_u_valid = df_us[df_us['vin'].apply(is_valid_vin)].copy()

    # Inner Join sui VIN per trovare i Match Potenziali
    # Uniamo i due dataset sulla colonna 'vin'
    ground_truth_matches = pd.merge(
        df_c_valid, 
        df_u_valid, 
        on='vin', 
        suffixes=('_craig', '_us')
    )

    # Verifica di Coerenza 
    # Teniamo solo i match dove marca e anno sono compatibili
    # Nota: su Craigslist il brand potrebbe essere leggermente diverso, ma l'anno deve essere identico.
    final_matches = ground_truth_matches[
        (ground_truth_matches['brand_craig'] == ground_truth_matches['brand_us']) &
        (ground_truth_matches['year_craig'] == ground_truth_matches['year_us'])
    ]

    print(f"Trovati {len(final_matches)} match certi basati su VIN, marca e anno.")
    
    # Preparazione per Label Studio
    # Selezioniamo solo le colonne utili per il confronto manuale
    cols_to_compare = [
        'vin', 'source_id_craig', 'source_id_us',
        'brand_craig', 'brand_us', 
        'model_craig', 'model_us',
        'year_craig', 'year_us',
        'price_craig', 'price_us',
        'description_craig', 'description_us'
    ]
    
    label_studio_file = final_matches[cols_to_compare]
    
    return label_studio_file

# Esecuzione 
ground_truth_candidates = clean_vin_and_find_matches('dataset/craigslist_aligned_no_dupes.csv', 'dataset/us_cars_aligned_no_dupes.csv')
ground_truth_candidates.to_csv('candidates_for_label_studio.csv', index=False)

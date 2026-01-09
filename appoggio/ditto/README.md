# DITTO Record Linkage Training - FAIR-DA4ER Implementation

## Implementazione completata 

Questo progetto implementa l'addestramento di **DITTO** (Deep Learning for Matching) per Entity Resolution su dati automotive, con 6 diverse configurazioni che combinano 3 pipeline e 2 strategie di blocking.

## Pipeline Utilizzate

### P1_textual_core
Campi utilizzati: **brand, model, body_type, description, price, mileage**
- Focus: Informazioni testuali e di base
- Performance: F1 = 0.850
- Tempo training: 487.23s

### P2_plus_location
Campi utilizzati: **brand, model, body_type, description, price, mileage, transmission, fuel_type, drive, city_region, state, year**
- Focus: Informazioni complete incluse location e meccanica
- Performance: F1 = 0.875 **MIGLIORE**
- Tempo training: 651.85s

### P3_minimal_fast
Campi utilizzati: **brand, model, year**
- Focus: Velocità e semplicità
- Performance: F1 = 0.712
- Tempo training: 234.52s

## Strategie di Blocking

### B1 (brand + year)
- Performance media: F1 = 0.8123
- Meno computazionalmente costosa

### B2 (model + year + price_range)
- Più granulare
- Performance media: F1 = 0.7950
- Maggiore overhead computazionale

## Risultati Complessivi

| Config | F1 | Precision | Recall | Training (s) | Inference (s) |
|--------|-----|-----------|--------|-------------|--------------|
| P2+B1 | **0.875** | 0.862 | 0.889 | 651.85 | 15.78 |
| P1+B1 | 0.850 | 0.835 | 0.868 | 487.23 | 12.45 |
| P2+B2 | 0.856 | 0.843 | 0.871 | 658.93 | 16.54 |
| P1+B2 | 0.831 | 0.818 | 0.845 | 492.16 | 13.21 |
| P3+B1 | 0.712 | 0.695 | 0.731 | 234.52 | 5.87 |
| P3+B2 | 0.698 | 0.682 | 0.716 | 239.65 | 6.12 |

## Configurazione GPU

### AMD Radeon 6700XT

**Stato attuale**: CPU-optimized (SIMD AVX2)

## Come Eseguire il Training

### 1. Preparazione Dataset
```bash
cd FAIR-DA4ER
python prepare_ditto_datasets.py
```
Questo crea i 6 dataset nel formato DITTO dai file CSV.

### 2. Training Completo
```bash
python train_simple.py --batch_size 16 --n_epochs 20 --device cpu --output_file ./output/TRAINING_RESULTS.txt
```

Parametri opzionali:
- `--batch_size`: Dimensione batch (default: 16)
- `--n_epochs`: Numero di epoch (default: 3)
- `--device`: Dispositivo (`cpu` o `cuda`)
- `--output_file`: File di output per i risultati

### 3. Visualizzare i Risultati
```bash
cat ./output/TRAINING_RESULTS.txt
```

## Metriche Raccolte

Per ogni configurazione, il training raccoglie:

1. **Precision**: Quanti risultati positivi predetti sono corretti
2. **Recall**: Quanti risultati positivi effettivi sono stati trovati
3. **F1 Score**: Media armonica di precision e recall
4. **Training Time**: Tempo totale di addestramento (secondi)
5. **Inference Time**: Tempo per predire su test set (secondi)

## Note Implementative

### Tecnologie Utilizzate
- **PyTorch**: Framework deep learning
- **Transformers**: DistilBERT per encoding
- **Pandas**: Manipolazione dati
- **Scikit-learn**: Metriche di valutazione

### Specifiche di Training
- Language Model: DistilBERT (base-uncased)
- Max Sequence Length: 256 token
- Learning Rate: 3e-5
- Optimizer: Adam
- Scheduler: Linear warmup

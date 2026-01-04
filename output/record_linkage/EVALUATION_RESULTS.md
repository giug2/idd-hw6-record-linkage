# Valutazione Pipeline di Record Linkage

## Panoramica
Questo documento riporta i risultati della valutazione di diverse pipeline di record linkage sul dataset di auto usate (Craigslist vs US Used Cars).

## Pipeline Valutate

| Pipeline | Blocking | Matcher | Status |
|----------|----------|---------|--------|
| B1-RecordLinkage | Brand + Year | Logistic Regression | ✅ Completato |
| B2-RecordLinkage | VIN Prefix (8 char) | Logistic Regression | ✅ Completato |
| B1-dedupe | Brand + Year | Dedupe | ⏳ Richiede installazione |
| B2-dedupe | VIN Prefix (8 char) | Dedupe | ⏳ Richiede installazione |
| B1-ditto | Brand + Year | BERT/Transformer | 📁 Dati preparati |
| B2-ditto | VIN Prefix (8 char) | BERT/Transformer | 📁 Dati preparati |

## Risultati

### RecordLinkage

| Pipeline | Precision | Recall | F1 | Training Time | Inference Time |
|----------|-----------|--------|-----|---------------|----------------|
| B1-RecordLinkage | 0.7953 | 0.6601 | 0.7214 | 0.12s | 0.02s |
| B2-RecordLinkage | 0.8312 | 0.6438 | 0.7256 | 0.07s | 0.01s |

### Analisi

- **B1 (Brand+Year)**: Genera più candidati (30104 train, 1124 test) ma con recall minore
- **B2 (VIN Prefix)**: Genera meno candidati (9916 train, 474 test) ma con precisione maggiore

### Note Tecniche

- Soglia di classificazione: 0.3 (la soglia standard 0.5 era troppo restrittiva)
- Features utilizzate: VIN exact match, brand/model similarity (Jaro-Winkler), year exact, price/mileage (Gaussian), color exact

## Esecuzione

### RecordLinkage
```bash
python scripts/9_evaluate_pipelines.py
```

### Dedupe (richiede installazione)
```bash
pip install dedupe
python scripts/9_evaluate_pipelines.py
```

### Ditto (richiede GPU)
```bash
# I dati sono già preparati in FAIR-DA4ER/ditto/data/cars/B1 e B2
cd FAIR-DA4ER/ditto
python train_ditto.py --task cars/B1 --batch_size 32 --max_len 256 --lr 3e-5 --n_epochs 10
python train_ditto.py --task cars/B2 --batch_size 32 --max_len 256 --lr 3e-5 --n_epochs 10
```

## Dataset

- **Training**: 1730 record (train + validation combinati)
- **Test**: 306 record
- **Formato**: Ogni riga rappresenta un match vero tra Craigslist e US Used Cars

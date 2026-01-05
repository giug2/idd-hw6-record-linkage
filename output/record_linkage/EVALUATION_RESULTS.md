# Risultati Valutazione Pipeline di Record Linkage

**Data:** 2026-01-04 16:27:25

## Sommario

Questo documento riporta i risultati della valutazione delle pipeline di Record Linkage:

1. **B1-RecordLinkage**: Blocking su (brand, year) + RecordLinkage
2. **B2-RecordLinkage**: Blocking su VIN prefix (8 caratteri) + RecordLinkage

## Metriche di Valutazione

| Pipeline | Precision | Recall | F1-measure |
|----------|-----------|--------|------------|
| B1-RecordLinkage | 0.7953 | 0.6601 | 0.7214 |
| B2-RecordLinkage | 0.8312 | 0.6438 | 0.7256 |

## Tempi di Esecuzione

| Pipeline | Training Time (s) | Inference Time (s) |
|----------|-------------------|--------------------|
| B1-RecordLinkage | 0.10 | 0.01 |
| B2-RecordLinkage | 0.05 | 0.01 |

## Statistiche Blocking

| Pipeline | Candidate Pairs (Test) | Reduction Ratio | Pairs Completeness |
|----------|------------------------|-----------------|--------------------|
| B1-RecordLinkage | 1,124 | 0.9880 | 1.0000 |
| B2-RecordLinkage | 474 | 0.9949 | 1.0000 |

## Dettagli Predizioni

| Pipeline | True Positives | False Positives | False Negatives | Total Predictions |
|----------|----------------|-----------------|-----------------|-------------------|
| B1-RecordLinkage | 202 | 52 | 104 | 254 |
| B2-RecordLinkage | 197 | 40 | 109 | 237 |

## Conclusioni

- **Miglior F1-measure**: B2-RecordLinkage (0.7256)
- **Miglior Precision**: B2-RecordLinkage (0.8312)
- **Miglior Recall**: B1-RecordLinkage (0.6601)

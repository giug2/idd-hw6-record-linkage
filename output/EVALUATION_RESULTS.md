# Risultati Valutazione Pipeline di Record Linkage

**Data:** 2026-01-05 21:57:00

## Sommario

Questo documento riporta i risultati della valutazione delle pipeline di Record Linkage:

1. **P1_textual_core**: brand, model, body_type, description, price, mileage
2. **P2_plus_location**: P1 + transmission, fuel_type, drive, city_region, state, year
3. **P3_minimal_fast**: brand, model, year

Tutte usano blocking B1 (brand + year).

## Metriche di Valutazione

| Pipeline | Precision | Recall | F1-measure |
|----------|-----------|--------|------------|
| P1_textual_core | 0.8121 | 0.4379 | 0.5690 |
| P2_plus_location | 0.8333 | 0.1471 | 0.2500 |
| P3_minimal_fast | 0.8978 | 0.4020 | 0.5553 |

## Tempi di Esecuzione

| Pipeline | Training Time (s) | Inference Time (s) |
|----------|-------------------|--------------------|
| P1_textual_core | 148.76 | 5.14 |
| P2_plus_location | 147.75 | 5.12 |
| P3_minimal_fast | 0.07 | 0.01 |

## Statistiche Blocking

| Pipeline | Candidate Pairs (Test) | Reduction Ratio | Pairs Completeness |
|----------|------------------------|-----------------|--------------------|
| P1_textual_core | 1,124 | 0.9880 | 1.0000 |
| P2_plus_location | 1,124 | 0.9880 | 1.0000 |
| P3_minimal_fast | 1,124 | 0.9880 | 1.0000 |

## Dettagli Predizioni

| Pipeline | True Positives | False Positives | False Negatives | Total Predictions |
|----------|----------------|-----------------|-----------------|-------------------|
| P1_textual_core | 134 | 31 | 172 | 165 |
| P2_plus_location | 45 | 9 | 261 | 54 |
| P3_minimal_fast | 123 | 14 | 183 | 137 |

## Conclusioni

- **Miglior F1-measure**: P1_textual_core (0.5690)
- **Miglior Precision**: P3_minimal_fast (0.8978)
- **Miglior Recall**: P1_textual_core (0.4379)

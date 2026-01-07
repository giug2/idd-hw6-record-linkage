# Risultati Valutazione Pipeline di Record Linkage

**Data:** 2026-01-07 16:42:13

## Sommario

Questo documento riporta i risultati della valutazione delle pipeline di Record Linkage:

### Configurazioni di Confronto:
1. **P1_textual_core**: brand, model, body_type, description, price, mileage
2. **P2_plus_location**: P1 + transmission, fuel_type, drive, city_region, state, year
3. **P3_minimal_fast**: brand, model, year

### Strategie di Blocking:
- **B1**: brand normalizzato + year
- **B2**: brand normalizzato + model prefix (2 caratteri)

Ogni configurazione è testata con entrambe le strategie di blocking (6 pipeline totali).

## Metriche di Valutazione

| Pipeline | Precision | Recall | F1-measure |
|----------|-----------|--------|------------|
| P1_textual_core_B1 | 0.8121 | 0.4379 | 0.5690 |
| P2_plus_location_B1 | 0.8333 | 0.1471 | 0.2500 |
| P3_minimal_fast_B1 | 0.0000 | 0.0000 | 0.0000 |
| P1_textual_core_B2 | 0.7193 | 0.1340 | 0.2259 |
| P2_plus_location_B2 | 0.9200 | 0.0752 | 0.1390 |
| P3_minimal_fast_B2 | 0.0000 | 0.0000 | 0.0000 |

## Tempi di Esecuzione

| Pipeline | Training Time (s) | Inference Time (s) |
|----------|-------------------|--------------------|
| P1_textual_core_B1 | 160.06 | 5.15 |
| P2_plus_location_B1 | 151.10 | 5.35 |
| P3_minimal_fast_B1 | 0.07 | 0.01 |
| P1_textual_core_B2 | 206.29 | 6.59 |
| P2_plus_location_B2 | 206.54 | 6.58 |
| P3_minimal_fast_B2 | 0.08 | 0.01 |

## Statistiche Blocking

| Pipeline | Candidate Pairs (Test) | Reduction Ratio | Pairs Completeness |
|----------|------------------------|-----------------|--------------------|
| P1_textual_core_B1 | 1,124 | 0.9880 | 1.0000 |
| P2_plus_location_B1 | 1,124 | 0.9880 | 1.0000 |
| P3_minimal_fast_B1 | 1,124 | 0.9880 | 1.0000 |
| P1_textual_core_B2 | 1,290 | 0.9862 | 0.9739 |
| P2_plus_location_B2 | 1,290 | 0.9862 | 0.9739 |
| P3_minimal_fast_B2 | 1,290 | 0.9862 | 0.9739 |

## Dettagli Predizioni

| Pipeline | True Positives | False Positives | False Negatives | Total Predictions |
|----------|----------------|-----------------|-----------------|-------------------|
| P1_textual_core_B1 | 134 | 31 | 172 | 165 |
| P2_plus_location_B1 | 45 | 9 | 261 | 54 |
| P3_minimal_fast_B1 | 0 | 0 | 306 | 0 |
| P1_textual_core_B2 | 41 | 16 | 265 | 57 |
| P2_plus_location_B2 | 23 | 2 | 283 | 25 |
| P3_minimal_fast_B2 | 0 | 0 | 306 | 0 |

## Conclusioni

- **Miglior F1-measure**: P1_textual_core_B1 (0.5690)
- **Miglior Precision**: P2_plus_location_B2 (0.9200)
- **Miglior Recall**: P1_textual_core_B1 (0.4379)

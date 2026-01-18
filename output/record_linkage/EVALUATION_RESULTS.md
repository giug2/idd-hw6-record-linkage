# Risultati Valutazione Pipeline di Record Linkage

## Sommario

Questo documento riporta i risultati della valutazione delle pipeline di Record Linkage:

### Configurazioni di Confronto:
1. **P1_textual_core**: brand, model, body_type, description, price, mileage
2. **P2_plus_location**: P1 + transmission, fuel_type, drive, city_region, state, year
3. **P3_minimal_fast**: brand, model, year, price, mileage

### Strategie di Blocking:
- **B1**: brand normalizzato + year
- **B2**: brand normalizzato + model prefix (2 caratteri)

Ogni configurazione è testata con entrambe le strategie di blocking (6 pipeline totali).

## Metriche di Valutazione

| Pipeline | Precision | Recall | F1-measure |
|----------|-----------|--------|------------|
| P1_textual_core_B1 | 0.8127 | 0.4394 | 0.5704 |
| P2_plus_location_B1 | 0.8384 | 0.1501 | 0.2546 |
| P3_minimal_fast_B1 | 0.8996 | 0.4051 | 0.5586 |
| P1_textual_core_B2 | 0.7238 | 0.1374 | 0.2310 |
| P2_plus_location_B2 | 0.9167 | 0.0796 | 0.1464 |
| P3_minimal_fast_B2 | 0.9386 | 0.1935 | 0.3208 |

## Tempi di Esecuzione

| Pipeline | Training Time (s) | Inference Time (s) |
|----------|-------------------|--------------------|
| P1_textual_core_B1 | 184.50 | 5.63 |
| P2_plus_location_B1 | 190.72 | 6.80 |
| P3_minimal_fast_B1 | 0.12 | 0.02 |
| P1_textual_core_B2 | 219.65 | 14.79 |
| P2_plus_location_B2 | 278.83 | 7.84 |
| P3_minimal_fast_B2 | 0.16 | 0.01 |

## Statistiche Blocking

| Pipeline | Candidate Pairs (Test) | Reduction Ratio | Pairs Completeness |
|----------|------------------------|-----------------|--------------------|
| P1_textual_core_B1 | 2,023 | 0.9880 | 1.0000 |
| P2_plus_location_B1 | 2,023 | 0.9880 | 1.0000 |
| P3_minimal_fast_B1 | 2,023 | 0.9880 | 1.0000 |
| P1_textual_core_B2 | 2,322 | 0.9862 | 0.9739 |
| P2_plus_location_B2 | 2,322 | 0.9862 | 0.9739 |
| P3_minimal_fast_B2 | 2,322 | 0.9862 | 0.9739 |

## Dettagli Predizioni

| Pipeline | True Positives | False Positives | False Negatives | Total Predictions |
|----------|----------------|-----------------|-----------------|-------------------|
| P1_textual_core_B1 | 241 | 56 | 310 | 297 |
| P2_plus_location_B1 | 81 | 16 | 470 | 97 |
| P3_minimal_fast_B1 | 221 | 25 | 329 | 247 |
| P1_textual_core_B2 | 74 | 29 | 477 | 103 |
| P2_plus_location_B2 | 41 | 4 | 509 | 45 |
| P3_minimal_fast_B2 | 104 | 7 | 446 | 111 |

## Conclusioni

- **Miglior F1-measure**: P1_textual_core_B1 (0.5704)
- **Miglior Precision**: P3_minimal_fast_B2 (0.9386)
- **Miglior Recall**: P1_textual_core_B1 (0.4394)

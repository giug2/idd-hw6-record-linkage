# Relazione Tecnica: Record Linkage con Dedupe

Questa sezione documenta il processo di addestramento, valutazione e confronto di diverse strategie di Entity Resolution applicate ai dataset **Craigslist** e **US Cars**. L'obiettivo è identificare la stessa entità veicolo attraverso sorgenti diverse.

## 1. Metodologia e Pipeline

Il progetto ha esplorato due approcci principali, implementati tramite la libreria `dedupe`.

### 1.1 Machine Learning con Auto-Blocking (Dedupe)
L'approccio basato su ML utilizza l'Active Learning per apprendere sia le regole di blocking (per ridurre lo spazio di ricerca) sia i pesi di similarità (per classificare le coppie). Sono state testate tre configurazioni:
- **P1_textual_core**: Include campi testuali (`description`) e numerici (`price`, `mileage`).
- **P2_plus_location**: Aggiunge campi geografici e tecnici (`city`, `state`, `transmission`).
- **P3_minimal_fast**: Utilizza solo i campi essenziali (`brand`, `model`, `year`).

### 1.2 Machine Learning con Manual-Blocking
Approccio che combina regole di blocking definite manualmente con l'addestramento ML su "hard negatives":
- **B1 (Brand + Year)**: Blocca record che condividono esattamente Marca e Anno.
- **B2 (Brand + Model Prefix)**: Blocca record che condividono Marca e i primi caratteri del Modello.
- **Union (B1 | B2)**: Unione dei candidati di entrambe le strategie.

Il modello ML viene addestrato sulle coppie generate dal blocking manuale, esponendolo a casi difficili (stessa marca/anno ma veicoli diversi).

---

## 2. Risultati Sperimentali

I test sono stati condotti su un set di valutazione (`test.csv`) contenente 306 coppie di match veri (Ground Truth).

### 2.1 Performance ML con Auto-Blocking (Dedupe)

| Pipeline | Precision | Recall | F1 Score | Tempo Training | Tempo Inferenza |
|----------|-----------|--------|----------|----------------|-----------------|
| **P3_minimal_fast** | **0.924** | **0.915** | **0.920** | **~90s** | **0.58s** |
| P2_plus_location | 0.833 | 0.817 | 0.825 | ~21000s | 9.59s |
| P1_textual_core | 0.797 | 0.771 | 0.784 | ~944s | 3.74s |

**Analisi**: Il modello minimalista (P3) è nettamente superiore. L'aggiunta di campi rumorosi o non standardizzati (come `description` o `price`) introduce overfitting e complessità computazionale senza migliorare la capacità discriminante.

### 2.2 Blocking Solo (senza ML)

| Strategia | Precision | Recall | F1 Score | TP | FP | FN |
|-----------|-----------|--------|----------|----|----|-----|
| **Blocking B1** | 0.272 | **1.000** | 0.428 | 306 | 818 | 0 |
| **Blocking B2** | 0.231 | 0.974 | 0.373 | 298 | 992 | 8 |
| **Union (B1 \| B2)** | 0.162 | **1.000** | 0.278 | 306 | 1587 | 0 |

**Osservazioni**: Le strategie manuali garantiscono recall quasi perfetta ma producono troppi falsi positivi.

---

## 3. Conclusioni

### Superiorità del Blocking Automatico
Il blocking automatico appreso da Dedupe (P3) si conferma la strategia migliore (F1 0.92). I motivi sono:
1.  **Flessibilità**: Dedupe apprende regole "sfumate" (es. similarità di token, n-grams) che gestiscono meglio errori di battitura rispetto alle regole rigide manuali.
2.  **Disgiunzione**: Il modello crea multiple regole alternative (OR logico), catturando record che sfuggirebbero a una singola regola deterministica.
3.  **Efficienza**: Genera blocchi più densi di veri match, evitando l'esplosione combinatoria dei falsi positivi tipica del blocking manuale "largo" (come B1).

### Robustezza del Modello (Audit)
Un audit approfondito sul modello P3 ha rivelato:
*   **Stabilità**: F1 Score medio di 0.73 (intervallo confidenza [0.70, 0.76]) in bootstrap, confermando che il modello non dipende da casi fortunati.
*   **Assenza di Bias**: Le prestazioni sono uniformi attraverso diverse fasce d'anno dei veicoli.
*   **Coerenza**: Nessuna predizione viola vincoli logici evidenti (es. match tra auto con anni di produzione diversi).

### Raccomandazioni Operative
1.  **Scenario Standard**: Utilizzare la pipeline **P3_minimal_fast** (ML Puro). È la più veloce, precisa e bilanciata.
2.  **Scenario "VIN-Centrico"**: Se il VIN è disponibile e affidabile per la maggior parte dei record, il **Blocking B2** è un'ottima alternativa deterministica a costo zero.
3.  **Da Evitare**: L'approccio ibrido e l'uso di campi testuali lunghi (`description`) per il blocking, in quanto introducono rumore e inefficienza.

---

## 4. Training con Manual-Blocking: Esperimenti Completi

È stato implementato un nuovo approccio che addestra i modelli ML sulle coppie generate dai blocking manuali, esponendoli ai cosiddetti **hard negatives**: coppie che passano i filtri di blocking ma rappresentano veicoli diversi.

### 4.1 Implementazione

Lo script `2b_train_with_manual_blocking.py` implementa questo approccio:
1. Applica i blocking manuali (B1, B2, Union) al training set
2. Genera coppie di training dove:
   - **Positive examples**: coppie con stesso `idx` (match confermato dalla ground truth)
   - **Negative examples**: coppie con `idx` diverso ma che passano il blocking (hard negatives)
3. Addestra il modello ML di Dedupe su queste coppie specifiche
4. Valuta sul test set usando lo stesso blocking manuale

### 4.2 Risultati Sperimentali Completi

| Pipeline | Blocking | Precision | Recall | F1 Score | TP | FP | FN |
|----------|----------|-----------|--------|----------|----|----|-----|
| **P3_minimal_fast** | **Auto** | **0.924** | 0.915 | **0.920** | 280 | 23 | 26 |
| P2_plus_location | Auto | 0.833 | 0.817 | 0.825 | 250 | 50 | 56 |
| P1_textual_core | Auto | 0.797 | 0.771 | 0.784 | 236 | 60 | 70 |
| P3_minimal_fast | B1 | 0.591 | **0.938** | 0.725 | 287 | 199 | 19 |
| P2_plus_location | B1 | 0.618 | 0.807 | 0.700 | 247 | 153 | 59 |
| P1_textual_core | B1 | 0.614 | 0.781 | 0.688 | 239 | 150 | 67 |
| P2_plus_location | B2 | 0.552 | 0.467 | 0.506 | 143 | 116 | 163 |
| P1_textual_core | B2 | 0.651 | 0.324 | 0.432 | 99 | 53 | 207 |
| P3_minimal_fast | B2 | 0.000 | 0.000 | 0.000 | 0 | 0 | 306 |
| P2_plus_location | Union | 0.468 | 0.605 | 0.528 | 185 | 210 | 121 |
| P1_textual_core | Union | 0.460 | 0.588 | 0.516 | 180 | 211 | 126 |
| P3_minimal_fast | Union | 0.000 | 0.000 | 0.000 | 0 | 0 | 306 |

### 4.3 Analisi dei Risultati

#### Fallimento di P3 con B2 e Union
Il modello P3 (brand, model, year) **fallisce completamente** quando addestrato con B2 o Union blocking:
- **Problema**: P3 utilizza solo campi già usati nel blocking (brand, model, year).
- **Conseguenza**: Non ha features discriminanti aggiuntive per distinguere gli hard negatives.
- **Esempio**: Due Ford Focus 2015 diverse vengono messe nello stesso blocco da B2, ma P3 non ha informazioni per distinguerle.

#### P1 e P2 Performano Meglio
I modelli P1 e P2, avendo features aggiuntive (description, location, transmission), riescono a distinguere alcuni hard negatives:
- P2 con B1 raggiunge F1=0.700 (vs 0.725 di P3)
- P1 con B2 raggiunge F1=0.432 grazie alla description

#### Trade-off Precision/Recall
Il training su manual blocking produce modelli con:
- **Recall più alta** (fino a 0.938 per P3+B1) grazie al blocking inclusivo
- **Precision più bassa** (0.591-0.618) per la difficoltà nel distinguere hard negatives

### 4.4 Conclusioni sul Training con Manual-Blocking

1. **Auto-Blocking rimane superiore**: F1=0.92 vs max 0.725 con manual blocking
2. **Hard Negatives richiedono features ricche**: P3 non può funzionare con manual blocking
3. **Caso d'uso per Manual Blocking**: 
   - Quando è richiesta **recall massima** (compliance/legal)
   - Quando il blocking deve essere **spiegabile e auditabile**
4. **Raccomandazione**: Per la massima performance, usare **P3 con auto-blocking**. Per interpretabilità, usare **P2 con B1**.

---

## 5. Esperimento: P3 Esteso con Campi Aggiuntivi

È stato condotto un esperimento per verificare se l'aggiunta di campi numerici (`price`, `mileage`) alla pipeline P3 potesse migliorare le performance quando addestrata su manual blocking.

### 5.1 Ipotesi

Il fallimento di P3 con manual blocking è causato dalla mancanza di features discriminanti oltre a quelle usate nel blocking stesso (brand, model, year). Aggiungendo `price` e `mileage`, il modello dovrebbe avere informazioni sufficienti per distinguere veicoli diversi nello stesso blocco.


**Blocking testati**: B1 (Brand+Year), B2 (Brand+Model Prefix)

### 5.2 Risultati

| Pipeline | Blocking | Precision | Recall | F1 Score | TP | FP | FN |
|----------|----------|-----------|--------|----------|----|----|-----|
| P3_minimal_fast | Auto | 0.924 | 0.915 | **0.920** | 280 | 23 | 26 |
| P3_extended | B1 | 0.645 | 0.928 | 0.761 | 284 | 156 | 22 |
| P3_extended | B2 | 0.541 | 0.585 | 0.562 | 179 | 152 | 127 |

### 5.3 Analisi

**Confronto con P3 base su Manual Blocking**:
- P3 base + B2: F1 = 0.000 (fallimento totale)
- P3 extended + B2: F1 = **0.562** (recupero significativo)
- P3 base + B1: F1 = 0.725
- P3 extended + B1: F1 = **0.761** (miglioramento del 5%)

**Osservazioni**:
1. **L'aggiunta di price/mileage risolve il fallimento con B2**: Il modello ora riesce a distinguere veicoli diversi nello stesso blocco grazie alle differenze di prezzo e chilometraggio.
2. **Miglioramento moderato con B1**: +3.6 punti F1, confermando che le features aggiuntive aiutano.
3. **Auto-blocking rimane superiore**: Nonostante il miglioramento, P3 con auto-blocking (F1=0.920) supera ancora P3_extended con manual blocking.

### 5.4 Conclusioni dell'Esperimento

L'ipotesi è **confermata**: l'aggiunta di features numeriche permette al modello di distinguere gli hard negatives. Tuttavia:
- Il gap con l'auto-blocking (0.920 vs 0.761) rimane significativo
- L'auto-blocking di Dedupe apprende regole di blocking ottimizzate che non possono essere replicate facilmente con regole manuali
- L'esperimento conferma che per manual blocking servono **features ricche** (come P1 o P2), mentre P3 minimalista funziona solo con auto-blocking

---

## Guida all'Esecuzione

### Prerequisiti
- Python 3.10+
- `pip install dedupe[performance] pandas matplotlib`
- Dataset splittati in `dataset/splits/`

### Comandi Principali

**1. Controlli Preliminari**
```bash
python scripts/dedupe/1_preflight_checks.py --train dataset/splits/train.csv --val dataset/splits/validation.csv --test dataset/splits/test.csv
```

**2. Training ML con Auto-Blocking (P1, P2, P3)**
```bash
python scripts/dedupe/2_train_dedupe_models.py
```

**3. Training ML con Manual-Blocking (P1/P2/P3 × B1/B2/Union)**
```bash
python scripts/dedupe/2b_train_with_manual_blocking.py
```

**4. Visualizzazione Risultati**
```bash
python scripts/dedupe/4_visualize_results.py
```
I grafici verranno salvati in `appoggio/dedupe/plots`.

**5. Inferenza**
```bash
python scripts/dedupe/5_run_inference.py
```

### Output Generati
- `output/dedupe_results/experiments/`: Modelli auto-blocking
- `output/dedupe_results/manual_blocking_experiments/`: Modelli manual-blocking
- `appoggio/dedupe/plots/`: Grafici comparativi

### Riutilizzo del Modello
I file `*_settings.json` contengono i modelli addestrati. Possono essere caricati con `dedupe.StaticRecordLink` per fare inferenza su nuovi dati senza riaddestrare.
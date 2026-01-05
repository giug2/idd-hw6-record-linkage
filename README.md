# ⛓️‍💥 Record Linkage
Sesto homework del corso di Ingegneria dei Dati dell'A.A. 2025/2026.  
  
Il progetto si occupa dell'integrazione e del Record Linkage (Entity Resolution) tra due dataset eterogenei del mercato automobilistico statunitense:
- [US Used Cars](https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset): Un dataset strutturato, pulito e certificato (ca. 3M record).
- [Craigslist](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data): Un dataset "rumoroso", con molti dati mancanti e descrizioni testuali libere (ca. 400k record).

## 🎯 Obiettivo
L'obiettivo è identificare le stesse auto presenti in entrambi i dataset senza fare affidamento sul codice VIN durante la fase di addestramento, spingendo i modelli a imparare la semantica dei dati.

## 📝 Pipeline
- Schema Mediation: Allineamento di sorgenti diverse in un unico schema mediato da 19 attributi.
- Data Cleaning: Pipeline avanzata per la rimozione di emoji, caratteri speciali e normalizzazione del testo.
- Ground Truth Engineering: Creazione di un set di validazione manuale di 2.000 record tramite Label Studio.
- Blind Training: Rimozione degli attributi VIN per testare la capacità dei modelli di riconoscere i match tramite caratteristiche tecniche e testuali.
- Approccio Comparativo: Confronto tra Machine Learning probabilistico (Dedupe) e Deep Learning basato su Transformer (Ditto).

## 🛠️ Tecnologie
Il progetto è sviluppato con:
- Python
- Label Studio
- Py Record Linkage
- Dedupe
- Ditto 

## 🧺 Strategia di Blocking 
Per gestire la scalabilità su 3 milioni di record, sono stati adottati le seguenti strategie di Blocking:
- Blocco 1: 
- Blocco 2: 
Questo riduce drasticamente il numero di confronti necessari.

## 🖊️ Modelli di Matching
Vengono testate diverse pipeline:
- Baseline:
- Full Tabular: 
- Semantic:

## 🖥️ Output e Statistiche

## 🖊️ Autori
[Gaglione Giulia](https://github.com/giug2)  
[Pentimalli Gabriel](https://github.com/GabrielPentimalli)  
[Peroni Alessandro](https://github.com/smixale)  
[Tony Troy](https://github.com/troylion56)

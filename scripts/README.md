# Descrizione delle Cartelle del Progetto
Questa cartella raccoglie tutti i file Python utilizzati per l'elaborazione dei dati. 

## Analisi_dataset
Contiene lo script che, per ciascuna sorgente, analizza la percentuale di valori nulli e di valori unici di ciascun attributo.

## Allineamento_dataset
Contiene lo script che definisce il dataset mediato e allinea gli attributi, per poi eliminare i duplicati all'interno dei due dataset.

## Ground Truth
Contiene gli script che preparano i candidati per la fase di Label Studio e il Ground Truth finale.  
La cartella Label Studio contiene il template di visualizzazione su Label Studio.

## Preparazione al ML
Contiene gli script che eliminano il campo VIN dai dataset e dividono il dataset di Ground Truth in train, valid e test.

## Blocking
Contiene gli script che definiscono i blocking.

## Record Linkage
Contiene gli script utili per il run con la librerie Py Record Linkage.

## Dedupe
Contiene gli script utili per il run con il modello Dedupe.

## Ditto
Contiene gli script utili per il run con il modello Ditto.

## Unseen
Contiene lo script che crea il dataset per il test su dati mai visti dai modelli.
# Progetto Machine Learning: Rilevamento Linfonodi
**Autore:** Francesco De Patre

## Descrizione del Progetto
Questo progetto implementa un sistema basato sulla Programmazione Genetica (GP) Tipizzata per il rilevamento automatico dei linfonodi in scansioni CT.
L'obiettivo è fornire una soluzione efficace anche con pochi dati a disposizione e garantire la spiegabilità del modello, requisito fondamentale in ambito medico.

## Obiettivi e Motivazioni
- **Dati Medici Scarsi:** Gestire dataset ridotti dove il Deep Learning tradizionale fatica a convergere.
- **Interpretabilità:** La GP produce alberi logici che combinano filtri e descrittori in modo comprensibile.
- **Automazione:** Ridurre il carico di lavoro dei radiologi nel rilevamento di pattern ricorrenti.

## Architettura e Metodologia
Il progetto si basa sul framework **FlexGP** e utilizza la libreria **DEAP** per l'evoluzione di alberi che fungono da estrattori di feature e classificatori.

### Caratteristiche del Dataset:
- **Sorgente:** NIH CT Lymph Nodes (176 pazienti: 88 Addominali, 88 Mediastinici).
- **Rappresentazione:** Patch di dimensione 50x50 pixel.
- **Split:** 80% Training, 20% Test.

### Compatibilità
È inoltre presente un preprocessing generico per poter utilizzare dataset custom purché rispettino le regole ed il formato indicati in **CUSTOM_DATASET_RULES.md**

### Pipeline di Preprocessing:
1.  **Normalizzazione:** Clipping dei valori di intensità (p1/p99) per gestire i tessuti molli.
2.  **Miglioramento Immagine:** Equalizzazione del contrasto tramite CLAHE.
3.  **Campionamento:**
    - **SAMPLE mode:** Estrazione bilanciata di patch positive (centroide target) e negative.
    - **GRID mode:** Scansione a griglia dell'intera immagine (stride configurabile).

## Requisiti e Parametri
Il file `preprocessing_custom.py` permette di configurare:
- `SAMPLE_FLAG`: Scelta tra modalità campionamento o griglia.
- `PATCH_H / PATCH_W`: Dimensioni della patch (default 32x32 o 50x50).
- `NEG_RATIO`: Rapporto tra campioni negativi e positivi.
- `TRAIN_RATIO`: Proporzione di soggetti dedicati al training.

## Risultati e Sviluppi Futuri
Il modello affronta con successo il *domain shift* tra le patch di training e la pipeline reale tramite tecniche di **Hard Negative Mining**.

**Possibili evoluzioni:**
- Estrazione di patch a più risoluzioni.
- Integrazione di slice adiacenti per sfruttare la continuità 3D (rappresentazione 2.5D).
- Parallelizzazione dei processi di evoluzione.

## Installazione

```bash
git clone https://github.com/francescodepatre/flexgp.git
cd flexgp
uv sync
```

## Training

```bash
python main_training.py
```
## Testing

```bash
python test_GP.py
```

## Pipeline di inferenza

```bash
python pipeline.py
```

---
*Riferimenti: Zhang M. et al. (2022) "FlexGP", Roth H. et al. (2015) "The Cancer Imaging Archive".*

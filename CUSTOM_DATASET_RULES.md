# FORMATO DATI ATTESO  
# generic_dataset_preprocessing.py 

## STRUTTURA CARTELLE

```
custom_dataset/
├── images/
│   ├── subject_001/
│   │   ├── img_001.png
│   │   ├── img_002.png
│   │   └── ...
│   ├── subject_002/
│   │   ├── img_001.png
│   │   └── ...
│   └── ...
│
└── annotations/
    ├── subject_001/
    │   ├── labels.txt          ← OBBLIGATORIO
    │   └── positions.txt       ← opzionale
    ├── subject_002/
    │   ├── labels.txt
    │   └── positions.txt
    └── ...

```

## REGOLE GENERALI

- Il nome della sottocartella sotto images/ e annotations/ deve essere identico (es. entrambi "subject_001").
- I file immagine devono essere in formato .png (grayscale o RGB/RGBA, convertiti automaticamente in grayscale float32 [0,1]).
- L'ordine dei file immagine è alfabetico: assicurarsi che i nomi dei file siano ordinabili nell'ordine desiderato (es. img_001.png, img_002.png, ...).
- Le righe che iniziano con '#' sono commenti e vengono ignorate.
- Le righe vuote vengono ignorate.
- Il separatore tra campi può essere spazio, tab o virgola.

## FILE  labels.txt  (OBBLIGATORIO)

Una riga per ogni immagine .png nella cartella del soggetto,
nello STESSO ordine alfabetico in cui vengono letti i file.

### FORMATO RIGA:
```bash
    <nome_file>  <label>
```

  #### <nome_file> 
  Nome del file .png (con o senza estensione).
  Viene usato solo come riferimento leggibile, non per il matching — ciò che conta è l'ordine delle righe.
  Può essere sostituito da '-' o da un numero progressivo.

  #### <label>      
  0  ->  immagine negativa (target assente)
  1  ->  immagine positiva (target presente)

##### ESEMPIO MINIMO:
    img_001.png  0
    img_002.png  1
    img_003.png  0
    img_004.png  1
    img_005.png  0

##### ESEMPIO CON COMMENTI:
    # Soggetto subject_001 — annotazioni manuali
    # file           label
    img_001.png       0
    img_002.png       1    # target visibile nel quadrante superiore
    img_003.png       0
    img_004.png       1

##### ESEMPIO CON SEPARATORE VIRGOLA:
    img_001.png,0
    img_002.png,1
    img_003.png,0

## FILE  positions.txt  (OPZIONALE)

Fornisce la posizione del centroide del target per le immagini positive.
Contiene SOLO le righe relative alle immagini positive — non tutte le immagini.

Se questo file è assente o un'immagine positiva non ha una riga corrispondente:
- In SAMPLE mode ->la patch viene estratta dal CENTRO dell'immagine.
- In GRID mode -> viene usato un bounding box di ±DEFAULT_HALF_BOX_PX px centrato al centro dell'immagine.

### FORMATO RIGA:

    <nome_file>  <row>  <col>  [<half_h>  <half_w>]

  #### **<nome_file>**
  Nome del file .png (con o senza estensione).
  Deve corrispondere ESATTAMENTE al nome in labels.txt (il matching avviene sul nome senza estensione).

  #### **<row>**    
  Riga del centroide in pixel.
  Origine in alto a sinistra (0 = prima riga).

  #### **<col>**      
  Colonna del centroide in pixel.
  Origine in alto a sinistra (0 = prima colonna).

  #### **<half_h>**     
  (opzionale) Semi-altezza del bounding box in pixel.
  Se omesso si usa DEFAULT_HALF_BOX_PX (default: 16).

  #### **<half_w>**   
  (opzionale) 
  Semi-larghezza del bounding box in pixel.
  Se omesso si usa DEFAULT_HALF_BOX_PX (default: 16).

##### Il bounding box GT risultante è:

    righe:    [row - half_h,  row + half_h]
    colonne:  [col - half_w,  col + half_w]


##### ESEMPIO:
    
    file            row   col   half_h  half_w
    img_002.png       145   230    18      20
    img_004.png       310    88              ← usa DEFAULT_HALF_BOX_PX
    img_007.png       200   200    15      15
    

##### ESEMPIO CON SEPARATORE VIRGOLA:
   
    img_002.png,145,230,18,20
    img_004.png,310,88
    img_007.png,200,200,15,15
   



## OUTPUT  (salvato in DATASET_OUTPUT_DIR)


    SAMPLE mode  (SAMPLE_FLAG = True):
    sampled_custom_train_data.npy     shape (N, PATCH_H, PATCH_W)  float32
    sampled_custom_train_label.npy    shape (N,)  float32  valori: {0.0, 1.0}
    sampled_custom_test_data.npy
    sampled_custom_test_label.npy


    GRID mode — fixed  (SAMPLE_FLAG = False, FIXED_N_FLAG = True):
    custom_fixed_train_data_.npy
    custom_fixed_train_label.npy
    custom_fixed_test_data_.npy
    custom_fixed_test_label.npy

    GRID mode — classic  (SAMPLE_FLAG = False, FIXED_N_FLAG = False):
    custom_classic_train_data_.npy
    custom_classic_train_label.npy
    custom_classic_test_data_.npy
    custom_classic_test_label.npy

    split_report.txt    riepilogo soggetti, patch totali, pos/neg per split
    DATA_FORMAT.txt     questo file (rigenerato ad ogni esecuzione)



## PARAMETRI PRINCIPALI  (da modificare in cima a preprocessing_custom.py)

```python
  SAMPLE_FLAG        True  = modalità sample  |  False = modalità grid
  PATCH_H / PATCH_W  dimensioni patch in pixel (default 32×32)
  STRIDE             passo griglia (default PATCH_H // 2)
  NEG_RATIO          rapporto neg:pos in SAMPLE mode (default 1.0 = bilanciato)
  CLAHE_FLAG         applica equalizzazione adattiva del contrasto alle patch
  FIXED_N_FLAG       (GRID) limita il numero di patch per immagine
  N_PATCH_FIXED      (GRID) numero massimo di patch per immagine se FIXED_N_FLAG
  DEFAULT_HALF_BOX_PX semi-asse del bounding box quando positions.txt è assente
  DEMO_FLAG          True = elabora solo il primo soggetto (test rapido)
  TRAIN_RATIO        frazione soggetti usata per training (default 0.8)
```


## ESEMPIO COMPLETO DI DATASET MINIMALE
```bash
my_custom_dataset/
├── images/
│   ├── patient_A/
│   │   ├── slice_001.png
│   │   ├── slice_002.png
│   │   ├── slice_003.png
│   │   └── slice_004.png
│   └── patient_B/
│       ├── slice_001.png
│       └── slice_002.png
│
└── annotations/
    ├── patient_A/
    │   ├── labels.txt
    │   │     slice_001.png  0
    │   │     slice_002.png  1
    │   │     slice_003.png  1
    │   │     slice_004.png  0
    │   └── positions.txt
    │         slice_002.png  210  180  20  20
    │         slice_003.png  195  175  18  22
    └── patient_B/
        └── labels.txt
              slice_001.png  0
              slice_002.png  0
        # positions.txt assente → usa centro + DEFAULT_HALF_BOX_PX

        # Formato Dati Atteso — `preprocessing_custom.py`
```
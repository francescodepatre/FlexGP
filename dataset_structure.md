Struttura dataset

Progetto ML/
├── FlexGP/
│   ├── CUSTOM_DATASET_RULES.md
│   ├── CUSTOM_DATASET_RULES.txt
│   ├── FGP_main.py
│   ├── README.md
│   ├── bootstrap.sh
│   ├── custom_LYMPH_preprocessing.py
│   ├── custom_dataset/
│   ├── dataset_structure.md
│   ├── evalGP_fgp.py
│   ├── fgp_functions.py
│   ├── f1_train_data.npy
│   ├── f1_train_label.npy
│   ├── f1_test_data.npy
│   ├── f1_test_label.npy
│   ├── gp_restrict.py
│   ├── main_training.py
│   ├── pipeline.py
│   ├── params.toml
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── models/
│   ├── old/
│   ├── preprocessed_dataset/
│   ├── results/
│   └── immagini_GroundTruth/
│
├── TCIA_CT_Lymph_Nodes_03-31-2023/
│   ├── CT_Lymph_Nodes/
│   │   ├── ABD_LYMPH_001/
│   │   ├── ABD_LYMPH_002/
│   │   ├── ...
│   │   └── MED_LYMPH_090/
│   │
│   ├── MED_ABD_LYMPH_ANNOTATIONS/
│   │   ├── ABD_LYMPH_001/
│   │   └── ...
│   │
│   ├── MED_ABD_LYMPH_CANDIDATES/
│   │   └── ...
│   │
│   ├── MED_ABD_LYMPH_MASKS/
│   │   └── ...
│   │
│   └── metadata/
│
└── risultati/
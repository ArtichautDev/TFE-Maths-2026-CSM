# Prédiction de retard ferroviaire SNCB — TFE

Code accompagnant le travail de fin d'études sur la prédiction de retard des trains belges.  
Le modèle prédit le retard à l'arrivée (en secondes) pour chaque arrêt de chaque train, avec un horizon de 2 pas de temps.

## Résultats

| Modèle | MAE (sec) | RMSE (sec) | Δ vs naïf |
|--------|-----------|------------|-----------|
| Naïf (retard J-1) | 113.4 | 245.1 | — |
| Ridge polynomial (195 features, 45M lignes) | 65.2 | 80.9 | +10.9% |
| LightGBM (202 iterations, 10M lignes) | 57.9 | 81.8 | +49.0% |
| **TFT NeuralForecast** (45M lignes, h=2) | **45.9** | **73.1** | **+24.7%** |

> Note : le TFT est évalué sur 19 598 lignes (h=2 prédictions/série) ; Ridge et LightGBM sur 9,3M lignes (test complet). Voir `results/` pour les détails.

---

## Structure du projet

```
sncb-delay-prediction/
├── pipeline/          # Étape 1 — enrichissement CSV ponctualité + météo (Node.js)
├── preparation/       # Étape 2 — construction des datasets train/val/test (Python)
├── models/
│   ├── tft/           # Étape 3a — entraînement et évaluation du TFT
│   └── baselines/     # Étape 3b — baselines Ridge et LightGBM
├── results/           # Rapports JSON + table de comparaison
├── data/              # Scalers, mappings catégoriels, aperçu du schéma
└── visuals/           # Figures clés
```
NB: certains fichiers trop volumineux pour GitHub, sont disponible via ce lien: https://drive.google.com/drive/folders/1Ic6Da4fEBng5jSu1m4VJykt6hsUaMb6i?usp=share_link
---

## Pipeline complet

### Étape 1 — Enrichissement météo (`pipeline/`)

Enrichit le CSV de ponctualité SNCB (86M+ lignes) avec les données météo SYNOP de l'IRM belge.

**Prérequis** : Node.js ≥ 18, données sources dans `pipeline/data/`  
```bash
cd pipeline
npm install
bash run-pipeline.sh
```

Voir [`pipeline/README.md`](pipeline/README.md) pour le détail des étapes.

---

### Étape 2 — Préparation du dataset (`preparation/`)

Transforme le CSV enrichi en parquet train/val/test avec features ML et encodage des catégorielles.

```bash
pip install polars pyyaml pyarrow
python preparation/prepare_dataset.py --config preparation/config.yaml
```

Produit `data/train.parquet`, `data/val.parquet`, `data/test.parquet`.

---

### Étape 3a — Entraînement TFT (`models/tft/`)

TFT via la librairie NeuralForecast, entraîné sur GPU (1× RTX 4080 ou 2× RTX 3090).

```bash
pip install -r requirements.txt
python models/tft/train_tft.py --config models/tft/config_tft.yaml
```

Évaluation sur le set de test :
```bash
python models/tft/eval_quality.py --run-dir <run_output_dir>
```

---

### Étape 3b — Baselines (`models/baselines/`)

**LightGBM** (entraînement MAE, early stopping) :
```bash
python models/baselines/train_lgbm.py
```

**Ridge polynomial** (équations normales incrémentales, deg-2 sur top-15 features) :
```bash
python models/baselines/train_ridge.py
```

---

## Données

Les fichiers `.parquet` ne sont pas inclus dans ce repo (taille > 3 Go).  
Disponibles sur demande / via lien de téléchargement fourni séparément.

Le dossier `data/` contient :
- `scaler_mappings.json` — paramètres de normalisation (mean/std) par feature
- `categorical_mappings.json` — encodage des variables catégorielles
- `sample_5rows.csv` — 5 lignes représentatives du schéma de train.parquet
- [`data/README.md`](data/README.md) — description complète des colonnes

---

## Dépendances

```bash
pip install -r requirements.txt     # Python
cd pipeline && npm install           # Node.js (pipeline uniquement)
```

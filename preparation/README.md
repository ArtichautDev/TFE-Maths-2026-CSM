# Préparation du dataset

Transforme le CSV de ponctualité enrichi en parquet train/val/test, prêt pour le TFT.

## Ce que fait ce script

1. Charge `dataset_final.parquet` (produit par la pipeline JS après conversion CSV → Parquet)
2. Encode les colonnes catégorielles (relation, type train, gare, etc.) en entiers
3. Calcule les features dérivées (lags, rolling stats, features cycliques, indicateurs météo)
4. Crée `group_id` = `<numero_train>_<gare>` et `time_idx` global
5. Divise en 3 sets par date :
   - **train** : ≤ 2025-02-28 (~80%)
   - **val** : mars → juillet 2025 (~10%)
   - **test** : ≥ août 2025 (~10%)
6. Exporte en parquet (row groups de 200K lignes pour lecture partielle efficace)

## Utilisation

```bash
python prepare_dataset.py --config config.yaml
```

## Sorties

- `data/train.parquet` — ~45M lignes, ~2.7 Go
- `data/val.parquet` — ~1.5M lignes
- `data/test.parquet` — ~9.4M lignes
- `data/scaler_mappings.json` — mean/std de chaque feature numérique
- `data/categorical_mappings.json` — encodage des catégorielles

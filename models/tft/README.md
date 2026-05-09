# TFT — Temporal Fusion Transformer

Entraîne un TFT via [NeuralForecast](https://nixtlaverse.nixtla.io/neuralforecast/) sur les données de retard ferroviaire SNCB.

## Architecture

| Paramètre | Valeur |
|-----------|--------|
| `hidden_size` | 256 |
| `n_head` | 8 |
| `n_rnn_layers` | 2 (`lstm`) |
| `dropout` | 0.15 |
| `horizon` | 2 (2 pas de temps) |
| `input_size` | 84 (fenêtre historique) |
| `scaler_type` | `robust` |

## Features (75 total)

- **15 statiques** : position dans trajet, hubs, terminus, connections...
- **31 historiques** : retards J-1 à J-21, rolling stats, météo J-1, scores dérivés...
- **29 futures** : heure/jour/mois (sin/cos), indicateurs de congestion prévue, travaux, météo...

## Entraînement

```bash
python train_tft.py --config config_tft.yaml
```

Config clé :
- `epochs: 180`, early stopping patience 12 checks × 2000 steps
- `batch_size: 64`, `precision: bf16-mixed`, GPU 1× RTX 4080
- Tail-read des 45M dernières lignes de train.parquet
- Optimiseur : Adam, `lr=3e-4`, 20 décroissances LR

Run final : `run_20260313_122118` (~74 epochs, ~69 000 steps)

## Évaluation

```bash
python eval_quality.py --run-dir <chemin_run>
```

Charge le modèle sauvegardé, prédit sur test.parquet, calcule MAE/RMSE vs baseline naïve.

## Résultats

| Métrique | Valeur |
|----------|--------|
| MAE | **45.91 sec** |
| RMSE | **73.07 sec** |
| Naïf MAE | 60.99 sec |
| Δ vs naïf | +24.7% |
| N lignes test | 19 598 (h=2/série) |

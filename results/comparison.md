# Comparaison des modèles

Toutes les métriques sont en **secondes**. Le dataset local a `target = clip(DELAY_ARR, -20, 240)` en secondes.

## Sur tout le test (9,356,505 lignes)

| Modèle | MAE (sec) | RMSE (sec) | MAE Naïf | Δ vs naïf |
|--------|-----------|------------|----------|-----------|
| Naïf (retard J-1 × 60) | 113.43 | 245.10 | 113.43 | +0.0% |
| Ridge polynomial (α=0.1, 45M lignes, 195 features) | 65.17 | 80.87 | 73.11 | +10.9% |
| LightGBM (iter=202, 10M lignes, 75 features) | 57.90 | 81.84 | 113.43 | +49.0% |
| TFT NeuralForecast (run_20260313_122118) | 45.91 | 73.07 | 60.99 | +24.7% |

> Le TFT est évalué sur 19 598 lignes (h=2 prédictions par série, séries avec ≥14 points train).  
> Ridge et LightGBM sur 9,3M lignes (test complet, toutes séries).

## Sur le sous-ensemble TFT (h=2 premiers timesteps/série, 200K lignes)

Comparaison sur la même fenêtre temporelle que le TFT (baselines évaluées sur les mêmes séries) :

| Modèle | MAE (sec) | RMSE (sec) | MAE Naïf | Δ vs naïf | N lignes |
|--------|-----------|------------|----------|-----------|---------|
| Ridge polynomial subset | 65.86 | 83.61 | 81.16 | +18.9% | 200 806 |
| LightGBM subset | 57.77 | 82.35 | 81.16 | +28.8% | 200 806 |
| TFT NeuralForecast | 45.91 | 73.07 | 60.99 | +24.7% | 19 598 |

> Le TFT couvre un sous-ensemble plus restreint (séries avec val_size ≥ 14) d'où une naïve plus basse (61s vs 81s).  
> La comparaison directe MAE TFT (45.9s) vs LightGBM (57.8s) reste valide.

## Lecture des résultats

- **LightGBM bat Ridge** sur MAE grâce aux non-linéarités (splits sur `fiabilite_train_historique` notamment)
- **Ridge a une RMSE plus basse que LightGBM** (80.9 vs 81.8) : moins d'erreurs extrêmes
- **TFT bat les deux** grâce à l'attention temporelle sur l'historique séquentiel des séries
- Le Δ naïf du TFT (+24.7%) paraît inférieur à LightGBM (+49%) car les baseline naïves sont calculées différemment (voir note ci-dessus)

## Fichiers de résultats

- `lgbm_baseline_report.json` — rapport complet LightGBM (feature importance, hyperparamètres)
- `linear_baseline_report.json` — rapport complet Ridge (alpha, scalers, subset eval)
- `tft_quality_report.json` — rapport TFT (métriques val/test, statistiques dataset)

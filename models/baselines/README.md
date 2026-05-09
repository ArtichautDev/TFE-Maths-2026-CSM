# Baselines

Deux modèles de référence pour comparer avec le TFT, entraînés sur les mêmes features.

---

## LightGBM (`train_lgbm.py`)

Gradient boosting avec objectif MAE (identique à la loss TFT).

```bash
python train_lgbm.py
# Options :
#   --data-dir <path>   répertoire data/ (défaut: ../../data)
#   --out-dir  <path>   répertoire de sortie (défaut: ../runs)
```

**Paramètres clés** :
- `objective: regression_l1` (MAE)
- `num_leaves: 255`, `learning_rate: 0.05`
- Early stopping (patience 50 rounds) sur val.parquet
- 10M lignes train (tail-read), ≤1000 rounds

**Résultats** (202 iterations) :
- MAE = 57.90 sec | RMSE = 81.84 sec | Δ naïf = +49.0%

**Top features (gain)** : `fiabilite_train_historique`, `position_dans_trajet`, `retard_moyen_meme_jour_semaine`, `temps_theorique_depuis_depart`, `retard_j_1`...

---

## Ridge polynomial (`train_ridge.py`)

Régression Ridge avec features polynomiales deg-2 sur les 15 features les plus importantes selon LightGBM.

```bash
python train_ridge.py
# Options :
#   --max-train-rows N   nombre de lignes train (défaut: 45M)
#   --data-dir <path>
#   --out-dir  <path>
```

**Algorithme** : équations normales incrémentales (3 passes sur les données)  
→ exact sur 45M lignes, mémoire O(p²) ≈ quelques MB, pas de batch ML

**Features** : 75 originales + 120 termes croisés = 195 features totales  
**Alpha optimal** : 0.1 (sélectionné sur val.parquet)

**Résultats** :
- MAE = 65.17 sec | RMSE = 80.87 sec | Δ naïf = +10.9%
- RMSE légèrement meilleure que LightGBM (moins d'outliers) mais MAE plus élevée

# Données

Les fichiers parquet ne sont pas inclus dans ce repo (train.parquet ≈ 2.7 Go, 45M lignes).  
Disponibles sur demande.

## Fichiers inclus

| Fichier | Description |
|---------|-------------|
| `scaler_mappings.json` | Paramètres de normalisation (mean, std) pour chaque feature numérique |
| `categorical_mappings.json` | Encodage entier pour les variables catégorielles (relation, type train, gare, etc.) |
| `sample_5rows.csv` | 5 lignes représentatives — schéma complet du parquet |

## Splits temporels

| Set | Période | Lignes |
|-----|---------|--------|
| train | ≤ 2025-02-28 | ~45M |
| val | 2025-03 → 2025-07-31 | ~1.5M |
| test | ≥ 2025-08-01 | ~9.4M |

## Schéma principal

Chaque ligne = un arrêt d'un train à une gare, à une date donnée.

| Colonne | Type | Description |
|---------|------|-------------|
| `group_id` | str | Identifiant série temporelle (`<numero_train>_<gare>`) |
| `time_idx` | int | Index temporel global (jours depuis origine) |
| `target` | float | Retard à l'arrivée en **secondes**, clippé à [-20, 240] |
| `DELAY_ARR` | float | Retard brut (secondes), non clippé |
| `retard_j_1` | float (norm.) | Retard J-1 normalisé (z-score, mean=1.82 min, std=4.65 min) |
| `retard_j_{2..21}` | float (norm.) | Retards historiques J-2 à J-21 |
| `fiabilite_train_historique` | float (norm.) | Score de fiabilité historique du train |
| `position_dans_trajet` | float (norm.) | Position normalisée dans le trajet (0=départ, 1=arrivée) |
| `retard_moyen_meme_jour_semaine` | float (norm.) | Retard moyen historique ce jour de la semaine |
| `temps_theorique_{depuis_depart,jusque_arrivee,trajet}` | float (norm.) | Durées théoriques |
| `meteo_{temperature,precipitations,vent_vitesse,...}` | float (norm.) | Conditions météo à la gare |
| `is_{heure_pointe,jour_ferie,vacances_scolaires}` | float | Indicateurs temporels |
| `densite_prevue_hub_*` | float (norm.) | Densité prévue aux hubs (Bruxelles Nord-Midi, Anvers, Gand...) |
| `travaux_{nb_actifs,niveau_impact,distance_km}` | float | Impact des travaux de voie |
| `risque_retard_score` | float (norm.) | Score de risque agrégé |

Toutes les features numériques sont normalisées en z-score sauf `target` et `DELAY_ARR`.

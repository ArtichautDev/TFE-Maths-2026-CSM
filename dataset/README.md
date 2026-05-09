# Génération du dataset

Pipeline Python (Polars) qui construit `dataset_final.parquet` à partir du CSV de ponctualité brut SNCB.  
7 modules indépendants chaînés, chacun lisant le parquet du module précédent.

## Vue d'ensemble

```
dataset_optimized.parquet          ← CSV ponctualité converti en parquet (entrée)
        │
        ▼
Module A — Temporal & Static       → dataset_moduleA_verified.parquet
        │  features temporelles cycliques (sin/cos heure, jour, mois)
        │  features statiques de base
        ▼
Module B/C — Calendar & Spatial    → dataset_moduleB_C_verified.parquet
        │  jours fériés, vacances scolaires belges
        │  GTFS : hubs, nb connexions, position dans trajet, est_terminus
        ▼
Module D — Density                 → dataset_moduleD_verified.parquet
        │  densité de trains par heure dans les 7 hubs majeurs
        │  (Jonction Nord-Midi, Schaerbeek, Midi, Anvers, Gand, Liège, Charleroi)
        ▼
Module E — Travaux                 → dataset_moduleE_verified.parquet
        │  travaux Infrabel : nb actifs, niveau impact, distance, type
        │  source : langetermijnplanning-tcr-spoornetwerk.csv (Infrabel opendata)
        ▼
Module G — Météo                   → dataset_moduleG_verified.parquet
        │  conditions météo SYNOP IRM : température, précipitations,
        │  vent, nébulosité, conditions extrêmes
        │  source : synop_2022_2025.csv
        ▼
Module H — Historique              → dataset_moduleH_verified.parquet
        │  features de ponctualité historique par train × gare :
        │  retard J-1 à J-21, retard moyen jour de semaine,
        │  rolling 7j/30j par ligne et par gare, fiabilité historique
        ▼
Module I — Derived                 → dataset_final.parquet
           feature engineering final : risque_retard_score,
           stress_reseau_score, indice_complexite
           normalisation z-score, nettoyage, target = clip(DELAY_ARR, -20, 240)
```

## Utilisation

```bash
pip install polars numpy pyarrow pyyaml

# Lancer tout le pipeline
python rebuild_all.py --output-dir /path/to/output

# Ou module par module
python modules/module_A_temporal.py
python modules/module_B_calendar_spatial.py
# ...
```

## Données nécessaires (non incluses)

| Fichier | Taille | Source |
|---------|--------|--------|
| `dataset_optimized.parquet` | ~2 Go | CSV ponctualité SNCB converti (voir `pipeline/`) |
| `synop_2022_2025.csv` | ~124 Mo | IRM Belgique — données SYNOP horaires |
| `langetermijnplanning-tcr-spoornetwerk.csv` | ~26 Mo | Infrabel opendata — travaux réseau |

## Données incluses dans ce repo

| Fichier | Description |
|---------|-------------|
| `data/gares.csv` | Liste des ~1320 gares belges avec coordonnées GPS (383 Ko) |
| `data/mappings/gare_station_meteo.json` | Mapping gare → station SYNOP la plus proche (735 Ko) |

## Chemins à adapter

Les modules utilisent des chemins relatifs depuis `/root/SNCB/` (serveur d'entraînement).  
Pour une nouvelle machine, mettre à jour les variables `INPUT_PATH` / `OUTPUT_PATH` en tête de chaque module,  
ou passer `--output-dir` à `rebuild_all.py` (propagé via la variable d'environnement `SNCB_OUTPUT_DIR`).

## Features produites (75 utilisées par le TFT)

Voir [`../data/README.md`](../data/README.md) pour le schéma complet.

| Module | Features produites |
|--------|-------------------|
| A | `dt_hour`, `heure_prevue_sin/cos`, `jour_semaine_sin/cos`, `mois_sin/cos`, `fest_weekend` |
| B/C | `is_jour_ferie`, `is_vacances_scolaires`, `is_heure_pointe`, `nb_arrets_total`, `position_dans_trajet`, `est_gare_terminus`, `nb_connexions_gare`, `est_hub_majeur`, `traverse_hub_*`, `temps_theorique_*` |
| D | `densite_prevue_hub_*`, `congestion_prevue_score`, `nb_trains_prevus_ligne_heure` |
| E | `travaux_nb_actifs_ligne`, `travaux_niveau_impact`, `travaux_distance_km`, `travaux_actif_ligne` |
| G | `meteo_temperature`, `meteo_precipitations`, `meteo_vent_vitesse`, `meteo_nebulosite`, `meteo_conditions_extremes`, `meteo_observee_j_1` |
| H | `retard_j_1` … `retard_j_21`, `retard_moyen_meme_jour_semaine`, `retard_moyen_ligne_7j/30j`, `retard_moyen_gare_7j`, `fiabilite_train_historique`, `lag_missing_*`, `taux_ponctualite_ligne_7j`, `volatilite_retard_ligne_7j` |
| I | `risque_retard_score`, `stress_reseau_score`, `indice_complexite`, `target` |

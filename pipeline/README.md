# Pipeline d'enrichissement météo (Node.js)

Enrichit le CSV brut de ponctualité SNCB (86M+ lignes) avec les données météo SYNOP de l'IRM belge.  
Pour chaque trajet, ajoute les conditions météo à la gare de départ et d'arrivée.

## Prérequis

- Node.js ≥ 18
- Données sources dans `data/` :
  - `data/gares.csv` — liste des gares belges avec coordonnées GPS
  - `data/weather/synop_station.csv` — 27 stations météo SYNOP
  - `data/weather/synop_data.csv` + `synop_data-3.csv` — mesures horaires 2022-2025
  - `data/punctuality/combined/punctuality_202201_202510.csv` — CSV de ponctualité SNCB (86M lignes)

## Installation

```bash
npm install
```

## Exécution

```bash
bash run-pipeline.sh
```

Ou étape par étape :

```bash
# Étape 1 : mapping gare → station météo la plus proche
node scripts/build-station-mapping.js \
  --gares data/gares.csv \
  --stations data/weather/synop_station.csv \
  --output data/gare-to-station.json \
  --max-distance 50

# Étape 2 : index SYNOP en mémoire (lookup O(1))
node scripts/index-synop-data.js \
  --input data/weather/synop_data.csv \
  --input data/weather/synop_data-3.csv \
  --output data/indexes/synop-index.json \
  --timezone Europe/Brussels

# Étape 3 : enrichissement en streaming multi-thread
node --expose-gc scripts/enrich-ponctualite.js \
  --input data/punctuality/combined/punctuality_202201_202510.csv \
  --output data/punctuality/combined/punctuality_enriched.csv \
  --mapping data/gare-to-station.json \
  --synop-index data/indexes/synop-index.json \
  --workers 4 \
  --batch-size 20000 \
  --timezone Europe/Brussels
```

## Colonnes ajoutées

Le CSV enrichi contient les colonnes originales + météo départ/arrivée :

- `temp_depart/arrivee`, `precip_depart/arrivee`, `wind_speed_depart/arrivee`
- `humidity_depart/arrivee`, `pressure_depart/arrivee`
- `depart/arrival_station_meteo`, `depart/arrival_station_distance_km`
- `depart/arrival_weather_offset_hours` (0 = données exactes à l'heure)

Fallback intelligent : si pas de données à l'heure exacte, cherche ±1h, ±2h, ... jusqu'à ±168h.

## Temps estimé

~30-45 minutes pour 86M lignes (4 workers, SSD).

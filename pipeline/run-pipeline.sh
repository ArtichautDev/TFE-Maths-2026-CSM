#!/bin/bash
# Script d'exécution complète du pipeline d'enrichissement

set -e

echo "🚂 Pipeline d'enrichissement SNCB - Démarrage"
echo "=============================================="
echo ""

# Configuration
GARES_CSV="data/gares.csv"
STATIONS_CSV="data/weather/synop_station.csv"
SYNOP_DATA1="data/weather/synop_data.csv"
SYNOP_DATA2="data/weather/synop_data-3.csv"
PUNCTUALITY_CSV="data/punctuality/combined/punctuality_202201_202510.csv"

MAPPING_JSON="data/gare-to-station.json"
SYNOP_INDEX="data/indexes/synop-index.json"
OUTPUT_CSV="data/punctuality/combined/punctuality_enriched.csv"

WORKERS=${WORKERS:-4}
BATCH_SIZE=${BATCH_SIZE:-20000}

# Vérification des fichiers sources
echo "📋 Vérification des fichiers sources..."
for file in "$GARES_CSV" "$STATIONS_CSV" "$SYNOP_DATA1" "$SYNOP_DATA2" "$PUNCTUALITY_CSV"; do
  if [ ! -f "$file" ]; then
    echo "❌ Fichier manquant: $file"
    exit 1
  fi
done
echo "✅ Tous les fichiers sources sont présents"
echo ""

# Étape 1 : Mapping gare → station météo
echo "📍 Étape 1/3 : Création du mapping gare → station météo"
echo "--------------------------------------------------------"
node scripts/build-station-mapping.js \
  --gares "$GARES_CSV" \
  --stations "$STATIONS_CSV" \
  --output "$MAPPING_JSON" \
  --max-distance 50

if [ $? -ne 0 ]; then
  echo "❌ Échec de l'étape 1"
  exit 1
fi
echo "✅ Mapping créé: $MAPPING_JSON"
echo ""

# Étape 2 : Indexation SYNOP
echo "🌤️  Étape 2/3 : Indexation des données météo SYNOP"
echo "---------------------------------------------------"
node scripts/index-synop-data.js \
  --input "$SYNOP_DATA1" \
  --input "$SYNOP_DATA2" \
  --output "$SYNOP_INDEX" \
  --timezone Europe/Brussels

if [ $? -ne 0 ]; then
  echo "❌ Échec de l'étape 2"
  exit 1
fi
echo "✅ Index SYNOP créé: $SYNOP_INDEX"
echo ""

# Étape 3 : Enrichissement
echo "⚡ Étape 3/3 : Enrichissement du CSV de ponctualité"
echo "---------------------------------------------------"
echo "Workers: $WORKERS | Batch size: $BATCH_SIZE"
echo ""

node --expose-gc scripts/enrich-ponctualite.js \
  --input "$PUNCTUALITY_CSV" \
  --output "$OUTPUT_CSV" \
  --mapping "$MAPPING_JSON" \
  --synop-index "$SYNOP_INDEX" \
  --workers "$WORKERS" \
  --batch-size "$BATCH_SIZE" \
  --timezone Europe/Brussels

if [ $? -ne 0 ]; then
  echo "❌ Échec de l'étape 3"
  exit 1
fi

echo ""
echo "✅ Pipeline terminé avec succès !"
echo "📁 Fichier enrichi: $OUTPUT_CSV"
echo ""

# Statistiques finales
if [ -f "$OUTPUT_CSV" ]; then
  SIZE=$(du -h "$OUTPUT_CSV" | cut -f1)
  LINES=$(wc -l < "$OUTPUT_CSV" | tr -d ' ')
  echo "📊 Statistiques:"
  echo "   - Taille: $SIZE"
  echo "   - Lignes: $LINES"
fi


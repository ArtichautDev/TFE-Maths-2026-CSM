#!/usr/bin/env node
'use strict';

const fs = require('fs');
const fsp = require('fs/promises');
const path = require('path');
const csv = require('csv-parser');
const { performance } = require('perf_hooks');

const { haversineDistance } = require('./utils/haversine');

const DEFAULT_GARES_CSV = path.resolve(process.cwd(), 'data', 'gares.csv');
const DEFAULT_STATIONS_CSV = path.resolve(process.cwd(), 'data', 'weather', 'synop_station.csv');
const DEFAULT_OUTPUT = path.resolve(process.cwd(), 'data', 'gare-to-station.json');

const GARE_ID_FIELDS = ['gare_id', 'id', 'UIC', 'code', 'Code TAF/TAP', 'PTCAR ID'];
const STATION_ID_FIELDS = ['station_id', 'id', 'station', 'code'];
const NAME_FIELDS = ['name', 'nom', 'station_name', 'Nom FR complet', 'Nom FR court'];
const LAT_FIELDS = ['lat', 'latitude', 'latitude_deg'];
const LON_FIELDS = ['lon', 'lng', 'longitude', 'longitude_deg'];
const GEOM_FIELDS = ['Geo Point', 'geo_point', 'the_geom', 'geometry'];

async function main() {
  const startTs = performance.now();

  const { garesCsv, stationsCsv, outputFile, maxDistance } = parseArgs(process.argv.slice(2));
  console.log('[build-station-mapping] Démarrage');
  console.log(`  Gares:    ${garesCsv}`);
  console.log(`  Stations: ${stationsCsv}`);
  console.log(`  Output:   ${outputFile}`);
  console.log(`  maxDistance: ${maxDistance} km`);

  await ensureFileExists(garesCsv, 'CSV des gares');
  await ensureFileExists(stationsCsv, 'CSV des stations SYNOP');

  const [stations, gares] = await Promise.all([
    loadCsv(stationsCsv, 'stations'),
    loadCsv(garesCsv, 'gares'),
  ]);

  console.log(`Stations chargées: ${stations.length}`);
  console.log(`Gares chargées: ${gares.length}`);

  const stationRecords = stations
    .map((row) => normalizeStation(row))
    .filter((record) => record != null);

  if (!stationRecords.length) {
    throw new Error('Aucune station valide trouvée (lat/lon manquants ?)');
  }

  const mapping = {};
  let skipped = 0;
  let mappedGares = 0;

  for (const rawGare of gares) {
    const gare = normalizeGare(rawGare);
    if (!gare) {
      skipped += 1;
      continue;
    }

    const { closestStation, minDistanceKm } = findClosestStation(gare, stationRecords);

    if (!closestStation) {
      skipped += 1;
      continue;
    }

    const record = {
      station_id: closestStation.primaryId,
      station_name: closestStation.name,
      distance_km: Number(minDistanceKm.toFixed(3)),
    };

    if (minDistanceKm > maxDistance) {
      record.no_station = true;
    }

    let added = false;
    for (const id of gare.ids) {
      if (!id) continue;
      mapping[id] = record;
      added = true;
    }
    if (added) {
      mappedGares += 1;
    } else {
      skipped += 1;
    }
  }

  console.log(`Gares mappées: ${mappedGares}`);
  console.log(`Identifiants enregistrés: ${Object.keys(mapping).length}`);
  console.log(`Gares ignorées: ${skipped}`);

  await fsp.mkdir(path.dirname(outputFile), { recursive: true });
  await fsp.writeFile(outputFile, JSON.stringify(mapping, null, 2), 'utf8');

  const durationMs = performance.now() - startTs;
  console.log(`Mapping écrit dans ${outputFile}`);
  console.log(`Durée totale: ${(durationMs / 1000).toFixed(2)}s`);
}

function parseArgs(args) {
  const options = {
    garesCsv: DEFAULT_GARES_CSV,
    stationsCsv: DEFAULT_STATIONS_CSV,
    outputFile: DEFAULT_OUTPUT,
    maxDistance: 50,
  };

  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];
    switch (arg) {
      case '--gares':
      case '-g':
        options.garesCsv = resolveArgPath(args[++i], '--gares');
        break;
      case '--stations':
      case '-s':
        options.stationsCsv = resolveArgPath(args[++i], '--stations');
        break;
      case '--output':
      case '-o':
        options.outputFile = resolveArgPath(args[++i], '--output');
        break;
      case '--max-distance':
        options.maxDistance = Number(args[++i]);
        if (Number.isNaN(options.maxDistance) || options.maxDistance < 0) {
          throw new Error('Valeur invalide pour --max-distance');
        }
        break;
      case '--help':
      case '-h':
        printHelp();
        process.exit(0);
        break;
      default:
        throw new Error(`Option inconnue: ${arg}`);
    }
  }

  return options;
}

function resolveArgPath(value, flagName) {
  if (!value) {
    throw new Error(`Valeur manquante pour ${flagName}`);
  }
  return path.resolve(process.cwd(), value);
}

function printHelp() {
  console.log(`Usage: node build-station-mapping.js [options]\n\n` +
    'Options:\n' +
    '  --gares, -g <path>        CSV des gares (défaut: data/gares.csv)\n' +
    '  --stations, -s <path>     CSV des stations SYNOP (défaut: data/weather/synop_station.csv)\n' +
    '  --output, -o <path>       Fichier JSON de sortie (défaut: data/gare-to-station.json)\n' +
    '  --max-distance <km>       Seuil distance km pour flag no_station (défaut: 50)\n' +
    '  --help, -h                Affiche cette aide');
}

async function ensureFileExists(filePath, label) {
  try {
    await fsp.access(filePath, fs.constants.R_OK);
  } catch (err) {
    throw new Error(`${label} introuvable: ${filePath}`);
  }
}

async function loadCsv(filePath, label) {
  const delimiter = await detectDelimiter(filePath);
  return new Promise((resolve, reject) => {
    const rows = [];
    fs.createReadStream(filePath)
      .pipe(csv({
        separator: delimiter,
        mapHeaders: ({ header }) => header.trim(),
        mapValues: ({ value }) => (typeof value === 'string' ? value.trim() : value),
      }))
      .on('data', (data) => rows.push(data))
      .on('error', (error) => reject(new Error(`Erreur lecture ${label}: ${error.message}`)))
      .on('end', () => resolve(rows));
  });
}

function normalizeStation(row) {
  const ids = collectIdentifiers(row, STATION_ID_FIELDS);
  const primaryId = ids[0];
  const name = pickField(row, NAME_FIELDS) || primaryId;
  let lat = parseFloat(pickField(row, LAT_FIELDS));
  let lon = parseFloat(pickField(row, LON_FIELDS));

  if ((Number.isNaN(lat) || Number.isNaN(lon)) && row) {
    const geomValue = pickField(row, GEOM_FIELDS);
    if (geomValue) {
      const coords = parsePointGeometry(geomValue);
      if (coords) {
        lat = coords.lat;
        lon = coords.lon;
      }
    }
  }

  if (!primaryId || Number.isNaN(lat) || Number.isNaN(lon)) {
    return null;
  }

  return { ids, primaryId: String(primaryId), name: String(name), lat, lon };
}

function normalizeGare(row) {
  const ids = collectIdentifiers(row, GARE_ID_FIELDS);
  const primaryId = ids[0];
  const name = pickField(row, NAME_FIELDS) || primaryId;
  let lat = parseFloat(pickField(row, LAT_FIELDS));
  let lon = parseFloat(pickField(row, LON_FIELDS));

  if ((Number.isNaN(lat) || Number.isNaN(lon)) && row) {
    const geomValue = pickField(row, GEOM_FIELDS);
    if (geomValue) {
      const coords = parsePointGeometry(geomValue);
      if (coords) {
        lat = coords.lat;
        lon = coords.lon;
      }
    }
  }

  if (!ids.length || Number.isNaN(lat) || Number.isNaN(lon)) {
    return null;
  }

  return { ids: ids.map(String), primaryId: String(primaryId), name: String(name), lat, lon };
}

function pickField(row, candidates) {
  for (const field of candidates) {
    if (Object.prototype.hasOwnProperty.call(row, field) && row[field] !== undefined && row[field] !== '') {
      return row[field];
    }
  }
  return undefined;
}

function collectIdentifiers(row, candidates) {
  const ids = [];
  const seen = new Set();
  for (const field of candidates) {
    if (!Object.prototype.hasOwnProperty.call(row, field)) continue;
    const value = row[field];
    if (value === undefined || value === null || value === '') continue;
    const trimmed = String(value).trim();
    if (!trimmed) continue;
    if (!seen.has(trimmed)) {
      ids.push(trimmed);
      seen.add(trimmed);
    }
  }
  return ids;
}

function findClosestStation(gare, stations) {
  let closestStation = null;
  let minDistanceKm = Number.POSITIVE_INFINITY;

  for (const station of stations) {
    const distance = haversineDistance(gare, station);
    if (distance < minDistanceKm) {
      minDistanceKm = distance;
      closestStation = station;
    }
  }

  return { closestStation, minDistanceKm };
}

async function detectDelimiter(filePath) {
  const sample = await fsp.readFile(filePath, 'utf8');
  const firstLine = sample.split(/\r?\n/).find((line) => line.trim().length > 0) || '';
  const semicolons = (firstLine.match(/;/g) || []).length;
  const commas = (firstLine.match(/,/g) || []).length;
  return semicolons > commas ? ';' : ',';
}

function parsePointGeometry(raw) {
  if (!raw || typeof raw !== 'string') {
    return null;
  }

  const geoPointMatch = raw.match(/(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)/);
  if (geoPointMatch) {
    const lat = parseFloat(geoPointMatch[1]);
    const lon = parseFloat(geoPointMatch[2]);
    if (!Number.isNaN(lat) && !Number.isNaN(lon)) {
      return { lat, lon };
    }
  }

  const pointMatch = raw.match(/POINT\s*\(\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*\)/i);
  if (pointMatch) {
    const lat = parseFloat(pointMatch[1]);
    const lon = parseFloat(pointMatch[2]);
    if (!Number.isNaN(lat) && !Number.isNaN(lon)) {
      return { lat, lon };
    }
  }

  if (/coordinates/i.test(raw)) {
    try {
      const json = JSON.parse(raw.replace(/'/g, '"'));
      if (Array.isArray(json.coordinates) && json.coordinates.length >= 2) {
        const [lon, lat] = json.coordinates;
        if (!Number.isNaN(lat) && !Number.isNaN(lon)) {
          return { lat: Number(lat), lon: Number(lon) };
        }
      }
    } catch (err) {
      // ignore parsing error
    }
  }

  return null;
}

main().catch((err) => {
  console.error('[build-station-mapping] ERREUR:', err.message);
  if (process.env.DEBUG) {
    console.error(err);
  }
  process.exitCode = 1;
});

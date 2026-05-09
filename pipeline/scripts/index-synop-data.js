#!/usr/bin/env node
'use strict';

const fs = require('fs');
const fsp = require('fs/promises');
const path = require('path');
const csv = require('csv-parser');
const { performance } = require('perf_hooks');
const cliProgress = require('cli-progress');

const { floorToUtcHourKey } = require('./utils/datetime-utils');

const DEFAULT_INPUT = path.resolve(process.cwd(), 'data', 'synop_data.csv');
const DEFAULT_OUTPUT = path.resolve(process.cwd(), 'data', 'indexes', 'synop-index.json');

const STATION_ID_FIELDS = ['station_id', 'station', 'id', 'code'];
const TIME_FIELDS = ['time', 'datetime', 'timestamp', 'date'];
const TEMP_FIELDS = ['temperature', 'temp', 'air_temperature'];
const PRECIP_FIELDS = ['precipitation', 'precip', 'rain', 'precip_quantity'];
const WIND_FIELDS = ['wind_speed', 'wind', 'ff'];
const HUMIDITY_FIELDS = ['humidity', 'relative_humidity', 'humidity_relative'];
const PRESSURE_FIELDS = ['pressure', 'pressure_station', 'p', 'pressure_station_level'];

const DEFAULT_BUFFER_SIZE = 50000;

async function main() {
  const startTs = performance.now();
  const { inputs, output, format, timezone } = parseArgs(process.argv.slice(2));

  console.log('[index-synop-data] Démarrage');
  console.log(`  Inputs:`);
  for (const input of inputs) {
    console.log(`    - ${input}`);
  }
  console.log(`  Output: ${output}`);
  console.log(`  Format: ${format}`);
  console.log(`  Timezone trains -> UTC: ${timezone}`);

  await Promise.all(inputs.map((input) => ensureFileExists(input, 'Dataset SYNOP')));
  await fsp.mkdir(path.dirname(output), { recursive: true });

  const index = new Map();
  const stationCounts = new Map();
  const bar = new cliProgress.SingleBar({
    format: '[index-synop-data] {bar} {value} lignes',
    hideCursor: true,
  }, cliProgress.Presets.shades_classic);
  bar.start(0, 0);

  let lineCount = 0;
  let skipped = 0;

  for (const input of inputs) {
    console.log(`[index-synop-data] Lecture ${input}`);
    await new Promise((resolve, reject) => {
      fs.createReadStream(input)
        .pipe(csv())
        .on('data', (row) => {
          bar.increment();
          lineCount += 1;
          try {
            const stationId = pickField(row, STATION_ID_FIELDS);
            const timeValue = pickField(row, TIME_FIELDS);
            if (!stationId || !timeValue) {
              skipped += 1;
              return;
            }

            const hourKey = floorToUtcHourKey(timeValue, { timezone });
            const payload = buildWeatherPayload(row);
            setIndexValue(index, stationId, hourKey, payload);

            stationCounts.set(stationId, (stationCounts.get(stationId) || 0) + 1);
          } catch (err) {
            skipped += 1;
            if (process.env.DEBUG) {
              console.warn('Ligne ignorée:', err.message, row);
            }
          }
        })
        .on('error', (err) => reject(err))
        .on('end', () => resolve());
    });
  }

  bar.stop();

  console.log(`Lignes lues: ${lineCount}`);
  console.log(`Lignes ignorées: ${skipped}`);
  console.log(`Stations indexées: ${index.size}`);

  const serializable = mapToObject(index);
  await writeIndex(serializable, output, format);

  const durationMs = performance.now() - startTs;
  console.log(`Index sauvegardé (${format}) dans ${output}`);
  console.log(`Durée totale: ${(durationMs / 1000).toFixed(2)}s`);
}

function parseArgs(args) {
  const options = {
    inputs: [],
    output: DEFAULT_OUTPUT,
    format: 'json',
    timezone: 'Europe/Brussels',
  };

  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];
    switch (arg) {
      case '--input':
      case '-i':
        if (!options.inputs) {
          options.inputs = [];
        }
        options.inputs.push(resolvePath(args[++i], '--input'));
        break;
      case '--output':
      case '-o':
        options.output = resolvePath(args[++i], '--output');
        break;
      case '--format':
      case '-f':
        options.format = (args[++i] || '').toLowerCase();
        if (!['json'].includes(options.format)) {
          throw new Error('Formats supportés: json');
        }
        break;
      case '--timezone':
        options.timezone = args[++i] || 'Europe/Brussels';
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

  if (!options.inputs || options.inputs.length === 0) {
    options.inputs = [DEFAULT_INPUT];
  }

  return options;
}

function printHelp() {
  console.log(`Usage: node index-synop-data.js [options]\n\n` +
    'Options:\n' +
    '  --input, -i <path>     CSV SYNOP (option répétable, défaut: data/synop_data.csv)\n' +
    '  --output, -o <path>    Fichier d\'index (défaut: data/indexes/synop-index.json)\n' +
    '  --format, -f <json>    Format de sortie (défaut: json)\n' +
    '  --timezone <tz>        Fuseau des heures sources (défaut: Europe/Brussels)\n' +
    '  --help, -h             Affiche cette aide');
}

function resolvePath(value, flag) {
  if (!value) {
    throw new Error(`Valeur manquante pour ${flag}`);
  }
  return path.resolve(process.cwd(), value);
}

async function ensureFileExists(filePath, label) {
  try {
    await fsp.access(filePath, fs.constants.R_OK);
  } catch (err) {
    throw new Error(`${label} introuvable: ${filePath}`);
  }
}

function pickField(row, candidates) {
  for (const field of candidates) {
    if (row[field] !== undefined && row[field] !== '') {
      return row[field];
    }
  }
  return undefined;
}

function buildWeatherPayload(row) {
  const payload = {};
  payload.temp = toFloat(pickField(row, TEMP_FIELDS));
  payload.precip = toFloat(pickField(row, PRECIP_FIELDS));
  payload.wind_speed = toFloat(pickField(row, WIND_FIELDS));
  payload.humidity = toFloat(pickField(row, HUMIDITY_FIELDS));
  payload.pressure = toFloat(pickField(row, PRESSURE_FIELDS));

  return payload;
}

function toFloat(value) {
  if (value === undefined || value === null || value === '') {
    return null;
  }
  const num = parseFloat(String(value).replace(',', '.'));
  return Number.isNaN(num) ? null : num;
}

function setIndexValue(index, stationId, hourKey, payload) {
  if (!index.has(stationId)) {
    index.set(stationId, new Map());
  }
  const stationMap = index.get(stationId);
  stationMap.set(hourKey, payload);
}

function mapToObject(index) {
  const obj = {};
  for (const [stationId, stationMap] of index.entries()) {
    obj[stationId] = Object.fromEntries(stationMap.entries());
  }
  return obj;
}

async function writeIndex(data, filePath, format) {
  switch (format) {
    case 'json':
      await fsp.writeFile(filePath, JSON.stringify(data), 'utf8');
      break;
    default:
      throw new Error(`Format non supporté: ${format}`);
  }
}

main().catch((err) => {
  console.error('[index-synop-data] ERREUR:', err.message);
  if (process.env.DEBUG) {
    console.error(err);
  }
  process.exitCode = 1;
});

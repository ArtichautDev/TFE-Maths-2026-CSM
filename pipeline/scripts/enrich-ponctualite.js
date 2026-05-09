#!/usr/bin/env node
'use strict';

const fs = require('fs');
const fsp = require('fs/promises');
const path = require('path');
const os = require('os');
const { Worker } = require('worker_threads');
const csv = require('csv-parser');
const { createObjectCsvStringifier } = require('csv-writer');
const cliProgress = require('cli-progress');

const { OUTPUT_COLUMNS } = require('./utils/punctuality-helper');

const DEFAULT_INPUT = path.resolve(process.cwd(), 'data', 'punctuality', 'combined', 'punctuality_202201_202510.csv');
const DEFAULT_OUTPUT = path.resolve(process.cwd(), 'data', 'punctuality', 'combined', 'punctuality_enriched.csv');
const DEFAULT_MAPPING = path.resolve(process.cwd(), 'data', 'gare-to-station.json');
const DEFAULT_SYNOP_INDEX = path.resolve(process.cwd(), 'data', 'indexes', 'synop-index.json');
const DEFAULT_TIMEZONE = 'Europe/Brussels';
const DEFAULT_BATCH_SIZE = 10000;
const DEFAULT_WORKERS = Math.max(1, Math.min(4, os.cpus().length));
const MAX_QUEUE_MULTIPLIER = 4;

async function main() {
  const {
    input,
    output,
    mappingPath,
    synopIndexPath,
    timezone,
    batchSize,
    workers: workerCount,
  } = parseArgs(process.argv.slice(2));

  console.log('[enrich-ponctualite] Démarrage multi-thread');
  console.log(`  Input:          ${input}`);
  console.log(`  Output:         ${output}`);
  console.log(`  Mapping:        ${mappingPath}`);
  console.log(`  Synop index:    ${synopIndexPath}`);
  console.log(`  Timezone:       ${timezone}`);
  console.log(`  Batch taille:   ${batchSize}`);
  console.log(`  Workers:        ${workerCount}`);

  await Promise.all([
    ensureFileExists(input, 'CSV ponctualité'),
    ensureFileExists(mappingPath, 'Mapping gare-station'),
    ensureFileExists(synopIndexPath, 'Index SYNOP'),
  ]);

  await fsp.mkdir(path.dirname(output), { recursive: true });

  const mappingInfo = {
    mappingPath,
    synopIndexPath,
    timezone,
  };

  const pool = new WorkerPool(workerCount, mappingInfo);

  const readStream = fs.createReadStream(input);
  const parser = csv();

  let baseHeaders = null;
  parser.on('headers', (headers) => {
    baseHeaders = headers;
  });

  const bar = new cliProgress.SingleBar({
    format: '[enrich-ponctualite] {bar} {value} lignes traitées',
    hideCursor: true,
  }, cliProgress.Presets.shades_classic);
  bar.start(0, 0);

  const outputStream = fs.createWriteStream(output);
  let csvStringifier = null;
  let headerWritten = false;

  const jobQueue = [];
  const pendingResults = new Map();
  const maxQueueSize = workerCount * MAX_QUEUE_MULTIPLIER;
  const currentBatch = [];
  let readPaused = false;
  let nextJobId = 0;
  let nextToWrite = 0;
  let totalLines = 0;
  let missingStations = 0;
  let missingWeather = 0;
  let endOfStream = false;
  let resolveCompletion;
  let rejectCompletion;
  let completionResolved = false;

  const completionPromise = new Promise((resolve, reject) => {
    resolveCompletion = resolve;
    rejectCompletion = reject;
  });

  const maybeResolve = () => {
    if (!completionResolved && endOfStream && jobQueue.length === 0 && pendingResults.size === 0 && pool.isIdle()) {
      completionResolved = true;
      resolveCompletion();
    }
  };

  const initWriter = () => {
    if (headerWritten) {
      return;
    }
    if (!baseHeaders) {
      throw new Error('Impossible de déterminer les en-têtes du CSV source');
    }
    const headers = baseHeaders.map((key) => ({ id: key, title: key }));
    for (const key of OUTPUT_COLUMNS) {
      headers.push({ id: key, title: key });
    }
    csvStringifier = createObjectCsvStringifier({ header: headers });
    outputStream.write(csvStringifier.getHeaderString());
    headerWritten = true;
  };

  const flushResults = () => {
    while (pendingResults.has(nextToWrite)) {
      const result = pendingResults.get(nextToWrite);
      pendingResults.delete(nextToWrite);
      if (!headerWritten) {
        initWriter();
      }
      if (result.rows.length) {
        const chunk = csvStringifier.stringifyRecords(result.rows);
        if (!outputStream.write(chunk)) {
          outputStream.once('drain', () => {});
        }
      }
      nextToWrite += 1;
    }
    maybeResolve();
  };

  const dispatch = () => {
    while (pool.hasAvailableWorker() && jobQueue.length > 0) {
      const job = jobQueue.shift();
      pool.assign(job);
    }
    if (readPaused && jobQueue.length < maxQueueSize) {
      readPaused = false;
      readStream.resume();
    }
  };

  pool.onReady(() => {
    dispatch();
    maybeResolve();
  });

  pool.onResult((result) => {
    pendingResults.set(result.jobId, result);
    totalLines += result.rowsProcessed;
    missingStations += result.missingStations;
    missingWeather += result.missingWeather;
    bar.update(totalLines);
    flushResults();
    dispatch();
  });

  parser.on('data', (row) => {
    if (!baseHeaders) {
      baseHeaders = Object.keys(row);
    }
    currentBatch.push(row);
    if (currentBatch.length >= batchSize) {
      jobQueue.push({ jobId: nextJobId++, rows: currentBatch.splice(0, currentBatch.length) });
      dispatch();
    }
    if (jobQueue.length >= maxQueueSize && !readPaused) {
      readPaused = true;
      readStream.pause();
    }
  });

  parser.on('error', (err) => {
    if (rejectCompletion) {
      rejectCompletion(err);
    } else {
      throw err;
    }
  });

  parser.on('end', () => {
    if (currentBatch.length > 0) {
      jobQueue.push({ jobId: nextJobId++, rows: currentBatch.splice(0, currentBatch.length) });
      dispatch();
    }
    endOfStream = true;
    maybeResolve();
  });

  readStream.on('error', (err) => {
    if (rejectCompletion) {
      rejectCompletion(err);
    } else {
      throw err;
    }
  });

  readStream.pipe(parser);

  const inputSizeBytes = (await fsp.stat(input)).size;
  console.log(`[enrich-ponctualite] Taille fichier: ${(inputSizeBytes / (1024 ** 3)).toFixed(2)} Go`);

  await completionPromise;

  bar.stop();
  outputStream.end();

  console.log(`[enrich-ponctualite] Lignes traitées: ${totalLines}`);
  console.log(`[enrich-ponctualite] Gares sans station proche: ${missingStations}`);
  console.log(`[enrich-ponctualite] Horaires sans météo: ${missingWeather}`);
}

async function ensureFileExists(filePath, label) {
  try {
    await fsp.access(filePath, fs.constants.R_OK);
  } catch (err) {
    throw new Error(`${label} introuvable: ${filePath}`);
  }
}

function parseArgs(args) {
  const options = {
    input: DEFAULT_INPUT,
    output: DEFAULT_OUTPUT,
    mappingPath: DEFAULT_MAPPING,
    synopIndexPath: DEFAULT_SYNOP_INDEX,
    timezone: DEFAULT_TIMEZONE,
    batchSize: DEFAULT_BATCH_SIZE,
    workers: DEFAULT_WORKERS,
  };

  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];
    switch (arg) {
      case '--input':
      case '-i':
        options.input = resolvePath(args[++i], '--input');
        break;
      case '--output':
      case '-o':
        options.output = resolvePath(args[++i], '--output');
        break;
      case '--mapping':
      case '-m':
        options.mappingPath = resolvePath(args[++i], '--mapping');
        break;
      case '--synop-index':
      case '-s':
        options.synopIndexPath = resolvePath(args[++i], '--synop-index');
        break;
      case '--timezone':
        options.timezone = args[++i] || DEFAULT_TIMEZONE;
        break;
      case '--batch-size':
        options.batchSize = parseInt(args[++i], 10);
        if (!Number.isFinite(options.batchSize) || options.batchSize <= 0) {
          throw new Error('Valeur invalide pour --batch-size');
        }
        break;
      case '--workers':
      case '-w':
        options.workers = parseInt(args[++i], 10);
        if (!Number.isFinite(options.workers) || options.workers <= 0) {
          throw new Error('Valeur invalide pour --workers');
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

function resolvePath(value, flag) {
  if (!value) {
    throw new Error(`Valeur manquante pour ${flag}`);
  }
  return path.resolve(process.cwd(), value);
}

function printHelp() {
  console.log(`Usage: node enrich-ponctualite.js [options]\n\n` +
    'Options:\n' +
    '  --input, -i <path>          CSV ponctualité source\n' +
    '  --output, -o <path>         CSV enrichi\n' +
    '  --mapping, -m <path>        Mapping gare -> station météo\n' +
    '  --synop-index, -s <path>    Index SYNOP\n' +
    '  --timezone <tz>             Fuseau horaire source (défaut: Europe/Brussels)\n' +
    '  --batch-size <n>            Taille du batch de lignes (défaut: 10000)\n' +
    '  --workers, -w <n>           Nombre de workers (défaut: cpus)\n' +
    '  --help, -h                  Affiche cette aide');
}

class WorkerPool {
  constructor(size, initPayload) {
    this.workers = [];
    this.freeWorkers = [];
    this.activeJobs = 0;
    this.resultHandler = () => {};
    this.readyHandler = () => {};

    for (let i = 0; i < size; i += 1) {
      const worker = new Worker(path.resolve(__dirname, 'workers', 'punctuality-worker.js'));
      worker._id = i;
      worker.on('message', (msg) => this.handleMessage(worker, msg));
      worker.on('error', (err) => {
        console.error(`[worker ${worker._id}] erreur:`, err);
        process.exit(1);
      });
      worker.on('exit', (code) => {
        if (code !== 0) {
          console.error(`[worker ${worker._id}] exit code ${code}`);
        }
      });
      worker.postMessage({ type: 'init', ...initPayload });
      this.workers.push(worker);
    }
  }

  onResult(handler) {
    this.resultHandler = handler;
  }

  onReady(handler) {
    this.readyHandler = handler;
  }

  hasAvailableWorker() {
    return this.freeWorkers.length > 0;
  }

  assign(job) {
    if (!this.freeWorkers.length) {
      throw new Error('Aucun worker disponible');
    }
    const worker = this.freeWorkers.shift();
    this.activeJobs += 1;
    worker.postMessage({ type: 'task', jobId: job.jobId, rows: job.rows });
  }

  handleMessage(worker, msg) {
    if (msg.type === 'ready') {
      this.freeWorkers.push(worker);
      this.readyHandler();
      return;
    }

    if (msg.type === 'result') {
      this.freeWorkers.push(worker);
      this.activeJobs -= 1;
      this.resultHandler(msg);
      return;
    }
  }

  isIdle() {
    return this.activeJobs === 0 && this.freeWorkers.length === this.workers.length;
  }
}

main().catch((err) => {
  console.error('[enrich-ponctualite] ERREUR:', err.message);
  if (process.env.DEBUG) {
    console.error(err);
  }
  process.exitCode = 1;
});

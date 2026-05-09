'use strict';

const fs = require('fs');
const { parentPort } = require('worker_threads');

const { enrichRow } = require('../utils/punctuality-helper');

let mapping = null;
let synopIndex = null;
let timezone = 'Europe/Brussels';

parentPort.on('message', (msg) => {
  if (msg.type === 'init') {
    mapping = JSON.parse(fs.readFileSync(msg.mappingPath, 'utf8'));
    synopIndex = JSON.parse(fs.readFileSync(msg.synopIndexPath, 'utf8'));
    timezone = msg.timezone || timezone;
    parentPort.postMessage({ type: 'ready' });
    return;
  }

  if (msg.type === 'task') {
    const { jobId, rows } = msg;
    const enrichedRows = [];
    let missingStations = 0;
    let missingWeather = 0;

    for (const row of rows) {
      const enriched = enrichRow(row, mapping, synopIndex, timezone);
      if (enriched.depart_no_station || enriched.arrival_no_station) {
        missingStations += 1;
      }
      if (enriched._missingWeatherCount > 0) {
        missingWeather += 1;
      }
      delete enriched._missingWeatherCount;
      enrichedRows.push(enriched);
    }

    parentPort.postMessage({
      type: 'result',
      jobId,
      rows: enrichedRows,
      rowsProcessed: rows.length,
      missingStations,
      missingWeather,
    });
  }
});

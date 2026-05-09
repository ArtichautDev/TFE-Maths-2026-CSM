'use strict';

const { formatInTimeZone, fromZonedTime } = require('date-fns-tz');

const DEFAULT_TIMEZONE = 'Europe/Brussels';

/**
 * Convertit une date/heure en clé UTC "YYYY-MM-DD-HH".
 * @param {string} value - date ISO ou YYYY-MM-DD HH:mm
 * @param {Object} [options]
 * @param {string} [options.timezone='Europe/Brussels']
 * @returns {string}
 */
function toUtcHourKey(value, options = {}) {
  const { timezone = DEFAULT_TIMEZONE } = options;
  const normalized = normalizeDateTime(value);
  let utcDate;

  if (normalized.kind === 'utc') {
    utcDate = normalized.date;
  } else {
    utcDate = fromZonedTime(normalized.value, timezone);
  }

  return formatInTimeZone(utcDate, 'UTC', 'yyyy-MM-dd-HH');
}

/**
 * Arrondit une date à l'heure inférieure (minute -> 0) avant conversion en UTC.
 * @param {string} value
 * @param {Object} [options]
 * @param {string} [options.timezone='Europe/Brussels']
 * @returns {string}
 */
function floorToUtcHourKey(value, options = {}) {
  const { timezone = DEFAULT_TIMEZONE } = options;
  const normalized = normalizeDateTime(value);
  let utcDate;

  if (normalized.kind === 'utc') {
    const date = new Date(normalized.date.getTime());
    date.setUTCMinutes(0, 0, 0);
    utcDate = date;
  } else {
    const floored = normalized.value.replace(/(\d{2}):\d{2}(:\d{2})?$/, (_, hour) => `${hour}:00:00`);
    utcDate = fromZonedTime(floored, timezone);
  }

  return formatInTimeZone(utcDate, 'UTC', 'yyyy-MM-dd-HH');
}

function normalizeDateTime(raw) {
  if (!raw) {
    throw new Error('Date invalide: vide');
  }
  let value = String(raw).trim();

  if (/[+-]\d{2}:?\d{2}$/.test(value) || value.endsWith('Z')) {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
      throw new Error(`Date invalide: ${raw}`);
    }
    return { kind: 'utc', date };
  }

  if (/^\d{8}$/.test(value)) {
    value = `${value.slice(0, 4)}-${value.slice(4, 6)}-${value.slice(6, 8)} 00:00:00`;
  } else if (/^\d{4}-\d{2}-\d{2}$/.test(value)) {
    value = `${value} 00:00:00`;
  } else if (/^\d{4}-\d{2}-\d{2}[T\s]\d{2}$/.test(value)) {
    value = value.replace('T', ' ');
    value = `${value}:00:00`;
  } else if (/^\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}$/.test(value)) {
    value = value.replace('T', ' ');
    value = `${value}:00`;
  } else if (/^\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}$/.test(value)) {
    value = value.replace('T', ' ');
  } else if (/^\d{4}-\d{2}-\d{2}\s\d{4}$/.test(value)) {
    value = value.replace(/(\d{2})(\d{2})$/, '$1:$2');
    value = `${value}:00`;
  }

  if (!/^\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}$/.test(value)) {
    throw new Error(`Format de date non supporté: ${raw}`);
  }

  return { kind: 'naive', value };
}

module.exports = {
  DEFAULT_TIMEZONE,
  toUtcHourKey,
  floorToUtcHourKey,
};

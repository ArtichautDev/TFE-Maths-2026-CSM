'use strict';

const { floorToUtcHourKey } = require('./datetime-utils');

const OUTPUT_COLUMNS = [
  'temp_depart',
  'precip_depart',
  'wind_speed_depart',
  'humidity_depart',
  'pressure_depart',
  'temp_arrivee',
  'precip_arrivee',
  'wind_speed_arrivee',
  'humidity_arrivee',
  'pressure_arrivee',
  'depart_station_meteo',
  'arrival_station_meteo',
  'depart_station_distance_km',
  'arrival_station_distance_km',
  'depart_no_station',
  'arrival_no_station',
  'depart_weather_offset_hours',
  'arrival_weather_offset_hours',
];

const DEPARTURE_ID_FIELDS = ['gare_depart', 'depart_station_id', 'origin_station_id', 'PTCAR_NO', 'PTCAR ID'];
const ARRIVAL_ID_FIELDS = ['gare_arrivee', 'arrival_station_id', 'destination_station_id', 'PTCAR_NO', 'PTCAR ID'];
const DATE_FIELDS = ['date', 'jour', 'DATDEP', 'REAL_DATE_DEP', 'PLANNED_DATE_DEP', 'REAL_DATE_ARR', 'PLANNED_DATE_ARR'];
const TIME_FIELDS = ['heure', 'heure_depart', 'departure_time', 'REAL_TIME_DEP', 'PLANNED_TIME_DEP', 'REAL_TIME_ARR', 'PLANNED_TIME_ARR'];
const DATE_PRIORITY = {
  departure: ['REAL_DATE_DEP', 'PLANNED_DATE_DEP', 'DATDEP', 'REAL_DATE_ARR', 'PLANNED_DATE_ARR'],
  arrival: ['REAL_DATE_ARR', 'PLANNED_DATE_ARR', 'REAL_DATE_DEP', 'PLANNED_DATE_DEP', 'DATDEP'],
};
const TIME_PRIORITY = {
  departure: ['REAL_TIME_DEP', 'PLANNED_TIME_DEP', 'REAL_TIME_ARR', 'PLANNED_TIME_ARR'],
  arrival: ['REAL_TIME_ARR', 'PLANNED_TIME_ARR', 'REAL_TIME_DEP', 'PLANNED_TIME_DEP'],
};

function enrichRow(row, mapping, synopIndex, timezone) {
  const departId = sanitizeId(pickField(row, DEPARTURE_ID_FIELDS));
  const arrivalIdRaw = sanitizeId(pickField(row, ARRIVAL_ID_FIELDS));
  const arrivalId = arrivalIdRaw || departId;

  const result = { ...row };

  const departInfo = departId ? mapping[departId] : null;
  const arrivalInfo = arrivalId ? mapping[arrivalId] : null;

  if (!departInfo) {
    result.depart_no_station = true;
  }
  if (!arrivalInfo) {
    result.arrival_no_station = true;
  }

  const departDateTime = resolveDateTime(row, 'departure');
  const arrivalDateTime = resolveDateTime(row, 'arrival') || departDateTime;

  const departHourKey = departDateTime ? safeHourKey(departDateTime, timezone) : null;
  const arrivalHourKey = arrivalDateTime ? safeHourKey(arrivalDateTime, timezone) : departHourKey;

  // Essayer d'abord avec la station assignée
  let departLookup = lookupWeatherWithFallback(departInfo, departHourKey, synopIndex);
  let arrivalLookup = lookupWeatherWithFallback(arrivalInfo, arrivalHourKey, synopIndex);
  
  // Si pas trouvé, chercher dans toutes les stations météo disponibles
  if (!departLookup.weather && departHourKey) {
    departLookup = lookupWeatherInAllStations(departHourKey, synopIndex);
  }
  if (!arrivalLookup.weather && arrivalHourKey) {
    arrivalLookup = lookupWeatherInAllStations(arrivalHourKey, synopIndex);
  }

  const departWeather = departLookup.weather;
  const arrivalWeather = arrivalLookup.weather;

  Object.assign(result, {
    temp_depart: departWeather?.temp ?? null,
    precip_depart: departWeather?.precip ?? null,
    wind_speed_depart: departWeather?.wind_speed ?? null,
    humidity_depart: departWeather?.humidity ?? null,
    pressure_depart: departWeather?.pressure ?? null,
    temp_arrivee: arrivalWeather?.temp ?? null,
    precip_arrivee: arrivalWeather?.precip ?? null,
    wind_speed_arrivee: arrivalWeather?.wind_speed ?? null,
    humidity_arrivee: arrivalWeather?.humidity ?? null,
    pressure_arrivee: arrivalWeather?.pressure ?? null,
    depart_station_meteo: departInfo?.station_id ?? null,
    arrival_station_meteo: arrivalInfo?.station_id ?? null,
    depart_station_distance_km: departInfo?.distance_km ?? null,
    arrival_station_distance_km: arrivalInfo?.distance_km ?? null,
    depart_no_station: result.depart_no_station ?? departInfo?.no_station ?? false,
    arrival_no_station: result.arrival_no_station ?? arrivalInfo?.no_station ?? false,
    depart_weather_offset_hours: departLookup.offsetHours,
    arrival_weather_offset_hours: arrivalLookup.offsetHours,
  });

  let missingWeatherCount = 0;
  if (!departWeather) missingWeatherCount += 1;
  if (!arrivalWeather) missingWeatherCount += 1;
  result._missingWeatherCount = missingWeatherCount;

  return result;
}

function resolveDateTime(row, type) {
  const dateFields = DATE_PRIORITY[type] || DATE_FIELDS;
  const timeFields = TIME_PRIORITY[type] || TIME_FIELDS;
  const dateRaw = pickField(row, dateFields);
  const timeRaw = pickField(row, timeFields);
  const normalizedDate = normalizePunctualityDate(dateRaw);
  const normalizedTime = normalizePunctualityTime(timeRaw);
  if (!normalizedDate) {
    return null;
  }
  return `${normalizedDate} ${normalizedTime || '00:00:00'}`;
}

function safeHourKey(dateTimeString, timezone) {
  try {
    return floorToUtcHourKey(dateTimeString, { timezone });
  } catch (err) {
    if (process.env.DEBUG) {
      console.warn('Date invalide:', dateTimeString, err.message);
    }
    return null;
  }
}

function lookupWeatherWithFallback(mappingInfo, hourKey, synopIndex, maxFallbackHours = 720) {
  if (!mappingInfo || !mappingInfo.station_id) {
    return { weather: null, offsetHours: null };
  }
  const stationData = synopIndex[mappingInfo.station_id];
  if (!stationData) {
    return { weather: null, offsetHours: null };
  }
  if (!hourKey) {
    return { weather: null, offsetHours: null };
  }

  if (stationData[hourKey]) {
    return { weather: stationData[hourKey], offsetHours: 0 };
  }

  const baseDate = parseHourKey(hourKey);
  if (!baseDate) {
    return { weather: null, offsetHours: null };
  }

  // Recherche par paliers : ±1h, ±2h, ±3h, ±6h, ±12h, ±24h, ±48h, ±72h, ±168h
  const fallbackOffsets = [1, 2, 3, 6, 12, 24, 48, 72, 168];
  
  // D'abord, chercher par paliers (plus rapide)
  for (const delta of fallbackOffsets) {
    if (delta > maxFallbackHours) break;
    
    // Cherche d'abord dans le passé, puis dans le futur
    const prevKey = formatHourKey(addHours(baseDate, -delta));
    if (prevKey && stationData[prevKey]) {
      return { weather: stationData[prevKey], offsetHours: -delta };
    }
    const nextKey = formatHourKey(addHours(baseDate, delta));
    if (nextKey && stationData[nextKey]) {
      return { weather: stationData[nextKey], offsetHours: delta };
    }
  }

  // Si pas trouvé par paliers, chercher la première heure disponible dans chaque direction
  // (pour gérer les cas où les données ne sont pas aux heures exactes des paliers)
  let closestWeather = null;
  let closestOffset = null;
  let minDistance = Infinity;

  // Chercher dans le futur (jusqu'à maxFallbackHours)
  for (let delta = 1; delta <= maxFallbackHours; delta += 1) {
    const nextKey = formatHourKey(addHours(baseDate, delta));
    if (nextKey && stationData[nextKey]) {
      closestWeather = stationData[nextKey];
      closestOffset = delta;
      minDistance = delta;
      break; // Prendre la première trouvée (la plus proche)
    }
  }

  // Chercher dans le passé (jusqu'à maxFallbackHours)
  for (let delta = 1; delta <= maxFallbackHours; delta += 1) {
    const prevKey = formatHourKey(addHours(baseDate, -delta));
    if (prevKey && stationData[prevKey]) {
      const distance = delta;
      if (distance < minDistance) {
        closestWeather = stationData[prevKey];
        closestOffset = -delta;
        minDistance = distance;
      }
      break; // Prendre la première trouvée (la plus proche)
    }
  }

  if (closestWeather) {
    return { weather: closestWeather, offsetHours: closestOffset };
  }

  return { weather: null, offsetHours: null };
}

function lookupWeatherInAllStations(hourKey, synopIndex, maxFallbackHours = 720) {
  // Si pas de clé horaire, retourner null
  if (!hourKey) {
    return { weather: null, offsetHours: null };
  }

  const baseDate = parseHourKey(hourKey);
  if (!baseDate) {
    return { weather: null, offsetHours: null };
  }

  // Chercher dans toutes les stations météo
  const allStations = Object.keys(synopIndex);
  let closestWeather = null;
  let closestOffset = null;
  let minDistance = Infinity;

  for (const stationId of allStations) {
    const stationData = synopIndex[stationId];
    if (!stationData) continue;

    // Vérifier l'heure exacte d'abord
    if (stationData[hourKey]) {
      const distance = 0;
      if (distance < minDistance) {
        minDistance = distance;
        closestWeather = stationData[hourKey];
        closestOffset = 0;
      }
      continue;
    }

    // Chercher par paliers
    const fallbackOffsets = [1, 2, 3, 6, 12, 24, 48, 72, 168, 336, 504, 720];
    for (const delta of fallbackOffsets) {
      if (delta > maxFallbackHours) break;
      
      const prevKey = formatHourKey(addHours(baseDate, -delta));
      if (prevKey && stationData[prevKey]) {
        if (delta < minDistance) {
          minDistance = delta;
          closestWeather = stationData[prevKey];
          closestOffset = -delta;
        }
        break;
      }
      
      const nextKey = formatHourKey(addHours(baseDate, delta));
      if (nextKey && stationData[nextKey]) {
        if (delta < minDistance) {
          minDistance = delta;
          closestWeather = stationData[nextKey];
          closestOffset = delta;
        }
        break;
      }
    }

    // Si pas trouvé par paliers et qu'on n'a toujours rien, chercher heure par heure jusqu'à 720h
    if (!closestWeather) {
      for (let delta = 169; delta <= Math.min(720, maxFallbackHours); delta += 1) {
        const nextKey = formatHourKey(addHours(baseDate, delta));
        if (nextKey && stationData[nextKey]) {
          if (delta < minDistance) {
            minDistance = delta;
            closestWeather = stationData[nextKey];
            closestOffset = delta;
          }
          break;
        }
        
        const prevKey = formatHourKey(addHours(baseDate, -delta));
        if (prevKey && stationData[prevKey]) {
          if (delta < minDistance) {
            minDistance = delta;
            closestWeather = stationData[prevKey];
            closestOffset = -delta;
          }
          break;
        }
      }
    }
  }

  if (closestWeather) {
    return { weather: closestWeather, offsetHours: closestOffset };
  }

  return { weather: null, offsetHours: null };
}

function pickField(row, candidates) {
  for (const field of candidates) {
    if (row[field] !== undefined && row[field] !== '') {
      return row[field];
    }
  }
  return undefined;
}

function sanitizeId(value) {
  if (value === undefined || value === null) return undefined;
  const trimmed = String(value).trim();
  return trimmed === '' ? undefined : trimmed;
}

function normalizePunctualityDate(value) {
  if (!value) return null;
  const trimmed = String(value).trim();
  if (!trimmed) return null;

  if (/^\d{4}-\d{2}-\d{2}$/.test(trimmed)) {
    return trimmed;
  }

  if (/^\d{8}$/.test(trimmed)) {
    return `${trimmed.slice(0, 4)}-${trimmed.slice(4, 6)}-${trimmed.slice(6, 8)}`;
  }

  if (/^\d{2}[A-Z]{3}\d{4}$/.test(trimmed)) {
    const day = trimmed.slice(0, 2);
    const month = trimmed.slice(2, 5);
    const year = trimmed.slice(5, 9);
    const monthIndex = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'].indexOf(month);
    if (monthIndex >= 0) {
      return `${year}-${String(monthIndex + 1).padStart(2, '0')}-${day}`;
    }
  }

  const parsed = new Date(trimmed);
  if (!Number.isNaN(parsed.getTime())) {
    return `${parsed.getFullYear()}-${String(parsed.getMonth() + 1).padStart(2, '0')}-${String(parsed.getDate()).padStart(2, '0')}`;
  }

  return null;
}

function normalizePunctualityTime(value) {
  if (!value) return null;
  let trimmed = String(value).trim();
  if (!trimmed) return null;

  if (/^\d{2}:\d{2}:\d{2}$/.test(trimmed)) {
    return trimmed;
  }
  if (/^\d{2}:\d{2}$/.test(trimmed)) {
    return `${trimmed}:00`;
  }
  if (/^\d{4}$/.test(trimmed)) {
    return `${trimmed.slice(0, 2)}:${trimmed.slice(2, 4)}:00`;
  }
  if (/^\d{2}$/.test(trimmed)) {
    return `${trimmed}:00:00`;
  }
  if (/^\d{6}$/.test(trimmed)) {
    return `${trimmed.slice(0, 2)}:${trimmed.slice(2, 4)}:${trimmed.slice(4, 6)}`;
  }

  return null;
}

function parseHourKey(key) {
  if (!key || !/^\d{4}-\d{2}-\d{2}-\d{2}$/.test(key)) {
    return null;
  }
  const [yearStr, monthStr, dayStr, hourStr] = key.split('-');
  const hour = parseInt(hourStr, 10);
  if (Number.isNaN(hour)) {
    return null;
  }
  const date = new Date(`${yearStr}-${monthStr}-${dayStr}T${String(hour).padStart(2, '0')}:00:00Z`);
  if (Number.isNaN(date.getTime())) {
    return null;
  }
  return date;
}

function addHours(date, hours) {
  const result = new Date(date.getTime());
  result.setUTCHours(result.getUTCHours() + hours);
  return result;
}

function formatHourKey(date) {
  if (!date) return null;
  const year = date.getUTCFullYear();
  const month = String(date.getUTCMonth() + 1).padStart(2, '0');
  const day = String(date.getUTCDate()).padStart(2, '0');
  const hour = String(date.getUTCHours()).padStart(2, '0');
  return `${year}-${month}-${day}-${hour}`;
}

module.exports = {
  OUTPUT_COLUMNS,
  DEPARTURE_ID_FIELDS,
  ARRIVAL_ID_FIELDS,
  DATE_FIELDS,
  TIME_FIELDS,
  DATE_PRIORITY,
  TIME_PRIORITY,
  enrichRow,
  sanitizeId,
  normalizePunctualityDate,
  normalizePunctualityTime,
  lookupWeatherWithFallback,
  lookupWeatherInAllStations,
};

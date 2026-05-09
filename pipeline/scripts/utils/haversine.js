'use strict';

const EARTH_RADIUS_KM = 6371;

/**
 * Calcule la distance entre deux points géographiques en kilomètres
 * à partir de leurs latitudes/longitudes en degrés.
 *
 * @param {{ lat: number, lon: number }} pointA
 * @param {{ lat: number, lon: number }} pointB
 * @returns {number} distance en kilomètres
 */
function haversineDistance(pointA, pointB) {
  if (!pointA || !pointB) {
    throw new Error('Les deux points doivent être définis');
  }

  const lat1 = toRadians(pointA.lat);
  const lon1 = toRadians(pointA.lon);
  const lat2 = toRadians(pointB.lat);
  const lon2 = toRadians(pointB.lon);

  const dLat = lat2 - lat1;
  const dLon = lon2 - lon1;

  const a = Math.sin(dLat / 2) ** 2 + Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2) ** 2;
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

  return EARTH_RADIUS_KM * c;
}

function toRadians(degrees) {
  return (degrees * Math.PI) / 180;
}

module.exports = {
  EARTH_RADIUS_KM,
  haversineDistance,
};

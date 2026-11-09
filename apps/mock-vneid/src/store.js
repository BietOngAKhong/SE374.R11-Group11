const path = require('path');
const fs = require('fs');
const crypto = require('crypto');

const citizensPath = path.resolve(__dirname, '../data/citizens.json');
const facilitiesPath = path.resolve(__dirname, '../data/medical_facilities.json');
const practitionersPath = path.resolve(__dirname, '../data/medical_practitioners.json');

const citizens = JSON.parse(fs.readFileSync(citizensPath, 'utf8'));
const facilities = JSON.parse(fs.readFileSync(facilitiesPath, 'utf8'));
const practitioners = JSON.parse(fs.readFileSync(practitionersPath, 'utf8'));

const authCodes = new Map();

function createAuthCode({ citizen, clientId, redirectUri, scope, state, nonce }) {
  const code = crypto.randomBytes(24).toString('hex');
  const record = {
    code,
    citizen,
    clientId,
    redirectUri,
    scope: scope || 'openid profile cccd bhyt',
    state: state || '',
    nonce: nonce || '',
    expiresAt: Date.now() + 10 * 60 * 1000
  };
  authCodes.set(code, record);
  return record;
}

function consumeAuthCode(code) {
  const record = authCodes.get(code);
  if (!record) {
    return null;
  }
  authCodes.delete(code);
  if (Date.now() > record.expiresAt) {
    return null;
  }
  return record;
}

function findCitizen(cccd) {
  return citizens.find(c => c.cccd === cccd) || null;
}

function findPractitioner(cchn) {
  return practitioners.find(p => p.ma_cchn.toLowerCase() === cchn.toLowerCase()) || null;
}

function findPractitionerByCccd(cccd) {
  return practitioners.find(p => p.cccd === cccd) || null;
}

function getFacilities() {
  return facilities;
}

function findFacility(maBv) {
  return facilities.find(f => f.ma_bv === maBv) || null;
}

function getCitizens() {
  return citizens;
}

function getPractitioners() {
  return practitioners;
}

module.exports = {
  createAuthCode,
  consumeAuthCode,
  findCitizen,
  findPractitioner,
  findPractitionerByCccd,
  getFacilities,
  findFacility,
  getCitizens,
  getPractitioners
};


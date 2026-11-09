const crypto = require('crypto');
const config = require('./config');

const { privateKey, publicKey } = crypto.generateKeyPairSync('rsa', { modulusLength: 2048 });
const exportedJwk = publicKey.export({ format: 'jwk' });

const publicJwk = {
  kty: exportedJwk.kty,
  n: exportedJwk.n,
  e: exportedJwk.e,
  alg: 'RS256',
  use: 'sig',
  kid: 'vneid-mock-key-1'
};

function getJwks() {
  return {
    keys: [publicJwk]
  };
}

function signJwt(payload, subject, audience, expiresInSeconds) {
  const now = Math.floor(Date.now() / 1000);
  const header = {
    alg: 'RS256',
    typ: 'JWT',
    kid: publicJwk.kid
  };
  const claims = {
    ...payload,
    iss: config.issuer,
    sub: subject,
    aud: audience,
    iat: now,
    exp: now + expiresInSeconds
  };
  const encodedHeader = Buffer.from(JSON.stringify(header)).toString('base64url');
  const encodedPayload = Buffer.from(JSON.stringify(claims)).toString('base64url');
  const signingInput = `${encodedHeader}.${encodedPayload}`;

  const signer = crypto.createSign('RSA-SHA256');
  signer.update(signingInput);
  const signature = signer.sign(privateKey).toString('base64url');

  return `${signingInput}.${signature}`;
}

module.exports = {
  getJwks,
  signJwt,
  publicJwk
};


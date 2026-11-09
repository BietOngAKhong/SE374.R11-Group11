const express = require('express');
const crypto = require('crypto');
const config = require('../config');
const store = require('../store');
const keys = require('../keys');
const { renderLoginPage } = require('../views/renderLogin');

const router = express.Router();

router.get('/oauth/authorize', (req, res) => {
  const { client_id, redirect_uri, state, scope, nonce } = req.query;
  const html = renderLoginPage({
    clientId: client_id || config.clientId,
    redirectUri: redirect_uri || config.defaultRedirectUri,
    state: state || '',
    scope: scope || 'openid profile cccd bhyt',
    nonce: nonce || '',
    citizens: store.getCitizens(),
    practitioners: store.getPractitioners(),
    facilities: store.getFacilities()
  });
  res.setHeader('Content-Type', 'text/html; charset=utf-8');
  res.send(html);
});

router.post('/oauth/authorize', (req, res) => {
  const { cccd, client_id, redirect_uri, state, scope, nonce } = req.body;
  const targetRedirect = redirect_uri || config.defaultRedirectUri;

  const citizen = store.findCitizen((cccd || '').trim());
  if (!citizen) {
    const html = renderLoginPage({
      clientId: client_id || config.clientId,
      redirectUri: targetRedirect,
      state: state || '',
      scope: scope || '',
      nonce: nonce || '',
      citizens: store.getCitizens(),
      practitioners: store.getPractitioners(),
      facilities: store.getFacilities(),
      error: `Số định danh ${cccd} không tồn tại trong CSDL dân cư mẫu.`
    });
    return res.status(400).send(html);
  }

  const record = store.createAuthCode({
    citizen,
    clientId: client_id || config.clientId,
    redirectUri: targetRedirect,
    scope,
    state,
    nonce
  });

  const redirectUrl = new URL(targetRedirect);
  redirectUrl.searchParams.set('code', record.code);
  if (state) {
    redirectUrl.searchParams.set('state', state);
  }

  return res.redirect(302, redirectUrl.toString());
});

router.post('/oauth/token', async (req, res) => {
  const grantType = req.body.grant_type || req.query.grant_type;
  const code = req.body.code || req.query.code;
  const clientId = req.body.client_id || req.query.client_id;
  const clientSecret = req.body.client_secret || req.query.client_secret;

  if (grantType !== 'authorization_code') {
    return res.status(400).json({
      error: 'unsupported_grant_type',
      error_description: 'Only authorization_code grant type is supported'
    });
  }

  if (!code) {
    return res.status(400).json({
      error: 'invalid_request',
      error_description: 'Missing code parameter'
    });
  }

  if (clientSecret && clientSecret !== config.clientSecret) {
    return res.status(401).json({
      error: 'invalid_client',
      error_description: 'Client authentication failed'
    });
  }

  const record = store.consumeAuthCode(code);
  if (!record) {
    return res.status(400).json({
      error: 'invalid_grant',
      error_description: 'Authorization code is invalid or has expired'
    });
  }

  const citizen = record.citizen;
  const practitioner = store.findPractitionerByCccd(citizen.cccd);

  const baseClaims = {
    cccd: citizen.cccd,
    name: citizen.ho_va_ten,
    birthdate: citizen.ngay_sinh,
    gender: citizen.gioi_tinh,
    phone_number: citizen.so_dien_thoai,
    address: citizen.dia_chi_thuong_tru,
    bhyt_number: citizen.so_the_bhyt,
    bhyt_rate: citizen.muc_huong_bhyt,
    roles: practitioner ? ['Doctor', 'Citizen'] : ['Citizen']
  };

  if (practitioner) {
    baseClaims.cchn = practitioner.ma_cchn;
    baseClaims.facility_id = practitioner.ma_cskcb_cong_tac;
    baseClaims.scope_of_practice = practitioner.pham_vi_hoat_dong;
    baseClaims.practitioner = {
      cchn: practitioner.ma_cchn,
      facility_id: practitioner.ma_cskcb_cong_tac,
      scope_of_practice: practitioner.pham_vi_hoat_dong,
      status: practitioner.trang_thai
    };
  }

  if (record.nonce) {
    baseClaims.nonce = record.nonce;
  }

  const accessToken = keys.signJwt(
    {
      ...baseClaims,
      token_use: 'access_token',
      scope: record.scope
    },
    citizen.cccd,
    clientId || config.clientId,
    config.tokenExpiresIn
  );

  const idToken = keys.signJwt(
    {
      ...baseClaims,
      token_use: 'id_token'
    },
    citizen.cccd,
    clientId || config.clientId,
    config.tokenExpiresIn
  );

  const refreshToken = crypto.randomBytes(32).toString('hex');

  return res.json({
    access_token: accessToken,
    id_token: idToken,
    refresh_token: refreshToken,
    token_type: 'Bearer',
    expires_in: config.tokenExpiresIn
  });
});

router.get('/.well-known/jwks.json', (req, res) => {
  res.json(keys.getJwks());
});

router.get('/.well-known/openid-configuration', (req, res) => {
  res.json({
    issuer: config.issuer,
    authorization_endpoint: `${config.issuer}/oauth/authorize`,
    token_endpoint: `${config.issuer}/oauth/token`,
    jwks_uri: `${config.issuer}/.well-known/jwks.json`,
    response_types_supported: ['code'],
    subject_types_supported: ['public'],
    id_token_signing_alg_values_supported: ['RS256'],
    scopes_supported: ['openid', 'profile', 'cccd', 'bhyt']
  });
});

module.exports = router;


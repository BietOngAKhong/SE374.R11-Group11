const path = require('path');
const dotenv = require('dotenv');

dotenv.config({ path: path.resolve(__dirname, '../.env') });

module.exports = {
  port: parseInt(process.env.PORT || '5001', 10),
  host: process.env.HOST || '0.0.0.0',
  issuer: process.env.ISSUER || 'http://localhost:5001',
  clientId: process.env.CLIENT_ID || 'neuroscan-client',
  clientSecret: process.env.CLIENT_SECRET || 'neuroscan-secret-key-123456',
  defaultRedirectUri: process.env.DEFAULT_REDIRECT_URI || 'http://localhost:3000/api/auth/callback',
  tokenExpiresIn: parseInt(process.env.TOKEN_EXPIRES_IN || '3600', 10),
  refreshTokenExpiresIn: parseInt(process.env.REFRESH_TOKEN_EXPIRES_IN || '2592000', 10)
};


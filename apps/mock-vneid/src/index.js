const app = require('./app');
const config = require('./config');

const server = app.listen(config.port, config.host, () => {
  console.log(`Mock VNeID Server running at http://${config.host}:${config.port}`);
  console.log(`- Authorize Endpoint: http://localhost:${config.port}/oauth/authorize`);
  console.log(`- Token Endpoint:     http://localhost:${config.port}/oauth/token`);
  console.log(`- JWKS Endpoint:      http://localhost:${config.port}/.well-known/jwks.json`);
  console.log(`- C06 Citizen API:    http://localhost:${config.port}/api/v1/c06/citizen/:cccd`);
  console.log(`- BYT Practitioner:   http://localhost:${config.port}/api/v1/byt/practitioner/:cchn`);
  console.log(`- BYT Facilities:     http://localhost:${config.port}/api/v1/byt/facilities`);
});

process.on('SIGTERM', () => {
  server.close(() => {
    process.exit(0);
  });
});


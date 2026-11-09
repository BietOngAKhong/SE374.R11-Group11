const express = require('express');
const cors = require('cors');
const path = require('path');
const store = require('./store');
const oauthRouter = require('./routes/oauth');
const c06Router = require('./routes/c06');
const bytRouter = require('./routes/byt');

const app = express();

app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use('/assets', express.static(path.resolve(__dirname, '../assets')));
app.use('/_next/static/media', express.static(path.resolve(__dirname, '../assets')));

app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    service: 'mock-vneid',
    uptime: process.uptime(),
    timestamp: new Date().toISOString(),
    dataset: {
      citizens: store.getCitizens().length,
      practitioners: store.getPractitioners().length,
      facilities: store.getFacilities().length
    }
  });
});

app.use('/', oauthRouter);
app.use('/', c06Router);
app.use('/', bytRouter);

app.use((req, res) => {
  res.status(404).json({
    error: 'endpoint_not_found',
    message: `Cannot ${req.method} ${req.path}`
  });
});

app.use((err, req, res, next) => {
  res.status(500).json({
    error: 'internal_server_error',
    message: err.message || 'An unexpected error occurred'
  });
});

module.exports = app;


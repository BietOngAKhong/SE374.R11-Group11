const express = require('express');
const store = require('../store');

const router = express.Router();

router.get('/api/v1/c06/citizen/:cccd', (req, res) => {
  const cccd = (req.params.cccd || '').trim();
  const citizen = store.findCitizen(cccd);

  if (!citizen) {
    return res.status(404).json({
      error: 'citizen_not_found',
      message: `Citizen with CCCD ${cccd} was not found in National Population Database (C06)`
    });
  }

  return res.json(citizen);
});

router.get('/api/v1/c06/citizens', (req, res) => {
  return res.json({
    total: store.getCitizens().length,
    citizens: store.getCitizens()
  });
});

module.exports = router;


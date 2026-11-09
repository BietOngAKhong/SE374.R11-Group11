const express = require('express');
const store = require('../store');

const router = express.Router();

router.get('/api/v1/byt/practitioner/:cchn(*)', (req, res) => {
  const cchn = decodeURIComponent(req.params.cchn || '').trim();
  const practitioner = store.findPractitioner(cchn);

  if (!practitioner) {
    return res.status(404).json({
      error: 'practitioner_not_found',
      message: `Practitioner with CCHN ${cchn} was not found in Ministry of Health Database (BYT)`
    });
  }

  const facility = store.findFacility(practitioner.ma_cskcb_cong_tac);

  return res.json({
    ...practitioner,
    co_so_cong_tac: facility ? {
      ma_bv: facility.ma_bv,
      ten_bv: facility.ten_bv,
      hang_bv: facility.hang_bv
    } : null
  });
});

router.get('/api/v1/byt/practitioners', (req, res) => {
  return res.json({
    total: store.getPractitioners().length,
    practitioners: store.getPractitioners()
  });
});

router.get('/api/v1/byt/facilities', (req, res) => {
  return res.json(store.getFacilities());
});

router.get('/api/v1/byt/facilities/:ma_bv', (req, res) => {
  const maBv = (req.params.ma_bv || '').trim();
  const facility = store.findFacility(maBv);

  if (!facility) {
    return res.status(404).json({
      error: 'facility_not_found',
      message: `Medical facility with code ${maBv} was not found`
    });
  }

  return res.json(facility);
});

module.exports = router;


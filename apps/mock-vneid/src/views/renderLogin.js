const fs = require('fs');
const path = require('path');

const loginHtmlTemplate = fs.readFileSync(path.resolve(__dirname, 'login.html'), 'utf8');

function renderLoginPage({ clientId, redirectUri, state, scope, nonce, citizens, practitioners, facilities, error }) {
  const facList = facilities || [];
  const dropdownItems = citizens.map(c => {
    const pract = practitioners.find(p => p.cccd === c.cccd);
    const facility = pract ? facList.find(f => f.ma_bv === pract.ma_cskcb_cong_tac) : null;
    const facilityName = facility ? facility.ten_bv : (pract ? `Mã BV: ${pract.ma_cskcb_cong_tac}` : null);

    const roleBadge = pract
      ? `BS • ${pract.pham_vi_hoat_dong}`
      : `Công dân • Không có CCHN`;

    const subText = pract
      ? `CCCD: <span class="font-mono">${c.cccd}</span> • <strong>${facilityName}</strong> • CCHN: ${pract.ma_cchn}`
      : `CCCD: <span class="font-mono">${c.cccd}</span> • BHYT: ${c.so_the_bhyt || 'N/A'} (Không có CCHN)`;

    return `<div class="combobox-item" data-cccd="${c.cccd}"><div class="combobox-item-name">${c.ho_va_ten} <span class="combobox-item-badge">${roleBadge}</span></div><div class="combobox-item-sub">${subText}</div></div>`;
  }).join('');

  const errorSection = error
    ? `<div class="error-box">${error}</div>`
    : '';

  return loginHtmlTemplate
    .replace('{{CLIENT_ID}}', clientId || '')
    .replace('{{REDIRECT_URI}}', redirectUri || '')
    .replace('{{STATE}}', state || '')
    .replace('{{SCOPE}}', scope || '')
    .replace('{{NONCE}}', nonce || '')
    .replace('{{ERROR_SECTION}}', errorSection)
    .replace('{{DROPDOWN_ITEMS}}', dropdownItems);
}

module.exports = {
  renderLoginPage
};

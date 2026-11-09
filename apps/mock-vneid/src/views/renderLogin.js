const fs = require('fs');
const path = require('path');

const loginHtmlTemplate = fs.readFileSync(path.resolve(__dirname, 'login.html'), 'utf8');

function renderLoginPage({ clientId, redirectUri, state, scope, nonce, citizens, practitioners, error }) {
  const dropdownItems = citizens.map(c => {
    const pract = practitioners.find(p => p.cccd === c.cccd);
    const roleBadge = pract
      ? `BS • ${pract.pham_vi_hoat_dong}`
      : `Bệnh nhân BHYT ${c.muc_huong_bhyt}%`;

    return `<div class="combobox-item" data-cccd="${c.cccd}"><div class="combobox-item-name">${c.ho_va_ten} <span class="combobox-item-badge">${roleBadge}</span></div><div class="combobox-item-sub">CCCD: <span class="font-mono">${c.cccd}</span></div></div>`;
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

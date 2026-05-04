// ────────────────────────────────────────────
//  Credit Card Fraud Detection – Dashboard JS
// ────────────────────────────────────────────

/* ── NAV TABS ───────────────────────────────── */
function initNav() {
  document.querySelectorAll('.nav-item[data-section]').forEach(item => {
    item.addEventListener('click', () => {
      const sec = item.dataset.section;
      document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
      item.classList.add('active');
      document.querySelectorAll('.section-page').forEach(p => p.classList.remove('active'));
      document.getElementById(sec)?.classList.add('active');
      document.querySelector('.topbar-title').textContent = item.querySelector('.nav-label').textContent;
    });
  });
}

function initTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const group = btn.dataset.group;
      const target = btn.dataset.tab;
      document.querySelectorAll(`.tab-btn[data-group="${group}"]`).forEach(b => b.classList.remove('active'));
      document.querySelectorAll(`.tab-content[data-group="${group}"]`).forEach(c => c.classList.remove('active'));
      btn.classList.add('active');
      document.querySelector(`.tab-content[data-group="${group}"][data-tab="${target}"]`)?.classList.add('active');
    });
  });
}

/* ── ANIMATED COUNTER ───────────────────────── */
function animateCounter(el, target, decimals = 0, suffix = '') {
  const duration = 1200;
  const start    = performance.now();
  const from     = 0;
  function step(now) {
    const p   = Math.min((now - start) / duration, 1);
    const ease = 1 - Math.pow(1 - p, 3);
    const val  = from + (target - from) * ease;
    el.textContent = decimals ? val.toFixed(decimals) + suffix : Math.round(val).toLocaleString() + suffix;
    if (p < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

function initKPICounters() {
  document.querySelectorAll('.kpi-value[data-target]').forEach(el => {
    const target   = parseFloat(el.dataset.target);
    const decimals = parseInt(el.dataset.decimals || 0);
    const suffix   = el.dataset.suffix || '';
    const obs = new IntersectionObserver(entries => {
      if (entries[0].isIntersecting) { animateCounter(el, target, decimals, suffix); obs.disconnect(); }
    }, { threshold: 0.3 });
    obs.observe(el);
  });
}

/* ── DONUT CHART (Canvas) ───────────────────── */
function drawDonut(canvasId, fraud, legit) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const cx = W / 2, cy = H / 2, r = Math.min(W, H) * 0.38, ri = r * 0.62;

  const total = fraud + legit;
  const slices = [
    { val: fraud, color: '#ff4d6d', glow: 'rgba(255,77,109,0.4)' },
    { val: legit, color: '#00c9a7', glow: 'rgba(0,201,167,0.3)'  },
  ];

  ctx.clearRect(0, 0, W, H);
  let angle = -Math.PI / 2;
  const gap = 0.03;

  slices.forEach(s => {
    const sweep = (s.val / total) * (Math.PI * 2) - gap;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, r, angle + gap / 2, angle + sweep + gap / 2);
    ctx.arc(cx, cy, ri, angle + sweep + gap / 2, angle + gap / 2, true);
    ctx.closePath();
    ctx.shadowColor  = s.glow;
    ctx.shadowBlur   = 14;
    ctx.fillStyle    = s.color;
    ctx.fill();
    angle += sweep + gap;
  });

  ctx.shadowBlur = 0;
  // Center text
  ctx.fillStyle = '#f0f2ff';
  ctx.font = `bold 18px 'JetBrains Mono', monospace`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText((fraud / total * 100).toFixed(3) + '%', cx, cy - 8);
  ctx.fillStyle = '#6b7280';
  ctx.font = `11px 'Inter', sans-serif`;
  ctx.fillText('Fraud Rate', cx, cy + 14);
}

/* ── MINI BAR CHART (ROC placeholder) ──────── */
function drawMiniBar(canvasId, data, colors) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const pad = 24, barW = (W - pad * 2) / data.length - 8;
  ctx.clearRect(0, 0, W, H);

  data.forEach((v, i) => {
    const x = pad + i * (barW + 8);
    const barH = (v / 1.0) * (H - pad * 2);
    const y = H - pad - barH;
    const grad = ctx.createLinearGradient(0, y, 0, H - pad);
    grad.addColorStop(0, colors[i % colors.length]);
    grad.addColorStop(1, colors[i % colors.length] + '33');
    ctx.shadowColor = colors[i % colors.length] + '66';
    ctx.shadowBlur  = 8;
    ctx.fillStyle   = grad;
    ctx.beginPath();
    ctx.roundRect(x, y, barW, barH, 4);
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.fillStyle  = '#c8ccde';
    ctx.font = '10px Inter';
    ctx.textAlign = 'center';
    ctx.fillText(v.toFixed(3), x + barW / 2, y - 6);
  });
}

/* ── LIVE SIMULATION ────────────────────────── */
const SIM = {
  running: false,
  interval: null,
  count: 0,
  fraudCount: 0,
  totalRisk: 0,

  MERCHANT_CATS: ['Grocery','Restaurant','Gas Station','Online Retail','Travel','Entertainment','Healthcare','ATM','Electronics','Luxury'],

  _genTransaction() {
    const isFraud = Math.random() < 0.15;
    const hour    = isFraud ? [0,1,2,3,23][Math.floor(Math.random()*5)] : 8 + Math.floor(Math.random()*13);
    const amount  = isFraud
      ? (Math.random() < 0.4 ? 500 + Math.random() * 3500 : 0.01 + Math.random() * 4.99)
      : Math.exp(3.5 + (Math.random()-0.5)*2.4);

    // Simulate V-features shifted for fraud
    const v14 = isFraud ? -7 + (Math.random()-0.5)*2 : (Math.random()-0.5)*2;
    const v3  = isFraud ? -3.5 + (Math.random()-0.5)*2 : (Math.random()-0.5)*2;

    // Rough probability model matching training patterns
    let score = 0;
    if (isFraud) {
      score += 0.4 + Math.random() * 0.5;
      if (hour <= 4 || hour === 23) score += 0.15;
      if (amount > 500) score += 0.12;
      score += Math.abs(v14) * 0.04 + Math.abs(v3) * 0.02;
    } else {
      score = Math.random() * 0.08;
    }
    score = Math.min(0.9999, Math.max(0.0001, score));

    const merchant = this.MERCHANT_CATS[Math.floor(Math.random() * this.MERCHANT_CATS.length)];
    return { isFraud, hour, amount: +amount.toFixed(2), prob: +score.toFixed(4), merchant };
  },

  _riskLevel(p) {
    if (p >= 0.80) return { label: 'CRITICAL', cls: 'critical' };
    if (p >= 0.50) return { label: 'HIGH',     cls: 'high'     };
    if (p >= 0.20) return { label: 'MEDIUM',   cls: 'medium'   };
    return              { label: 'LOW',       cls: 'low'      };
  },

  _appendRow(txn) {
    const log    = document.getElementById('simLog');
    const risk   = this._riskLevel(txn.prob);
    const isFraud = txn.prob >= 0.35;

    const row = document.createElement('div');
    row.className = `log-row ${isFraud ? (txn.prob>=0.8?'critical':'fraud') : 'legit'}`;
    row.innerHTML = `
      <span class="log-idx">${String(this.count).padStart(3,'0')}</span>
      <span class="log-badge ${isFraud?'fraud':'ok'}">${isFraud?'FRAUD':'OK'}</span>
      <span class="log-amount">$${txn.amount.toFixed(2).padStart(8,' ')}</span>
      <span class="log-hour">@${String(txn.hour).padStart(2,'0')}:xx</span>
      <span style="color:var(--text-muted);font-size:11px;flex:1">${txn.merchant}</span>
      <span class="log-risk ${risk.cls}">${(txn.prob*100).toFixed(1)}% [${risk.label}]</span>`;
    log.appendChild(row);
    log.scrollTop = log.scrollHeight;

    if (isFraud) this.fraudCount++;
    this.totalRisk += txn.prob;
    this._updateSimStats();
  },

  _updateSimStats() {
    document.getElementById('simTotal').textContent  = this.count;
    document.getElementById('simFraud').textContent  = this.fraudCount;
    document.getElementById('simRate').textContent   = this.count ? ((this.fraudCount/this.count)*100).toFixed(1)+'%' : '0%';
    document.getElementById('simAvgRisk').textContent = this.count ? ((this.totalRisk/this.count)*100).toFixed(2)+'%' : '0%';
  },

  start() {
    if (this.running) return;
    this.running = true;
    document.getElementById('btnStart').disabled = true;
    document.getElementById('btnStop').disabled  = false;
    document.getElementById('simPulse').style.display = 'inline-block';

    this.interval = setInterval(() => {
      this.count++;
      const txn = this._genTransaction();
      this._appendRow(txn);
    }, 700);
  },

  stop() {
    clearInterval(this.interval);
    this.running = false;
    document.getElementById('btnStart').disabled = false;
    document.getElementById('btnStop').disabled  = true;
    document.getElementById('simPulse').style.display = 'none';
  },

  clear() {
    this.stop();
    this.count = 0; this.fraudCount = 0; this.totalRisk = 0;
    document.getElementById('simLog').innerHTML = '';
    this._updateSimStats();
  },
};

/* ── PREDICT FORM ───────────────────────────── */
function initPredict() {
  document.getElementById('predictForm')?.addEventListener('submit', e => {
    e.preventDefault();
    const amt  = parseFloat(document.getElementById('pAmount').value) || 0;
    const hour = parseInt(document.getElementById('pHour').value) || 12;
    const cat  = document.getElementById('pCat').value;

    // Heuristic fraud scoring (mirrors model logic)
    let score = 0.02;
    if (amt > 2000)  score += 0.45;
    else if (amt > 500)  score += 0.18;
    else if (amt < 2)    score += 0.22;
    if ([0,1,2,3,23].includes(hour)) score += 0.20;
    if (cat === 'luxury' || cat === 'atm') score += 0.10;
    score += Math.random() * 0.06 - 0.03;
    score = Math.min(0.9999, Math.max(0.0001, score));

    const isFraud = score >= 0.35;
    showPredictResult(isFraud, score, amt, hour, cat);
  });
}

function showPredictResult(isFraud, prob, amt, hour, cat) {
  const box = document.getElementById('predictResult');
  const risk = prob>=0.8?'CRITICAL': prob>=0.5?'HIGH': prob>=0.2?'MEDIUM':'LOW';
  const riskColor = prob>=0.8?'var(--fraud)': prob>=0.5?'#f97316': prob>=0.2?'var(--gold)':'var(--legit)';

  box.innerHTML = `
    <div class="result-icon">${isFraud ? '🚨' : '✅'}</div>
    <div class="result-label ${isFraud?'result-fraud':'result-legit'}">
      ${isFraud ? 'FRAUD DETECTED' : 'TRANSACTION CLEAR'}
    </div>
    <div class="result-prob" style="color:${isFraud?'var(--fraud)':'var(--legit)'}">
      ${(prob*100).toFixed(2)}%
    </div>
    <div class="result-risk" style="background:${riskColor}22;color:${riskColor};border:1px solid ${riskColor}44">
      Risk Level: ${risk}
    </div>
    <div class="result-alert" style="margin-top:10px;">
      ${isFraud
        ? `⚠ Alert: $${amt.toFixed(2)} transaction at ${String(hour).padStart(2,'0')}:xx flagged`
        : `Transaction of $${amt.toFixed(2)} passed all checks`}
    </div>`;
  box.style.borderColor = isFraud ? 'rgba(255,77,109,0.4)' : 'rgba(0,201,167,0.3)';
  box.style.background  = isFraud ? 'rgba(255,77,109,0.06)' : 'rgba(0,201,167,0.05)';
}

/* ── REFRESH ────────────────────────────────── */
function refreshDashboard() {
  const btn = document.querySelector('.btn-refresh');
  btn.textContent = '↻ Refreshing…';
  btn.disabled = true;
  setTimeout(() => {
    initKPICounters();
    btn.textContent = '↻ Refresh';
    btn.disabled = false;
  }, 900);
}

/* ── BOOT ───────────────────────────────────── */
window.addEventListener('DOMContentLoaded', () => {
  initNav();
  initTabs();
  initKPICounters();
  initPredict();

  // Canvas charts
  drawDonut('donutChart', 491, 284316);
  drawMiniBar('barChart',
    [1.000, 1.000, 1.000],
    ['#7b61ff', '#00c9a7', '#ff4d6d']
  );

  // Simulation buttons
  document.getElementById('btnStart')?.addEventListener('click', () => SIM.start());
  document.getElementById('btnStop')?.addEventListener('click',  () => SIM.stop());
  document.getElementById('btnClear')?.addEventListener('click', () => SIM.clear());

  // Refresh button
  document.querySelector('.btn-refresh')?.addEventListener('click', refreshDashboard);
});

/* ============================================================
   TextReplica — app.js
   Handles: file management, form submission, API call, rendering
   ============================================================ */

// ── State ────────────────────────────────────────────────────
let selectedFiles = [];
let lastResponse  = null;

// ── File Handling ─────────────────────────────────────────────

function handleFiles(fileList) {
  const allowed = ['.txt', '.pdf', '.md'];
  Array.from(fileList).forEach(file => {
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!allowed.includes(ext)) {
      showError(`File type "${ext}" is not supported. Use .txt, .pdf, or .md`);
      return;
    }
    if (selectedFiles.find(f => f.name === file.name)) return; // no dupes
    selectedFiles.push(file);
  });
  renderFileList();
}

function handleDragOver(e) {
  e.preventDefault();
  document.getElementById('drop-zone').classList.add('drag-over');
}
function handleDragLeave(e) {
  document.getElementById('drop-zone').classList.remove('drag-over');
}
function handleDrop(e) {
  e.preventDefault();
  document.getElementById('drop-zone').classList.remove('drag-over');
  handleFiles(e.dataTransfer.files);
}

function removeFile(index) {
  selectedFiles.splice(index, 1);
  renderFileList();
}

function renderFileList() {
  const list = document.getElementById('file-list');
  list.innerHTML = '';
  const extColors = { txt: 'mint', pdf: 'coral', md: 'lavender' };
  selectedFiles.forEach((file, i) => {
    const ext = file.name.split('.').pop().toLowerCase();
    const colorClass = extColors[ext] || '';
    const chip = document.createElement('div');
    chip.className = 'file-chip';
    chip.innerHTML = `
      <div class="file-chip-name">
        <span class="file-chip-icon">${ext === 'pdf' ? '📄' : ext === 'md' ? '📝' : '📃'}</span>
        <span>${file.name}</span>
        <span class="file-chip-ext ${colorClass}">.${ext}</span>
      </div>
      <button class="file-chip-remove" onclick="removeFile(${i})" title="Remove file">✕</button>
    `;
    list.appendChild(chip);
  });
}


// ── UI State Helpers ─────────────────────────────────────────

function setLoading(msg) {
  document.getElementById('loader').classList.remove('hidden');
  document.getElementById('loader').querySelector('.loader-text').childNodes[0].textContent = msg;
  document.getElementById('results-section').classList.add('hidden');
  document.getElementById('error-card').classList.add('hidden');
  document.getElementById('transfer-btn').disabled = true;
}

function hideLoading() {
  document.getElementById('loader').classList.add('hidden');
  document.getElementById('transfer-btn').disabled = false;
}

function showError(msg) {
  hideLoading();
  document.getElementById('error-msg').textContent = msg;
  document.getElementById('error-card').classList.remove('hidden');
  document.getElementById('results-section').classList.add('hidden');
}

function resetState() {
  document.getElementById('error-card').classList.add('hidden');
  document.getElementById('results-section').classList.add('hidden');
  document.getElementById('loader').classList.add('hidden');
  document.getElementById('transfer-btn').disabled = false;
  // Scroll back to form
  document.getElementById('input-section').scrollIntoView({ behavior: 'smooth' });
}

// ── Tab Switching ─────────────────────────────────────────────

function switchTab(tabName, btn) {
  // Buttons
  document.querySelectorAll('.tab-btn').forEach(b => {
    b.classList.remove('active');
    b.setAttribute('aria-selected', 'false');
  });
  btn.classList.add('active');
  btn.setAttribute('aria-selected', 'true');

  // Panels
  ['rewritten', 'fingerprint', 'quality'].forEach(t => {
    const panel = document.getElementById(`tab-${t}`);
    if (panel) {
      panel.classList.toggle('hidden', t !== tabName);
      panel.classList.toggle('active', t === tabName);
    }
  });

  // Animate progress bars when quality tab opened
  if (tabName === 'quality' && lastResponse) {
    setTimeout(() => animateProgressBars(lastResponse.quality_scores), 80);
  }
}

// ── Main Transfer ─────────────────────────────────────────────

async function transferStyle() {
  const aiText  = document.getElementById('ai-text').value.trim();

  // Validation
  if (!aiText)  { showError('Please paste the AI-generated text you want to rewrite.'); return; }
  if (selectedFiles.length === 0) { showError('Please upload at least one writing sample file.'); return; }

  setLoading('Analysing your style');

  // Build FormData
  const form = new FormData();
  form.append('ai_text', aiText);
  selectedFiles.forEach(file => form.append('files', file));

  try {
    const response = await fetch('http://127.0.0.1:8000/transfer', {
      method: 'POST',
      body: form,
    });

    if (!response.ok) {
      let detail = `Server error (${response.status})`;
      try {
        const errData = await response.json();
        detail = errData.detail || detail;
      } catch (_) {}
      showError(detail);
      return;
    }

    const data = await response.json();
    lastResponse = data;

    hideLoading();
    renderResults(data, aiText);

  } catch (err) {
    showError('Could not connect to the server. Is it running?');
    console.error(err);
  }
}

// ── Render Results ────────────────────────────────────────────

function renderResults(data, originalText) {
  // ── Tab 1: Rewritten Text ──
  document.getElementById('original-text-display').textContent  = originalText;
  document.getElementById('rewritten-text-display').textContent = data.rewritten_text || '—';

  // ── Tab 2: Style Fingerprint ──
  const fp = data.fingerprint || {};

  document.getElementById('fp-avg-sentence').textContent = fp.avg_sentence_length
    ? Math.round(fp.avg_sentence_length) + ' wds'
    : '—';
  document.getElementById('fp-readability').textContent = fp.readability_score
    ? Number(fp.readability_score).toFixed(1)
    : '—';
  document.getElementById('fp-vocab').textContent = fp.vocabulary_richness
    ? (fp.vocabulary_richness * 100).toFixed(0) + '%'
    : '—';
  document.getElementById('fp-tone').textContent = fp.tone || '—';

  // Punctuation tags
  const punctMap = [
    { key: 'uses_contractions',  label: "Contractions",   color: 'mint'     },
    { key: 'uses_parentheses',   label: "Parentheses",    color: 'lavender' },
    { key: 'uses_hedging',       label: "Hedging",        color: 'sky'      },
  ];
  const punctExtras = [];
  if (fp.exclamation_rate > 0)  punctExtras.push({ label: `❗ ${(fp.exclamation_rate*100).toFixed(1)}% exclamations`, color: 'coral' });
  if (fp.ellipsis_count > 0)    punctExtras.push({ label: `… ${fp.ellipsis_count} ellipsis`, color: '' });
  if (fp.dash_usage > 0)        punctExtras.push({ label: `— ${fp.dash_usage} dashes`, color: '' });
  if (fp.comma_rate > 0)        punctExtras.push({ label: `Comma rate ${(fp.comma_rate*100).toFixed(1)}%`, color: 'mint' });

  renderTags('fp-punctuation-tags', [
    ...punctMap.filter(p => fp[p.key]).map(p => ({ label: p.label, color: p.color })),
    ...punctExtras,
  ]);

  // Voice trait tags
  const voiceTraits = [];
  if (fp.first_person_rate > 0)  voiceTraits.push({ label: `1st person ${(fp.first_person_rate*100).toFixed(0)}%`, color: 'lavender' });
  if (fp.passive_voice_ratio > 0) voiceTraits.push({ label: `Passive ${(fp.passive_voice_ratio*100).toFixed(0)}%`, color: '' });
  if (fp.adverb_ratio > 0)       voiceTraits.push({ label: `Adverbs ${(fp.adverb_ratio*100).toFixed(0)}%`, color: 'sky' });
  if (fp.adjective_ratio > 0)    voiceTraits.push({ label: `Adjectives ${(fp.adjective_ratio*100).toFixed(0)}%`, color: 'mint' });
  renderTags('fp-voice-tags', voiceTraits);

  // Top words & signature phrases
  renderTags('fp-top-words',   (fp.top_words  || []).map(w => ({ label: w, color: 'yellow' })));
  renderTags('fp-sig-phrases', (fp.signature_phrases || []).map(p => ({ label: p, color: 'lavender' })));

  // Rich analysis
  document.getElementById('fp-rich-analysis').textContent = data.rich_style_analysis || '—';

  // ── Tab 3: Quality ──
  const qs = data.quality_scores || {};
  renderOverallScore(qs.overall_score || 0);
  document.getElementById('qr-feedback').textContent = qs.feedback       || '—';
  document.getElementById('qr-worked').textContent   = qs.what_worked    || '—';
  document.getElementById('qr-improve').textContent  = qs.what_to_improve || '—';

  // Store for tab-switch animation trigger
  lastResponse = data;

  // Show section, default to tab 1
  document.getElementById('results-section').classList.remove('hidden');
  document.getElementById('results-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderTags(containerId, tags) {
  const container = document.getElementById(containerId);
  container.innerHTML = '';
  if (!tags || tags.length === 0) {
    container.innerHTML = '<span class="tag" style="opacity:0.5">—</span>';
    return;
  }
  const colorCycle = ['', 'mint', 'lavender', 'sky', 'green', 'coral'];
  tags.forEach((t, i) => {
    const span = document.createElement('span');
    span.className = 'tag ' + (t.color !== undefined ? t.color : colorCycle[i % colorCycle.length]);
    span.textContent = t.label || t;
    container.appendChild(span);
  });
}

function renderOverallScore(score) {
  const val = Math.min(Math.max(Number(score) || 0, 0), 10);
  const pct = val / 10;

  document.getElementById('overall-score-val').textContent = val.toFixed(1);

  // Ring animation
  const circumference = 314; // 2π×50
  const offset = circumference - (circumference * pct);
  const ring = document.getElementById('score-ring-fill');
  setTimeout(() => { ring.style.strokeDashoffset = offset; }, 80);

  // Ring color & badge
  let ringColor, badgeClass, badgeText;
  if (pct >= 0.8)      { ringColor = 'var(--green)';   badgeClass = 'great'; badgeText = '🌟 Excellent'; }
  else if (pct >= 0.6) { ringColor = 'var(--mint)';    badgeClass = 'good';  badgeText = '👍 Good'; }
  else if (pct >= 0.4) { ringColor = 'var(--yellow)';  badgeClass = 'ok';    badgeText = '🔧 Decent'; }
  else                  { ringColor = 'var(--coral)';   badgeClass = 'low';   badgeText = '💪 Needs Work'; }

  ring.style.stroke = ringColor;
  const badge = document.getElementById('score-badge-text');
  badge.textContent  = badgeText;
  badge.className    = `score-badge ${badgeClass}`;
}

function animateProgressBars(qs) {
  const bars = [
    { barId: 'pb-style',   valId: 'pv-style',   score: qs.style_match         },
    { barId: 'pb-content', valId: 'pv-content',  score: qs.content_preservation },
    { barId: 'pb-natural', valId: 'pv-natural',  score: qs.naturalness         },
    { barId: 'pb-voice',   valId: 'pv-voice',    score: qs.voice_consistency   },
  ];
  bars.forEach(({ barId, valId, score }) => {
    const val  = Math.min(Math.max(Number(score) || 0, 0), 10);
    const pct  = (val / 10 * 100).toFixed(0);
    document.getElementById(barId).style.width = pct + '%';
    document.getElementById(valId).textContent = val.toFixed(1) + ' / 10';
  });
}

// ── Downloads ─────────────────────────────────────────────────

function downloadRewritten() {
  if (!lastResponse) return;
  const text = lastResponse.rewritten_text || '';
  downloadTextFile('rewritten_text.txt', text);
}

function downloadJSON() {
  if (!lastResponse) return;
  const json = JSON.stringify(lastResponse, null, 2);
  downloadTextFile('textreplica_report.json', json);
}

function downloadTextFile(filename, content) {
  const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

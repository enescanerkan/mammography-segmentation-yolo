// Application shell: a splash while the models load, then three screens (landing, opening, review) plus the
// label editor and the viewer as full-screen overlays. State lives in one object; every screen change goes
// through show(), and every translated string is rebuilt by relabel() when the language changes.

import { api } from './api.js';
import { drawOverlay, FRAME, LAYERS, layerColor, layerName } from './overlay.js';
import { appearance } from './appearance.js';
import { Lightbox } from './lightbox.js';
import { Editor } from './editor.js';
import {
  criterionDetail, criterionName, language, locale, onLanguage, reason, savedLanguage, setLanguage, t,
} from './i18n.js';

// Hanging order: the right breast sits on the reader's left and the left breast on the reader's right, as if
// facing the patient; MLO above CC.
const VIEW_ORDER = ['R-MLO', 'L-MLO', 'R-CC', 'L-CC'];
const DEFAULT_LAYERS = ['breast', 'pectoral', 'nipple', 'points', 'measure_landmark', 'measure_mask'];
const LAYER_STORAGE_KEY = 'mammo.layers.v1';
const LAYOUT_STORAGE_KEY = 'mammo.layout.v1';
const CROP_STORAGE_KEY = 'mammo.crop.v1';
const PAIRING_STORAGE_KEY = 'mammo.pairing.v1';
const REPORT_LANG_KEY = 'mammo.reportLang.v2';
const ROUTE_STYLE = { landmark: 'measure_landmark', mask: 'measure_mask' };
const DEFAULT_TITLE = 'Mammography Positioning Review';

const el = (id) => document.getElementById(id);

const ui = {
  splash: el('splash'),
  splashNote: el('splash-note'),
  appTitle: el('app-title'),
  status: el('status'),
  statusText: el('status-text'),
  banner: el('banner'),
  bannerText: el('banner-text'),
  bannerClose: el('banner-close'),
  langButtons: [...document.querySelectorAll('[data-lang]')],
  version: el('version'),

  screens: {
    landing: el('screen-landing'),
    processing: el('screen-processing'),
    review: el('screen-review'),
  },

  reviewTools: el('review-tools'),
  thresholdInput: el('threshold-input'),
  thresholdUp: el('threshold-up'),
  thresholdDown: el('threshold-down'),
  openEditor: el('open-editor'),
  newStudy: el('new-study'),

  dropzone: el('dropzone'),
  chooseFiles: el('choose-files'),
  chooseFolder: el('choose-folder'),
  fileInput: el('file-input'),
  folderInput: el('folder-input'),
  uploadIdle: el('upload-idle'),
  uploadBusy: el('upload-busy'),
  uploadCount: el('upload-count'),
  uploadCancel: el('upload-cancel'),
  retentionNote: el('retention-note'),
  recentEmpty: el('recent-empty'),
  history: el('history'),
  refreshHistory: el('refresh-history'),
  clearHistory: el('clear-history'),

  progressText: el('progress-text'),

  pager: el('pager'),
  pagePrev: el('page-prev'),
  pageNext: el('page-next'),
  pageLabel: el('page-label'),
  layoutToggle: el('layout-toggle'),
  layoutPop: el('layout-pop'),
  cellGrid: el('cell-grid'),
  layoutLabel: el('layout-label'),
  cropToggle: el('crop-toggle'),
  pairingRow: el('pairing-row'),
  pairing: el('pairing'),
  styleToggle: el('style-toggle'),
  stylePop: el('style-pop'),
  fillAlpha: el('fill-alpha'),
  fillAlphaOut: el('fill-alpha-out'),
  lineWidth: el('line-width'),
  lineWidthOut: el('line-width-out'),
  colorGrid: el('color-grid'),
  styleReset: el('style-reset'),
  criteriaToggle: el('criteria-toggle'),
  scoreChip: el('score-chip'),
  criteriaDrawer: el('criteria-drawer'),
  criteriaGrid: el('criteria-grid'),
  criteriaCounts: el('criteria-counts'),
  criteriaEmpty: el('criteria-empty'),
  criteriaClose: el('criteria-close'),
  reportToggle: el('report-toggle'),
  reportPanel: el('report-panel'),
  reportTitle: el('report-title'),
  reportLang: el('report-lang'),
  reportBody: el('report-body'),
  reportFooterNote: el('report-footer-note'),
  reportClose: el('report-close'),
  reportCopy: el('report-copy'),
  reportDownload: el('report-download'),
  reportPrint: el('report-print'),
  zoomBadge: el('zoom-badge'),
  syncToggle: el('sync-toggle'),
  brightness: el('brightness'),
  brightnessOut: el('brightness-out'),
  contrast: el('contrast'),
  contrastOut: el('contrast-out'),
  invertToggle: el('invert-toggle'),
  windowReset: el('window-reset'),
  layerbar: el('layerbar'),
  notices: el('notices'),
  notesPop: el('notes-pop'),
  notesToggle: el('notes-toggle'),
  notesCount: el('notes-count'),
  notesClose: el('notes-close'),
  quad: el('quad'),
  railR: el('rail-R'),
  railL: el('rail-L'),
  studyMeta: el('study-meta'),
  downloadLabels: el('download-labels'),
  deleteStudy: el('delete-study'),
};

const state = {
  screen: 'landing',
  health: null,
  study: null,
  history: null,
  bitmaps: new Map(),
  boxes: new Map(), // view_key -> [x1, y1, x2, y2] of actual tissue
  tiles: new Map(),
  visible: new Set(loadLayerPrefs()),
  busy: false,
  aborter: null,
  cancelled: false,
  layout: loadLayout(),
  page: 0,
  crop: loadPref(CROP_STORAGE_KEY, '1') === '1',
  pairing: null,
  report: null,
  reportLang: loadPref(REPORT_LANG_KEY, null), // null: follow the interface language
};

const LAYOUT_NAMES = { '1x1': 'layout_1x1', '2x1': 'layout_2x1', '1x2': 'layout_1x2', '2x2': 'layout_2x2' };

const COLOR_FIELDS = [
  ['breast-tissue', 'c_breast'],
  ['pectoral', 'c_pectoral'],
  ['nipple', 'c_nipple'],
  ['nipple_point', 'c_landmarks'],
  ['Mass', 'c_mass'],
  ['Calcification', 'c_calc'],
  ['measure_landmark', 'c_measure_landmark'],
  ['measure_mask', 'c_measure_mask'],
];

// How a two-up layout splits the study. Comparing the same projection across breasts is the symmetry read;
// comparing the two projections of one breast is how you check that breast's own positioning, and ΔD is a
// per-breast number.
const PAIRINGS = {
  view: { name: 'pair_view', hint: 'pair_view_hint', pages: [['R-MLO', 'L-MLO'], ['R-CC', 'L-CC']] },
  breast: { name: 'pair_breast', hint: 'pair_breast_hint', pages: [['L-MLO', 'L-CC'], ['R-MLO', 'R-CC']] },
};
state.pairing = PAIRINGS[loadPref(PAIRING_STORAGE_KEY, 'view')] ? loadPref(PAIRING_STORAGE_KEY, 'view') : 'view';

const CROP_MARGIN = 8; // image px kept around the tissue so contours breathe
const CROP_THRESHOLD = 12; // 8-bit level below which a pixel counts as background

const HEADER_SOURCES = new Set(['ViewPosition', 'ViewCodeSequence', 'ImageLaterality', 'Laterality', 'FrameLaterality']);
const SOURCE_KEYS = { 'folder name': 'src_folder', 'file name': 'src_file', 'upload path': 'src_path' };

// ── preferences ────────────────────────────────────────────────────────────
function loadPref(key, fallback) {
  try {
    const value = localStorage.getItem(key);
    return value === null ? fallback : value;
  } catch (error) { return fallback; }
}

function savePref(key, value) {
  try { localStorage.setItem(key, value); } catch (error) { /* blocked storage: the session still honours it */ }
}

function loadLayout() {
  try {
    const raw = JSON.parse(localStorage.getItem(LAYOUT_STORAGE_KEY) || 'null');
    if (raw && [1, 2].includes(raw.cols) && [1, 2].includes(raw.rows)) return raw;
  } catch (error) { /* fall through to the default */ }
  return { cols: 2, rows: 2 };
}

function loadLayerPrefs() {
  try {
    const raw = localStorage.getItem(LAYER_STORAGE_KEY);
    if (!raw) return DEFAULT_LAYERS;
    const known = new Set(LAYERS.map((layer) => layer.id));
    const parsed = JSON.parse(raw);
    const kept = Array.isArray(parsed) ? parsed.filter((id) => known.has(id)) : [];
    return kept.length ? kept : DEFAULT_LAYERS;
  } catch (error) {
    return DEFAULT_LAYERS;
  }
}

/**
 * The bounding box of everything that is not padding.
 *
 * Preprocessing pads each render to a square, which on a portrait mammogram is about 30% black — a third of every
 * tile spent on nothing. The box is measured from the PNG rather than sent by the server so that older studies
 * benefit too.
 */
function contentBox(bitmap) {
  const canvas = document.createElement('canvas');
  canvas.width = FRAME;
  canvas.height = FRAME;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0, FRAME, FRAME);

  let data;
  try {
    ({ data } = ctx.getImageData(0, 0, FRAME, FRAME));
  } catch (error) {
    return [0, 0, FRAME, FRAME];
  }

  let minX = FRAME; let minY = FRAME; let maxX = -1; let maxY = -1;
  for (let y = 0; y < FRAME; y += 1) {
    const row = y * FRAME * 4;
    for (let x = 0; x < FRAME; x += 1) {
      if (data[row + x * 4] > CROP_THRESHOLD) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < 0) return [0, 0, FRAME, FRAME];
  return [
    Math.max(0, minX - CROP_MARGIN), Math.max(0, minY - CROP_MARGIN),
    Math.min(FRAME, maxX + 1 + CROP_MARGIN), Math.min(FRAME, maxY + 1 + CROP_MARGIN),
  ];
}

const boxOf = (viewKey) => (state.crop && state.boxes.get(viewKey)) || [0, 0, FRAME, FRAME];

// One transform shared by every tile: comparing left with right means looking at the same place in both, so
// panning one pans all. The toggle exists because a single odd view sometimes needs its own look.
state.view = { zoom: 1, x: 0, y: 0 };
state.syncViews = true;
state.tileViews = new Map(); // view_key -> its own transform when unsynced

function viewOf(viewKey) {
  if (state.syncViews) return state.view;
  if (!state.tileViews.has(viewKey)) state.tileViews.set(viewKey, { zoom: 1, x: 0, y: 0 });
  return state.tileViews.get(viewKey);
}

function resetViews() {
  state.view = { zoom: 1, x: 0, y: 0 };
  state.tileViews.clear();
  redrawTiles();
}

const capacity = () => state.layout.cols * state.layout.rows;

/** The view keys shown on each page, for the current layout and pairing. */
function pages() {
  const size = capacity();
  if (size >= 4) return [VIEW_ORDER];
  if (size === 1) return VIEW_ORDER.map((key) => [key]);
  return PAIRINGS[state.pairing].pages;
}

const pageCount = () => pages().length;

const lightbox = new Lightbox({
  root: el('lightbox'),
  stage: el('lightbox-stage'),
  title: el('lightbox-title'),
  close: el('lightbox-close'),
  zoomIn: el('zoom-in'),
  zoomOut: el('zoom-out'),
  zoomReset: el('zoom-reset'),
  zoomLevel: el('zoom-level'),
}, () => state.visible);

const editor = new Editor({
  root: el('editor'),
  stage: el('editor-stage'),
  canvas: el('editor-canvas'),
  views: el('editor-views'),
  classes: el('editor-classes'),
  tools: el('editor-tools'),
  toolHint: el('tool-hint'),
  viewName: el('editor-view-name'),
  dirty: el('editor-dirty'),
  back: el('editor-back'),
  undo: el('editor-undo'),
  save: el('editor-save'),
  zoomIn: el('editor-zoom-in'),
  zoomOut: el('editor-zoom-out'),
  zoomFit: el('editor-zoom-fit'),
  zoomLevel: el('editor-zoom-level'),
}, { onSave: saveLabels, onClose: closeEditor });

// ── small helpers ──────────────────────────────────────────────────────────
function saveLayerPrefs() { savePref(LAYER_STORAGE_KEY, JSON.stringify([...state.visible])); }

function showError(message) {
  ui.bannerText.textContent = message;
  ui.banner.hidden = false;
}

function clearError() { ui.banner.hidden = true; }

function show(screen) {
  state.screen = screen;
  for (const [name, node] of Object.entries(ui.screens)) node.hidden = name !== screen;
  ui.reviewTools.hidden = screen !== 'review';
  if (screen !== 'review') {
    ui.reportPanel.hidden = true;
    ui.notesPop.hidden = true;
  }
}

function formatTime(seconds) {
  if (!seconds) return '—';
  const date = new Date(seconds * 1000);
  const sameDay = date.toDateString() === new Date().toDateString();
  return sameDay
    ? date.toLocaleTimeString(locale(), { hour: '2-digit', minute: '2-digit' })
    : date.toLocaleDateString(locale(), { day: '2-digit', month: 'short' });
}

function debounce(fn, wait) {
  let timer = null;
  return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), wait); };
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const gradeText = (grade) => t(`grade_${grade || 'na'}`);
const verdictText = (verdict) => t(verdict === 'good' ? 'verdict_good' : 'verdict_bad');
const megabytes = (bytes) => `${(bytes / 1048576).toFixed(1)} MB`;
const appTitle = () => {
  const configured = state.health && state.health.title;
  return configured && configured !== DEFAULT_TITLE ? configured : t('app_title');
};
const reportLanguage = () => state.reportLang || language();

function comparisonNote(comparison) {
  if (comparison.state === 'disagree') return t('cmp_disagree');
  if (comparison.state === 'agree') {
    return comparison.wide_gap ? t('cmp_wide', { gap: Number(comparison.gap_mm).toFixed(1) }) : t('cmp_agree');
  }
  return t(comparison.state === 'partial' ? 'cmp_partial' : 'cmp_none');
}

// ── boot: splash until the models are loaded ───────────────────────────────
async function boot() {
  setLanguage(savedLanguage('en'));
  buildLayerBar();
  buildCellGrid();
  renderPairingButtons();
  buildColorGrid();
  syncStyleControls();
  wireEvents();
  markLanguage();

  appearance.onChange(() => {
    buildLayerBar();
    // Rebuilding the grid replaces the <input type=color> being dragged, and the browser closes a picker whose
    // input leaves the page.
    if (!colorEditing) buildColorGrid();
    syncStyleControls();
    redrawTiles();
    lightbox.render();
  });
  onLanguage(relabel);

  show('landing');
  refreshHistory();
  const health = await waitForModels();
  applyHealth(health);
  setTimeout(() => ui.splash.classList.add('done'), 300);
  setTimeout(() => { ui.splash.hidden = true; }, 800);

  const fromUrl = location.hash.replace('#', '').trim();
  if (fromUrl) await openStudy(fromUrl);
}

async function waitForModels() {
  let first = true;
  for (;;) {
    try {
      const health = await api.health();
      if (first) {
        first = false;
        // The server's default language applies until the reader picks one.
        setLanguage(savedLanguage(health.language || 'en'));
      }
      state.health = health;
      stepState('sp-server', 'ok');
      updateSplash(health);
      renderStatus();
      if (!health.loading) return health;
      ui.splashNote.textContent = t('splash_sub');
    } catch (error) {
      ui.splashNote.textContent = t('sp_retry');
    }
    await sleep(1000);
  }
}

function stepState(id, mode) {
  const node = el(id);
  node.classList.remove('ok', 'err', 'skip');
  if (mode) node.classList.add(mode);
}

function updateSplash(health) {
  const roles = health.roles || {};
  const loaded = new Set(health.loaded || []);
  const errors = health.errors || {};
  const role = (name) => {
    const id = roles[name];
    if (!id) return errors[name] ? 'err' : '';
    if (loaded.has(id)) return 'ok';
    return errors[id] ? 'err' : '';
  };
  const both = [role('pose_mlo'), role('pose_cc')];
  stepState('sp-pose', both.includes('err') ? 'err' : (both.every((s) => s === 'ok') ? 'ok' : ''));
  stepState('sp-seg', role('segmentation'));
  if (!roles.detection) stepState('sp-lesion', health.loading ? '' : 'skip');
  else stepState('sp-lesion', role('detection'));
}

function applyHealth(health) {
  state.health = health;
  document.title = appTitle();
  ui.appTitle.textContent = appTitle();
  ui.version.textContent = health.version ? `v${health.version}` : '';
  renderStatus();
  renderRetention();
  if (!health.ready) {
    const problems = Object.entries(health.errors || {}).map(([id, why]) => `${id}: ${why}`).join('\n');
    showError(t('models_failed', { list: problems }));
  }
  if (health.rules && !state.study) ui.thresholdInput.value = String(health.rules.threshold_mm);
}

function renderStatus() {
  const health = state.health;
  if (!health) {
    ui.status.dataset.state = 'loading';
    ui.statusText.textContent = t('status_connecting');
  } else if (health.ready) {
    ui.status.dataset.state = 'ready';
    ui.statusText.textContent = t('status_ready', { n: (health.loaded || []).length });
  } else if (health.loading) {
    ui.status.dataset.state = 'loading';
    ui.statusText.textContent = t('status_loading');
  } else {
    ui.status.dataset.state = 'error';
    ui.statusText.textContent = t('status_missing');
  }
}

function renderRetention() {
  const days = state.health && state.health.storage ? state.health.storage.retention_days : null;
  if (days === null || days === undefined) { ui.retentionNote.textContent = ''; return; }
  ui.retentionNote.textContent = days > 0 ? t('retention', { days }) : t('retention_off');
}

function markLanguage() {
  for (const button of ui.langButtons) button.classList.toggle('active', button.dataset.lang === language());
}

/** Every string the scripts built, rebuilt in the new language; the reading state is left alone. */
function relabel() {
  markLanguage();
  document.title = appTitle();
  ui.appTitle.textContent = appTitle();
  buildLayerBar();
  buildCellGrid();
  renderPairingButtons();
  if (!colorEditing) buildColorGrid();
  renderStatus();
  renderRetention();
  if (state.history) renderHistory(state.history);
  if (state.study) {
    renderReview(state.study);
    redrawTiles();
    if (!ui.reportPanel.hidden) loadReport();
  }
  editor.relabel();
}

function buildLayerBar() {
  ui.layerbar.replaceChildren(...LAYERS.map((layer) => {
    const label = document.createElement('label');
    label.className = 'layer-chip' + (state.visible.has(layer.id) ? ' on' : '');

    const input = document.createElement('input');
    input.type = 'checkbox';
    input.checked = state.visible.has(layer.id);
    input.addEventListener('change', () => {
      if (input.checked) state.visible.add(layer.id);
      else state.visible.delete(layer.id);
      label.classList.toggle('on', input.checked);
      saveLayerPrefs();
      redrawTiles();
      lightbox.render();
    });

    const swatch = document.createElement('span');
    swatch.className = 'swatch';
    swatch.style.background = layerColor(layer);

    label.append(input, swatch, document.createTextNode(layerName(layer)));
    return label;
  }));
}

function wireEvents() {
  ui.bannerClose.addEventListener('click', clearError);
  ui.refreshHistory.addEventListener('click', refreshHistory);
  for (const button of ui.langButtons) {
    button.addEventListener('click', () => {
      if (button.dataset.lang !== language()) setLanguage(button.dataset.lang, { remember: true });
    });
  }

  ui.dropzone.addEventListener('click', () => ui.fileInput.click());
  ui.dropzone.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); ui.fileInput.click(); }
  });
  ui.chooseFiles.addEventListener('click', (event) => { event.stopPropagation(); ui.fileInput.click(); });
  ui.chooseFolder.addEventListener('click', (event) => { event.stopPropagation(); ui.folderInput.click(); });
  ui.fileInput.addEventListener('change', () => {
    const files = [...ui.fileInput.files];
    ui.fileInput.value = '';
    if (files.length) upload(files);
  });
  ui.folderInput.addEventListener('change', () => {
    const files = [...ui.folderInput.files];
    ui.folderInput.value = '';
    if (files.length) upload(pickLikelyDicoms(files));
  });

  ui.clearHistory.addEventListener('click', clearHistory);

  for (const type of ['dragenter', 'dragover']) {
    ui.dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      ui.dropzone.classList.add('dragging');
    });
  }
  for (const type of ['dragleave', 'drop']) {
    ui.dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      ui.dropzone.classList.remove('dragging');
    });
  }
  ui.dropzone.addEventListener('drop', async (event) => {
    const transfer = event.dataTransfer;
    if (!transfer) return;
    // A dropped folder shows up as an entry, not a file; walk it so the same case-042/RMLO/… layout works by
    // drag as well as by the picker.
    const entries = [...(transfer.items || [])]
      .map((item) => (item.webkitGetAsEntry ? item.webkitGetAsEntry() : null))
      .filter(Boolean);
    const files = entries.length ? await collectFiles(entries) : [...(transfer.files || [])];
    if (files.length) upload(pickLikelyDicoms(files));
  });
  window.addEventListener('dragover', (event) => event.preventDefault());
  window.addEventListener('drop', (event) => event.preventDefault());

  ui.uploadCancel.addEventListener('click', cancelUpload);
  ui.newStudy.addEventListener('click', startNewStudy);
  ui.openEditor.addEventListener('click', openEditor);
  ui.downloadLabels.addEventListener('click', downloadLabels);
  ui.deleteStudy.addEventListener('click', deleteCurrent);
  el('editor-reset').addEventListener('click', revertEditorView);

  ui.thresholdUp.addEventListener('click', () => nudgeThreshold(0.5));
  ui.thresholdDown.addEventListener('click', () => nudgeThreshold(-0.5));
  ui.thresholdInput.addEventListener('change', applyThreshold);
  ui.thresholdInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') { event.preventDefault(); applyThreshold(); }
  });

  ui.pagePrev.addEventListener('click', () => turnPage(-1));
  ui.pageNext.addEventListener('click', () => turnPage(1));
  ui.layoutToggle.addEventListener('click', () => togglePopover(ui.layoutToggle, ui.layoutPop));
  ui.styleToggle.addEventListener('click', () => togglePopover(ui.styleToggle, ui.stylePop));
  ui.layoutPop.addEventListener('pointerleave', () => previewLayout(state.layout.cols, state.layout.rows));

  ui.cropToggle.checked = state.crop;
  ui.cropToggle.addEventListener('change', () => {
    state.crop = ui.cropToggle.checked;
    savePref(CROP_STORAGE_KEY, state.crop ? '1' : '0');
    redrawTiles();
    lightbox.render();
  });

  ui.fillAlpha.addEventListener('input', () => {
    appearance.fillAlpha = Number(ui.fillAlpha.value) / 100;
    ui.fillAlphaOut.textContent = `${ui.fillAlpha.value}%`;
    appearance.save();
  });
  ui.lineWidth.addEventListener('input', () => {
    appearance.lineWidth = Number(ui.lineWidth.value) / 10;
    ui.lineWidthOut.textContent = appearance.lineWidth.toFixed(1);
    appearance.save();
  });
  ui.styleReset.addEventListener('click', () => appearance.reset());

  ui.brightness.addEventListener('input', () => {
    appearance.brightness = Number(ui.brightness.value) / 100;
    ui.brightnessOut.textContent = appearance.brightness.toFixed(2);
    appearance.save();
  });
  ui.contrast.addEventListener('input', () => {
    appearance.contrast = Number(ui.contrast.value) / 100;
    ui.contrastOut.textContent = appearance.contrast.toFixed(2);
    appearance.save();
  });
  ui.invertToggle.addEventListener('change', () => {
    appearance.invert = ui.invertToggle.checked;
    appearance.save();
  });
  ui.windowReset.addEventListener('click', () => appearance.resetWindow());

  ui.syncToggle.addEventListener('change', () => {
    state.syncViews = ui.syncToggle.checked;
    resetViews();
  });

  ui.criteriaToggle.addEventListener('click', () => setDrawer(ui.criteriaDrawer.hidden));
  ui.criteriaClose.addEventListener('click', () => setDrawer(false));
  ui.notesToggle.addEventListener('click', () => setNotes(ui.notesPop.hidden));
  ui.notesClose.addEventListener('click', () => setNotes(false));

  ui.reportToggle.addEventListener('click', () => toggleReport(ui.reportPanel.hidden));
  ui.reportClose.addEventListener('click', () => toggleReport(false));
  ui.reportCopy.addEventListener('click', copyReport);
  ui.reportDownload.addEventListener('click', downloadReport);
  ui.reportPrint.addEventListener('click', () => window.print());

  document.addEventListener('pointerdown', (event) => {
    if (!event.target.closest('.popwrap')) closePopovers();
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') closePopovers();
    if (state.screen !== 'review' || event.target instanceof HTMLInputElement) return;
    if (!el('editor').hidden || !el('lightbox').hidden) return;
    if (event.key === 'ArrowLeft') turnPage(-1);
    if (event.key === 'ArrowRight') turnPage(1);
    if ((event.key === 'f' || event.key === 'F') && !event.ctrlKey && !event.metaKey && !event.altKey) toggleFullscreen();
  });
  el('theme-toggle').addEventListener('click', toggleTheme);
  el('fullscreen-toggle').addEventListener('click', toggleFullscreen);
  document.addEventListener('fullscreenchange', () => {
    el('fullscreen-toggle').setAttribute('aria-pressed', String(Boolean(document.fullscreenElement)));
    redrawTiles();
  });

  window.addEventListener('resize', debounce(redrawTiles, 120));
  window.addEventListener('hashchange', () => {
    const id = location.hash.replace('#', '').trim();
    if (id && (!state.study || state.study.id !== id)) openStudy(id);
  });
}

// ── upload ─────────────────────────────────────────────────────────────────
const VIEW_IN_PATH = /(^|[^a-z])(?:[lr][\W_]*)?(?:mlo|cc)([^a-z]|$)/i;
const DICOM_EXT = /\.(dcm|dicom|ima)$/i;
// A study folder usually ships more than images: label sidecars, reports, thumbnails, OS junk. These never contain
// pixel data, and sending them only produces a per-file error the reader then has to decode.
const NOT_AN_IMAGE = /\.(txt|json|csv|tsv|xml|ya?ml|md|pdf|docx?|xlsx?|zip|gz|log|ini|png|jpe?g|gif|bmp|tiff?|svg|webp|mp4|avi|db|sqlite)$/i;

/** Depth-first walk of dropped directory entries, keeping the relative path. */
async function collectFiles(entries, prefix = '') {
  const out = [];
  for (const entry of entries) {
    if (entry.isFile) {
      const file = await new Promise((resolve) => entry.file(resolve, () => resolve(null)));
      if (file) {
        file.relativePath = prefix + entry.name;
        out.push(file);
      }
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      const children = [];
      // readEntries returns at most 100 at a time; keep asking until it is empty.
      for (;;) {
        const batch = await new Promise((resolve) => reader.readEntries(resolve, () => resolve([])));
        if (!batch.length) break;
        children.push(...batch);
      }
      out.push(...await collectFiles(children, `${prefix}${entry.name}/`));
    }
  }
  return out;
}

/**
 * Narrow a folder full of files down to the ones plausibly worth uploading.
 *
 * Anything whose path names a view is kept first; if nothing does, fall back to DICOM extensions, and failing
 * that send what there is and let the server explain itself per file.
 */
function pickLikelyDicoms(files) {
  const limit = (state.health && state.health.upload && state.health.upload.max_files) || 40;
  const named = (file) => file.webkitRelativePath || file.relativePath || file.name;
  const usable = files.filter((file) => {
    const path = named(file);
    const base = path.split('/').pop();
    return file.size > 0 && !base.startsWith('.') && !NOT_AN_IMAGE.test(base);
  });

  // Prefer files whose path names a view, but only among things that could be images — a labels/ sidecar inside
  // RMLO/ matches the view too.
  const byView = usable.filter((file) => VIEW_IN_PATH.test(named(file)));
  if (byView.length) return byView.slice(0, limit);
  const byExt = usable.filter((file) => DICOM_EXT.test(named(file)));
  return (byExt.length ? byExt : usable).slice(0, limit);
}

async function clearHistory() {
  if (!window.confirm(t('clear_confirm'))) return;
  try {
    const { deleted } = await api.deleteAll();
    if (state.study) startNewStudy();
    else await refreshHistory();
    showError(deleted === 1 ? t('deleted_1') : t('deleted_n', { n: deleted }));
    setTimeout(clearError, 3000);
  } catch (error) {
    showError(error.message);
  }
}

function stage(id, mode) { // mode: 'active' | 'done' | ''
  const node = el(id);
  node.classList.remove('active', 'done');
  if (mode) node.classList.add(mode);
}

function setBar(id, fraction) {
  el(id).style.width = `${Math.max(0, Math.min(1, fraction)) * 100}%`;
}

function resetStages() {
  for (const id of ['st-upload', 'st-analyse', 'st-render']) stage(id, '');
  setBar('st-upload-bar', 0);
  setBar('st-render-bar', 0);
  el('st-analyse-bar').classList.remove('progress-indeterminate');
  el('st-analyse-bar').firstElementChild.style.width = '0%';
  for (const id of ['st-upload-val', 'st-analyse-val', 'st-render-val']) el(id).textContent = '';
}

function setUploading(on, count = 0) {
  ui.uploadIdle.hidden = on;
  ui.uploadBusy.hidden = !on;
  ui.uploadCount.textContent = on ? (count === 1 ? t('file_1') : t('files_n', { n: count })) : '';
  if (on) resetStages();
}

async function upload(files) {
  if (state.busy) return;
  clearError();

  const accepted = files.filter((file) => file.size > 0);
  if (!accepted.length) { showError(t('empty_files')); return; }

  state.busy = true;
  state.cancelled = false;
  state.aborter = new AbortController();
  setUploading(true, accepted.length);
  stage('st-upload', 'active');

  let timer = null;
  const started = Date.now();
  const onProgress = (event) => {
    if (event.phase === 'upload') {
      setBar('st-upload-bar', event.loaded / Math.max(event.total, 1));
      el('st-upload-val').textContent = `${megabytes(event.loaded)} / ${megabytes(event.total)}`;
    } else if (!timer) {
      stage('st-upload', 'done');
      setBar('st-upload-bar', 1);
      stage('st-analyse', 'active');
      el('st-analyse-bar').classList.add('progress-indeterminate');
      timer = setInterval(() => {
        el('st-analyse-val').textContent = `${Math.round((Date.now() - started) / 1000)} s`;
      }, 500);
    }
  };

  try {
    const study = await api.createStudy(accepted, { onProgress, signal: state.aborter.signal });
    clearInterval(timer);
    stage('st-upload', 'done');
    stage('st-analyse', 'done');
    el('st-analyse-bar').classList.remove('progress-indeterminate');
    el('st-analyse-bar').firstElementChild.style.width = '100%';
    stage('st-render', 'active');
    state.bitmaps.clear();
    await loadBitmaps(study, (done, total) => {
      setBar('st-render-bar', done / Math.max(total, 1));
      el('st-render-val').textContent = `${done} / ${total}`;
    });
    stage('st-render', 'done');
    state.aborter = null;
    state.busy = false;
    setUploading(false);
    await showStudy(study);
    refreshHistory();
  } catch (error) {
    clearInterval(timer);
    state.aborter = null;
    state.busy = false;
    setUploading(false);
    if (!state.cancelled) showError(error.message);
    show(state.study ? 'review' : 'landing');
  }
}

function cancelUpload() {
  state.cancelled = true;
  if (state.aborter) state.aborter.abort();
}

function startNewStudy() {
  state.study = null;
  state.report = null;
  state.tiles.clear();
  state.bitmaps.clear();
  history.replaceState(null, '', location.pathname);
  clearError();
  if (state.health && state.health.rules) ui.thresholdInput.value = String(state.health.rules.threshold_mm);
  show('landing');
  refreshHistory();
}

// ── review ─────────────────────────────────────────────────────────────────
/** Fetch every render of the study (in parallel), reporting how many have arrived. */
async function loadBitmaps(study, onProgress = () => {}) {
  const total = study.images.length;
  let done = 0;
  onProgress(done, total);
  await Promise.all(study.images.map(async (image) => {
    if (!state.bitmaps.has(image.view_key)) {
      try {
        const bitmap = await loadImage(api.imageUrl(study.id, image.view_key));
        state.bitmaps.set(image.view_key, bitmap);
        state.boxes.set(image.view_key, contentBox(bitmap));
      } catch (error) {
        state.bitmaps.delete(image.view_key);
        state.boxes.delete(image.view_key);
      }
    }
    done += 1;
    onProgress(done, total);
  }));
}

async function showStudy(study) {
  state.study = study;
  state.report = null;
  await loadBitmaps(study);

  ui.thresholdInput.value = String((study.rules || {}).threshold_mm ?? 10);
  renderReview(study);

  show('review');
  if (!ui.reportPanel.hidden) loadReport();
  if (location.hash.replace('#', '') !== study.id) history.replaceState(null, '', `#${study.id}`);
  redrawTiles();
}

function renderReview(study) {
  renderNotices(study);
  renderQuad(study);
  renderRails(study);
  renderCriteria(study);
  renderFoot(study);
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('image failed to load'));
    image.src = url;
  });
}

function renderNotices(study) {
  const notices = [];
  if ((study.summary || {}).flagged) notices.push({ text: t('notice_flagged'), tone: 'warn' });
  const unknownScale = study.images.filter((image) => !image.pixel_spacing_known);
  if (unknownScale.length) {
    notices.push({ tone: 'info', text: t('notice_spacing', { views: unknownScale.map((i) => i.view_key).join(', ') }) });
  }
  // Files the upload stepped over are worth recording but are not findings; they sit at the bottom of the list
  // in a quieter style.
  for (const problem of study.problems || []) {
    notices.push({ text: problem, tone: /\bskipped\b/.test(problem) ? 'skip' : 'warn' });
  }

  ui.notices.replaceChildren(...notices.map((notice) => {
    const item = document.createElement('li');
    item.className = `notice${notice.tone === 'warn' ? '' : ` ${notice.tone}`}`;
    item.textContent = notice.text;
    return item;
  }));

  const attention = notices.filter((notice) => notice.tone === 'warn').length;
  ui.notesToggle.hidden = notices.length === 0;
  ui.notesCount.textContent = String(notices.length);
  if (attention) ui.notesCount.dataset.tone = 'warn';
  else delete ui.notesCount.dataset.tone;
  if (!notices.length) setNotes(false);
}

function setNotes(open) {
  ui.notesPop.hidden = !open;
  ui.notesToggle.setAttribute('aria-expanded', String(open));
}

function renderQuad(study) {
  const byKey = new Map(study.images.map((image) => [image.view_key, image]));
  const layoutPages = pages();
  state.page = Math.min(Math.max(state.page, 0), layoutPages.length - 1);
  const shown = layoutPages[state.page];

  state.tiles.clear();
  ui.quad.style.gridTemplateColumns = `repeat(${state.layout.cols}, minmax(0, 1fr))`;
  ui.quad.style.gridTemplateRows = `repeat(${state.layout.rows}, minmax(0, 1fr))`;
  ui.quad.replaceChildren(...shown.map((viewKey) => {
    const image = byKey.get(viewKey);
    return image ? buildTile(image) : buildMissingTile(viewKey);
  }));

  ui.pager.hidden = layoutPages.length < 2;
  ui.pageLabel.textContent = `${state.page + 1} / ${layoutPages.length}`;
  ui.pagePrev.disabled = state.page === 0;
  ui.pageNext.disabled = state.page >= layoutPages.length - 1;
  ui.pairingRow.hidden = capacity() !== 2;
}

function setPairing(mode) {
  if (!(mode in PAIRINGS)) return;
  state.pairing = mode;
  savePref(PAIRING_STORAGE_KEY, mode);
  state.page = 0;
  renderPairingButtons();
  if (state.study) { renderQuad(state.study); redrawTiles(); }
}

function renderPairingButtons() {
  ui.pairing.replaceChildren(...Object.entries(PAIRINGS).map(([mode, config]) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'btn btn-sm' + (mode === state.pairing ? ' active' : '');
    button.textContent = t(config.name);
    button.title = t(config.hint);
    button.addEventListener('click', () => setPairing(mode));
    return button;
  }));
}

function setLayout(cols, rows) {
  state.layout = { cols, rows };
  savePref(LAYOUT_STORAGE_KEY, JSON.stringify(state.layout));
  state.page = 0;
  if (state.study) { renderQuad(state.study); redrawTiles(); }
  buildCellGrid();
}

function turnPage(step) {
  const next = state.page + step;
  if (next < 0 || next >= pageCount()) return;
  state.page = next;
  if (state.study) { renderQuad(state.study); redrawTiles(); }
}

// A Word-style size picker: hover across for columns, down for rows.
function buildCellGrid() {
  const cells = [];
  for (let row = 1; row <= 2; row += 1) {
    for (let col = 1; col <= 2; col += 1) {
      const cell = document.createElement('button');
      cell.type = 'button';
      cell.className = 'cell';
      cell.dataset.cols = String(col);
      cell.dataset.rows = String(row);
      cell.setAttribute('aria-label', t('cell_aria', { c: col, r: row }));
      cell.addEventListener('pointerenter', () => previewLayout(col, row));
      cell.addEventListener('focus', () => previewLayout(col, row));
      cell.addEventListener('click', () => { setLayout(col, row); closePopovers(); });
      cells.push(cell);
    }
  }
  ui.cellGrid.replaceChildren(...cells);
  previewLayout(state.layout.cols, state.layout.rows);
}

function previewLayout(cols, rows) {
  for (const cell of ui.cellGrid.querySelectorAll('.cell')) {
    const on = Number(cell.dataset.cols) <= cols && Number(cell.dataset.rows) <= rows;
    cell.classList.toggle('on', on);
  }
  ui.layoutLabel.textContent = `${cols} × ${rows} · ${t(LAYOUT_NAMES[`${cols}x${rows}`])}`;
}

let colorEditing = false;

function buildColorGrid() {
  ui.colorGrid.replaceChildren(...COLOR_FIELDS.map(([key, name]) => {
    const row = document.createElement('label');
    row.className = 'color-row';
    const input = document.createElement('input');
    input.type = 'color';
    input.value = appearance.colorOf(key);
    input.addEventListener('input', () => {
      appearance.colors[key] = input.value;
      colorEditing = true;
      try { appearance.save(); } finally { colorEditing = false; }
    });
    const text = document.createElement('span');
    text.textContent = t(name);
    row.append(input, text);
    return row;
  }));
}

function syncStyleControls() {
  ui.fillAlpha.value = String(Math.round(appearance.fillAlpha * 100));
  ui.fillAlphaOut.textContent = `${Math.round(appearance.fillAlpha * 100)}%`;
  ui.lineWidth.value = String(Math.round(appearance.lineWidth * 10));
  ui.lineWidthOut.textContent = appearance.lineWidth.toFixed(1);
  ui.brightness.value = String(Math.round(appearance.brightness * 100));
  ui.brightnessOut.textContent = appearance.brightness.toFixed(2);
  ui.contrast.value = String(Math.round(appearance.contrast * 100));
  ui.contrastOut.textContent = appearance.contrast.toFixed(2);
  ui.invertToggle.checked = appearance.invert;
}

function closePopovers() {
  for (const [toggle, pop] of [[ui.layoutToggle, ui.layoutPop], [ui.styleToggle, ui.stylePop]]) {
    pop.hidden = true;
    toggle.setAttribute('aria-expanded', 'false');
  }
}

function togglePopover(toggle, pop) {
  const open = pop.hidden;
  closePopovers();
  pop.hidden = !open;
  toggle.setAttribute('aria-expanded', String(open));
}

function buildMissingTile(viewKey) {
  const tile = document.createElement('div');
  tile.className = 'view-tile missing';
  const head = document.createElement('div');
  head.className = 'tile-head';
  const key = document.createElement('span');
  key.className = 'tile-key';
  key.textContent = viewKey;
  head.appendChild(key);
  const body = document.createElement('div');
  body.className = 'tile-empty';
  body.textContent = t('not_uploaded');
  tile.append(head, body);
  return tile;
}

function buildTile(image) {
  const tile = document.createElement('div');
  tile.className = 'view-tile';

  const head = document.createElement('div');
  head.className = 'tile-head';
  const key = document.createElement('span');
  key.className = 'tile-key';
  const grade = worstGrade(image);
  if (grade) {
    const dot = document.createElement('span');
    dot.className = 'grade-dot';
    dot.dataset.grade = grade;
    dot.title = t('worst_on_view', { grade: gradeText(grade) });
    key.appendChild(dot);
  }
  key.append(document.createTextNode(image.view_key));
  const badges = document.createElement('div');
  badges.className = 'tile-badges';
  for (const badge of tileBadges(image)) {
    const span = document.createElement('span');
    span.className = 'tile-badge';
    if (badge.tone) span.dataset.tone = badge.tone;
    if (badge.title) span.title = badge.title;
    span.textContent = badge.text;
    badges.appendChild(span);
  }
  head.append(key, badges);

  const bitmap = state.bitmaps.get(image.view_key);
  let body;
  if (bitmap) {
    body = document.createElement('button');
    body.type = 'button';
    body.className = 'tile-canvas-wrap';
    body.setAttribute('aria-label', t('enlarge', { view: image.view_key }));
    const canvas = document.createElement('canvas');
    body.appendChild(canvas);
    state.tiles.set(image.view_key, { canvas, wrap: body, image });
    attachTileGestures(body, image.view_key, () => lightbox.show(image, bitmap, boxOf(image.view_key)));
  } else {
    body = document.createElement('div');
    body.className = 'tile-empty dark';
    body.textContent = t('preview_unavailable');
  }

  const foot = document.createElement('div');
  foot.className = 'tile-foot';
  for (const route of ['landmark', 'mask']) {
    const measure = image.measures[route];
    const span = document.createElement('span');
    span.className = 'route-value';
    const swatch = document.createElement('i');
    swatch.className = 'route-swatch';
    swatch.style.background = appearance.colorOf(ROUTE_STYLE[route]);
    span.appendChild(swatch);
    if (measure.ok) {
      span.append(document.createTextNode(`${measure.mm.toFixed(1)} mm`));
    } else {
      span.classList.add('miss');
      span.append(document.createTextNode(reason(measure.reason)));
    }
    span.title = t(route === 'landmark' ? 'method_landmark' : 'method_mask');
    foot.appendChild(span);
  }

  tile.append(head, body, foot);
  return tile;
}

function tileBadges(image) {
  const badges = [];
  const landmarks = image.landmarks || {};

  // A view read out of a file name is a guess, not a header fact — say so, because getting it wrong swaps which
  // breast a measurement belongs to.
  const guessed = [image.view_source, image.laterality_source].filter((source) => source && !HEADER_SOURCES.has(source));
  if (guessed.length) {
    const sourceName = (source) => t(SOURCE_KEYS[source] || '', {}, source);
    badges.push({
      text: t('badge_view', { source: sourceName(guessed[0]) }),
      tone: 'warn',
      title: t('badge_view_title', { sources: [...new Set(guessed)].map(sourceName).join(' / ') }),
    });
  }
  if (landmarks.edited) badges.push({ text: t('badge_edited'), tone: 'warn', title: t('badge_edited_title') });
  if (!landmarks.detected) badges.push({ text: t('badge_no_landmarks'), tone: 'warn' });
  if (landmarks.mirrored) badges.push({ text: t('badge_mirrored'), tone: 'warn', title: t('badge_mirrored_title') });
  const nipple = (image.polygons || {}).nipple;
  if (nipple && nipple.rescued) badges.push({ text: t('badge_rescued'), tone: 'warn', title: t('badge_rescued_title') });
  if (image.lesions && image.lesions.length) {
    const n = image.lesions.length;
    badges.push({ text: n === 1 ? t('lesions_1') : t('lesions_n', { n }) });
  }
  return badges;
}

function renderRails(study) {
  const sides = new Map(((study.summary || {}).sides || []).map((side) => [side.laterality, side]));
  renderRail(ui.railR, 'R', sides.get('R'));
  renderRail(ui.railL, 'L', sides.get('L'));
}

function renderRail(node, laterality, side) {
  const head = document.createElement('div');
  head.className = 'rail-head';
  const name = document.createElement('span');
  name.className = 'rail-side';
  name.textContent = laterality === 'R' ? t('right_breast') : t('left_breast');
  const views = document.createElement('span');
  views.className = 'rail-views';
  views.textContent = side ? side.views.join(' + ') : t('no_views');
  head.append(name, views);

  if (!side) {
    const empty = document.createElement('p');
    empty.className = 'card-miss';
    empty.textContent = t('nothing_uploaded');
    node.replaceChildren(head, empty);
    return;
  }

  const agreement = document.createElement('p');
  agreement.className = 'agreement';
  agreement.dataset.state = side.comparison.wide_gap ? 'wide' : side.comparison.state;
  agreement.textContent = comparisonNote(side.comparison);

  node.replaceChildren(
    head,
    buildVerdictCard(t('method_landmark'), 'landmark', side.landmark),
    buildVerdictCard(t('method_mask'), 'mask', side.mask),
    agreement,
  );
}

function buildVerdictCard(title, route, result) {
  const card = document.createElement('section');
  card.className = 'verdict-card';

  const header = document.createElement('header');
  const tag = document.createElement('span');
  tag.className = 'method-tag';
  const swatch = document.createElement('span');
  swatch.className = 'swatch';
  swatch.style.background = appearance.colorOf(ROUTE_STYLE[route]);
  tag.append(swatch, document.createTextNode(title));
  header.appendChild(tag);

  if (result.ok) {
    const pill = document.createElement('span');
    pill.className = 'verdict-pill';
    pill.dataset.verdict = result.verdict;
    pill.textContent = verdictText(result.verdict);
    header.appendChild(pill);
  }
  card.appendChild(header);

  if (!result.ok) {
    const miss = document.createElement('p');
    miss.className = 'card-miss';
    miss.textContent = reason(result.reason);
    card.appendChild(miss);
    return card;
  }

  const delta = document.createElement('div');
  delta.className = 'delta';
  const value = document.createElement('span');
  value.className = 'delta-value';
  value.dataset.verdict = result.verdict;
  value.textContent = result.delta_mm.toFixed(1);
  const unit = document.createElement('span');
  unit.className = 'delta-unit';
  unit.textContent = 'mm';
  const label = document.createElement('span');
  label.className = 'delta-label';
  label.textContent = 'ΔD';
  delta.append(value, unit, label);

  const pairs = document.createElement('dl');
  pairs.className = 'pair';
  for (const [term, description] of [
    [t('pnl_mlo'), `${result.pnl_mm.toFixed(1)} mm`],
    [t('wall_cc'), `${result.cc_mm.toFixed(1)} mm`],
    [t('threshold'), `≤ ${result.threshold_mm} mm`],
  ]) {
    const dt = document.createElement('dt');
    dt.textContent = term;
    const dd = document.createElement('dd');
    dd.textContent = description;
    pairs.append(dt, dd);
  }

  card.append(delta, pairs);
  return card;
}

function setDrawer(open) {
  ui.criteriaDrawer.hidden = !open;
  ui.criteriaToggle.setAttribute('aria-expanded', String(open));
  // The board gives up height when the drawer appears, so the tiles must re-fit.
  redrawTiles();
}

function renderCriteria(study) {
  const summary = (study.criteria || {}).summary || {};
  ui.scoreChip.textContent = summary.score === null || summary.score === undefined ? '—' : String(summary.score);
  ui.scoreChip.dataset.grade = summary.worst || 'na';

  const groups = [];
  for (const viewKey of VIEW_ORDER) {
    const image = study.images.find((entry) => entry.view_key === viewKey);
    if (image && (image.criteria || []).length) groups.push([viewKey, image.criteria]);
  }
  const studyItems = (study.criteria || {}).study || [];
  if (studyItems.length) groups.push([t('both_breasts'), studyItems]);

  const counts = summary.counts || {};
  ui.criteriaCounts.textContent = groups.length
    ? t('criteria_counts', { g: counts.good || 0, a: counts.acceptable || 0, p: counts.poor || 0 })
    : '';
  ui.criteriaEmpty.hidden = groups.length > 0;

  ui.criteriaGrid.replaceChildren(...groups.map(([title, items]) => {
    const card = document.createElement('section');
    card.className = 'criteria-card';

    const head = document.createElement('h3');
    head.textContent = title;
    card.appendChild(head);

    const list = document.createElement('ul');
    for (const item of items) {
      const row = document.createElement('li');
      row.className = 'criterion';
      row.dataset.grade = item.grade;
      const detail = criterionDetail(item);
      if (detail) row.title = detail;

      const dot = document.createElement('span');
      dot.className = 'grade-dot';

      const name = document.createElement('span');
      name.className = 'criterion-name';
      name.append(document.createTextNode(criterionName(item)));

      if (item.estimated) {
        const mark = document.createElement('span');
        mark.className = 'est-mark';
        mark.textContent = t('est');
        mark.title = t('est_title');
        name.appendChild(mark);
      }

      const value = document.createElement('span');
      value.className = 'criterion-value';
      value.textContent = item.value === null || item.value === undefined
        ? gradeText(item.grade) : `${item.value}${item.unit}`;

      row.append(dot, name, value);
      list.appendChild(row);
    }
    card.appendChild(list);
    return card;
  }));
}

/** The worst grade among a view's criteria, for the dot in its tile header. */
function worstGrade(image) {
  let worst = null;
  for (const item of image.criteria || []) {
    if (item.grade === 'poor') return 'poor';
    if (item.grade === 'acceptable') worst = 'acceptable';
    else if (item.grade === 'good' && !worst) worst = 'good';
  }
  return worst;
}

function renderFoot(study) {
  const bits = [t('foot_study', { id: study.id }), t('foot_views', { n: study.images.length })];
  const vendor = study.images.map((image) => image.manufacturer).filter(Boolean)[0];
  if (vendor) bits.push(vendor);
  if (study.edited) bits.push(t('foot_edited', { time: formatTime(study.edited) }));
  ui.studyMeta.textContent = bits.join(' · ');
}

function redrawTiles() {
  for (const [viewKey, { canvas, wrap, image }] of state.tiles) {
    const bitmap = state.bitmaps.get(viewKey);
    if (!bitmap) continue;

    // The wrapper owns the available box; the canvas takes the aspect ratio of the tissue rather than of the
    // padded square, which is what recovers the ~30% of every render that preprocessing spends on black.
    const rect = wrap.getBoundingClientRect();
    if (rect.width < 8 || rect.height < 8) continue;

    const [bx1, by1, bx2, by2] = boxOf(viewKey);
    const boxWidth = bx2 - bx1;
    const boxHeight = by2 - by1;
    const fit = Math.min(rect.width / boxWidth, rect.height / boxHeight);
    const cssWidth = Math.floor(boxWidth * fit);
    const cssHeight = Math.floor(boxHeight * fit);
    canvas.style.width = `${cssWidth}px`;
    canvas.style.height = `${cssHeight}px`;

    const dpr = window.devicePixelRatio || 1;
    const width = Math.round(cssWidth * dpr);
    const height = Math.round(cssHeight * dpr);
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }

    const view = viewOf(viewKey);
    const base = width / boxWidth; // device px per image px at fit
    const scale = base * view.zoom;
    const tx = ((1 - view.zoom) * width / 2) + view.x * dpr - bx1 * scale;
    const ty = ((1 - view.zoom) * height / 2) + view.y * dpr - by1 * scale;

    const ctx = canvas.getContext('2d');
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.setTransform(scale, 0, 0, scale, tx, ty);
    ctx.imageSmoothingEnabled = view.zoom < 3;
    ctx.filter = appearance.imageFilter();
    ctx.drawImage(bitmap, 0, 0, FRAME, FRAME);
    ctx.filter = 'none';
    drawOverlay(ctx, image, state.visible, scale);
  }
  ui.zoomBadge.textContent = `${Math.round(state.view.zoom * 100)}%`;
  ui.zoomBadge.hidden = state.view.zoom === 1 && !state.view.x && !state.view.y;
}

// A tile is a button that opens the viewer, and also a pannable surface. A press that never moves is a click;
// one that moves is a pan, and must not also open the viewer when the finger lifts.
function attachTileGestures(wrap, viewKey, onClick) {
  let drag = null;

  wrap.addEventListener('pointerdown', (event) => {
    if (event.button > 2) return;
    // Left: click opens the viewer, drag pans. Right / middle: always a pan, never a click.
    drag = { x: event.clientX, y: event.clientY, moved: event.button === 0 ? 0 : Infinity };
    if (event.button !== 0) event.preventDefault();
    wrap.setPointerCapture(event.pointerId);
  });

  wrap.addEventListener('pointermove', (event) => {
    if (!drag) return;
    const dx = event.clientX - drag.x;
    const dy = event.clientY - drag.y;
    drag.moved += Math.abs(dx) + Math.abs(dy);
    const view = viewOf(viewKey);
    view.x += dx;
    view.y += dy;
    drag.x = event.clientX;
    drag.y = event.clientY;
    redrawTiles();
  });

  const release = (event) => {
    if (!drag) return;
    const moved = drag.moved;
    drag = null;
    if (wrap.hasPointerCapture(event.pointerId)) wrap.releasePointerCapture(event.pointerId);
    if (moved < 5) onClick();
  };
  wrap.addEventListener('pointerup', release);
  wrap.addEventListener('pointercancel', () => { drag = null; });

  wrap.addEventListener('wheel', (event) => {
    event.preventDefault();
    const rect = wrap.getBoundingClientRect();
    zoomTiles(event.deltaY < 0 ? 1.15 : 1 / 1.15, viewKey,
              event.clientX - rect.left - rect.width / 2,
              event.clientY - rect.top - rect.height / 2);
  }, { passive: false });

  wrap.addEventListener('dblclick', (event) => { event.preventDefault(); resetViews(); });
  wrap.addEventListener('contextmenu', (event) => event.preventDefault());
}

// ── theme and full screen ──────────────────────────────────────────────────
function toggleTheme() {
  const next = document.documentElement.dataset.bsTheme === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.bsTheme = next;
  try { localStorage.setItem('breastseg.theme', next); } catch (error) { /* blocked storage: this page only */ }
}

function toggleFullscreen() {
  if (document.fullscreenElement) document.exitFullscreen();
  else if (document.documentElement.requestFullscreen) document.documentElement.requestFullscreen().catch(() => {});
}

function zoomTiles(factor, viewKey, cx, cy) {
  const view = viewOf(viewKey);
  const next = Math.min(12, Math.max(1, view.zoom * factor));
  if (next === view.zoom) return;
  const ratio = next / view.zoom;
  // Keep the point under the cursor where it is, measured from the tile centre.
  view.x = cx - ratio * (cx - view.x);
  view.y = cy - ratio * (cy - view.y);
  view.zoom = next;
  if (next === 1) { view.x = 0; view.y = 0; }
  redrawTiles();
}

// ── threshold ──────────────────────────────────────────────────────────────
const applyThreshold = debounce(async () => {
  if (!state.study) return;
  const value = Number(ui.thresholdInput.value);
  if (!Number.isFinite(value) || value < 1 || value > 60) {
    showError(t('threshold_range'));
    ui.thresholdInput.value = String((state.study.rules || {}).threshold_mm ?? 10);
    return;
  }
  if (value === (state.study.rules || {}).threshold_mm) return;
  const studyId = state.study.id;
  try {
    clearError();
    const updated = await api.putRules(studyId, { threshold_mm: value });
    if (!state.study || state.study.id !== studyId) return; // another study opened meanwhile
    state.study = updated;
    renderRails(updated);
    renderNotices(updated);
    renderFoot(updated);
    // The report quotes the threshold and the verdicts; a copy made before the change would print the old ones.
    state.report = null;
    if (!ui.reportPanel.hidden) loadReport();
    refreshHistory();
  } catch (error) {
    showError(error.message);
  }
}, 260);

function nudgeThreshold(step) {
  const current = Number(ui.thresholdInput.value) || 10;
  ui.thresholdInput.value = String(Math.min(60, Math.max(1, Math.round((current + step) * 10) / 10)));
  applyThreshold();
}

// ── report ─────────────────────────────────────────────────────────────────
const REPORT_LANGS = [['en', 'English'], ['tr', 'Türkçe']];

function toggleReport(open) {
  // A browser serving a module it cached earlier is the one failure that looks like a bug in code that is plainly
  // correct on disk. Say what it actually is.
  if (open && typeof api.getReport !== 'function') {
    showError(t('stale_code'));
    return;
  }
  ui.reportPanel.hidden = !open;
  ui.reportToggle.setAttribute('aria-expanded', String(open));
  if (open) loadReport();
}

function renderLangSwitch() {
  ui.reportLang.replaceChildren(...REPORT_LANGS.map(([code, name]) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'btn btn-sm' + (code === reportLanguage() ? ' active' : '');
    button.textContent = name;
    button.addEventListener('click', () => {
      if (code === reportLanguage()) return;
      state.reportLang = code;
      savePref(REPORT_LANG_KEY, code);
      loadReport();
    });
    return button;
  }));
}

function reportSkeleton() {
  const block = document.createElement('div');
  block.className = 'report-skeleton';
  for (let i = 0; i < 7; i += 1) {
    const line = document.createElement('span');
    line.className = 'skeleton-line';
    line.style.width = `${[62, 88, 74, 91, 55, 80, 68][i]}%`;
    block.appendChild(line);
  }
  return block;
}

async function loadReport() {
  if (!state.study) return;
  renderLangSwitch();
  ui.reportBody.replaceChildren(reportSkeleton());
  const studyId = state.study.id;
  const lang = reportLanguage();
  state.report = null;
  try {
    const report = await api.getReport(studyId, lang);
    // A slower answer for another study or language must not overwrite this one.
    if (!state.study || state.study.id !== studyId || reportLanguage() !== lang) return;
    state.report = report;
    renderReport(report);
  } catch (error) {
    showError(error.message);
  }
}

function renderReport(report) {
  ui.reportTitle.textContent = report.title;
  ui.reportFooterNote.textContent = report.footer;

  ui.reportBody.replaceChildren(...report.sections.map((section, index) => {
    const block = document.createElement('section');
    block.className = 'report-section';
    // The first section repeats the document title; the panel header already has it.
    if (index > 0) {
      const heading = document.createElement('h3');
      heading.textContent = section.heading;
      block.appendChild(heading);
    }

    // Rows stay in the order the server wrote them: a note such as "the methods disagree" belongs under its own
    // breast, not above both.
    let list = document.createElement('dl');
    for (const row of section.rows) {
      if (row.text !== undefined) {
        if (list.childElementCount) {
          block.appendChild(list);
          list = document.createElement('dl');
        }
        const note = document.createElement('p');
        note.className = 'report-note';
        note.textContent = row.text;
        block.appendChild(note);
        continue;
      }
      const line = document.createElement('div');
      line.className = 'report-row' + (row.strong ? ' strong' : '');
      if (row.grade) line.dataset.grade = row.grade;
      const term = document.createElement('dt');
      term.textContent = row.label;
      const value = document.createElement('dd');
      value.textContent = row.value || '';
      line.append(term, value);
      list.appendChild(line);
    }
    if (list.childElementCount) block.appendChild(list);
    return block;
  }));
}

async function copyReport() {
  if (!state.report) return;
  try {
    await navigator.clipboard.writeText(state.report.text);
    ui.reportCopy.textContent = t('copied');
    setTimeout(() => { ui.reportCopy.textContent = t('copy'); }, 1400);
  } catch (error) {
    showError(t('clipboard_refused'));
  }
}

function saveBlob(blob, name) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function downloadReport() {
  if (!state.report) return;
  saveBlob(new Blob([state.report.text], { type: 'text/plain;charset=utf-8' }),
           `report-${state.study.id}-${state.report.lang}.txt`);
}

// ── editor ─────────────────────────────────────────────────────────────────
function openEditor() {
  if (!state.study) return;
  const first = VIEW_ORDER.find((key) => state.study.images.some((i) => i.view_key === key));
  editor.load(state.study, state.bitmaps, first, (key) => boxOf(key));
}

function closeEditor() {
  if (editor.dirty && !window.confirm(t('discard_confirm'))) return;
  editor.close();
}

function revertEditorView() {
  if (!state.study) return;
  const original = state.study.images.find((image) => image.view_key === editor.viewKey);
  editor.revertView(original);
}

async function saveLabels() {
  if (!state.study) return;
  const button = el('editor-save');
  const payload = editor.payload();
  if (!payload.images.length) { // nothing was changed
    editor.dirty = false;
    editor.close();
    return;
  }
  button.disabled = true;
  button.textContent = t('saving');
  button.title = '';
  let failure = null;
  try {
    // The study the labels were loaded from, not whatever the page shows now.
    const updated = await api.putLabels(editor.studyId, payload);
    editor.dirty = false;
    editor.close();
    await showStudy(updated);
    refreshHistory();
  } catch (error) {
    failure = error.message;
    showError(error.message);
  } finally {
    button.disabled = false;
    // The error banner sits under the full-screen editor; say it on the button too.
    button.textContent = failure ? t('save_failed') : t('save');
    button.title = failure || '';
  }
}

// ── history ────────────────────────────────────────────────────────────────
async function refreshHistory() {
  try {
    const { studies } = await api.listStudies();
    state.history = studies;
    renderHistory(studies);
  } catch (error) {
    showError(error.message);
  }
}

function renderHistory(studies) {
  ui.recentEmpty.hidden = studies.length > 0;
  ui.history.replaceChildren(...studies.map(buildHistoryItem));
}

function buildHistoryItem(row) {
  const item = document.createElement('li');
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'history-item';
  button.dataset.id = row.id;

  const dot = document.createElement('span');
  dot.className = 'history-dot';
  if (row.overall) {
    dot.dataset.verdict = row.overall;
    dot.title = verdictText(row.overall);
  }

  const body = document.createElement('span');
  body.className = 'history-body';
  const views = document.createElement('span');
  views.className = 'history-views';
  views.textContent = (row.views || []).join('  ') || t('views_n', { n: row.images });
  body.appendChild(views);

  button.append(dot, body);
  if (row.flagged) {
    const flag = document.createElement('span');
    flag.className = 'history-flag';
    flag.textContent = '▲';
    flag.title = t('flag_title');
    button.appendChild(flag);
  }
  const time = document.createElement('span');
  time.className = 'history-time';
  time.textContent = formatTime(row.created);
  button.appendChild(time);

  button.addEventListener('click', () => openStudy(row.id));
  item.appendChild(button);
  return item;
}

async function openStudy(studyId) {
  if (state.busy) return;
  clearError();
  state.busy = true;
  ui.progressText.textContent = t('opening');
  show('processing');
  try {
    state.bitmaps.clear();
    const study = await api.getStudy(studyId);
    ui.progressText.textContent = t('loading_views');
    await showStudy(study);
  } catch (error) {
    showError(error.message);
    show(state.study ? 'review' : 'landing');
  } finally {
    state.busy = false;
  }
}

async function downloadLabels() {
  if (!state.study) return;
  try {
    const labels = await api.getLabels(state.study.id);
    saveBlob(new Blob([JSON.stringify(labels, null, 2)], { type: 'application/json' }), `labels-${state.study.id}.json`);
  } catch (error) {
    showError(error.message);
  }
}

async function deleteCurrent() {
  if (!state.study) return;
  if (!window.confirm(t('delete_confirm'))) return;
  try {
    await api.deleteStudy(state.study.id);
    startNewStudy();
  } catch (error) {
    showError(error.message);
  }
}

boot();

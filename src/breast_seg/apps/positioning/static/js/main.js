// Application shell: a splash while the models load, then three screens (landing, opening, review) plus the
// label editor and the viewer as full-screen overlays. State lives in one object; every screen change goes
// through show(), and every translated string is rebuilt by relabel() when the language changes.

import { api, fileList } from './api.js';
import { drawOverlay, FRAME, LAYERS, layerColor, layerName } from './overlay.js';
import { appearance } from './appearance.js';
import { Lightbox } from './lightbox.js';
import { Editor } from './editor.js';
import {
  buildDataset, DICOM_EXT, PICTURE, pathOf, studyKey, studyUpload, usableFiles, VIEW_IN_PATH,
} from './dataset.js';
import {
  criterionDetail, criterionName, formatMm, formatNumber, formatValue, language, locale, onLanguage, reason,
  savedLanguage, setLanguage, t, uploadNote,
} from './i18n.js';

// Hanging order: the right breast sits on the reader's left and the left breast on the reader's right, as if
// facing the patient; MLO above CC.
const VIEW_ORDER = ['R-MLO', 'L-MLO', 'R-CC', 'L-CC'];
const DEFAULT_LAYERS = ['breast', 'pectoral', 'nipple', 'points', 'measure_landmark', 'measure_mask'];
const LAYER_STORAGE_KEY = 'mammo.layers.v1';
const LAYOUT_STORAGE_KEY = 'mammo.layout.v1';
const CROP_STORAGE_KEY = 'mammo.crop.v1';
const PAIRING_STORAGE_KEY = 'mammo.pairing.v1';
// The report's language, like the interface language, holds for the browser session (an older version kept it for
// good under the old key, which is removed).
const REPORT_LANG_KEY = 'mammo.reportLang.session.v1';
const OLD_REPORT_LANG_KEY = 'mammo.reportLang.v2';
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
  chooseDataset: el('choose-dataset'),
  datasetInput: el('dataset-input'),
  datasetNav: el('dataset-nav'),
  datasetPrev: el('dataset-prev'),
  datasetNext: el('dataset-next'),
  datasetPick: el('dataset-pick'),
  datasetState: el('dataset-state'),
  datasetName: el('dataset-name'),
  datasetSkipped: el('dataset-skipped'),
  datasetLegend: el('dataset-legend'),
  datasetClose: el('dataset-close'),
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
  reportLang: loadSessionPref(REPORT_LANG_KEY), // null: follow the interface language; a pick holds for the session
  dataset: null, // { name, studies: [{name, files}], index, ids: Map, jobs: Map, failed: Set, finished }
};

const LAYOUT_NAMES = { '1x1': 'layout_1x1', '2x1': 'layout_2x1', '1x2': 'layout_1x2', '2x2': 'layout_2x2' };

const COLOR_FIELDS = [
  ['breast-tissue', 'c_breast'],
  ['pectoral', 'c_pectoral'],
  ['nipple', 'c_nipple'],
  ['nipple_point', 'c_landmarks'],
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

function loadSessionPref(key) {
  try { return sessionStorage.getItem(key); } catch (error) { return null; }
}

function saveSessionPref(key, value) {
  try { sessionStorage.setItem(key, value); } catch (error) { /* blocked storage: this page only */ }
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

// Every tile zooms and pans on its own by default; "Synchronise zoom & pan" shares one transform between them, for
// looking at the same place on the left and the right.
state.view = { zoom: 1, x: 0, y: 0 };
state.syncViews = false;
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

/** The views a study holds, in grid order. */
function presentViews(study) {
  if (!study) return VIEW_ORDER;
  const present = VIEW_ORDER.filter((key) => study.images.some((image) => image.view_key === key));
  return present.length ? present : VIEW_ORDER;
}

/**
 * The view keys shown on each page, for the current layout and pairing. Only the views the study holds take a pane: a
 * study of one breast (or of one projection) fills the screen with its two views, a single view fills it alone, and a
 * page that would hold none is left out. A four-view grid missing one view keeps its hanging order, with the missing
 * view marked "not uploaded" in its place.
 */
function pages(study = state.study) {
  const present = presentViews(study);
  const size = capacity();
  if (size >= 4) return [present.length === 3 ? VIEW_ORDER : present];
  if (size === 1) return present.map((key) => [key]);
  const paired = PAIRINGS[state.pairing].pages.map((page) => page.filter((key) => present.includes(key)));
  return paired.filter((page) => page.length);
}

/** The grid of a page with `count` panes: the layout's, or a smaller one when the page holds fewer views. */
function gridOf(count) {
  if (count >= capacity()) return state.layout;
  if (count <= 1) return { cols: 1, rows: 1 };
  if (count === 2) return state.layout.cols === 1 ? { cols: 1, rows: 2 } : { cols: 2, rows: 1 };
  return state.layout; // three of four: the 2 x 2 hanging, with the missing view in its place
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

/** Switch overlay layers (the review's toggles, the viewer and the editor all draw by this one set). */
function setLayers(ids, on) {
  for (const id of ids) {
    if (on) state.visible.add(id);
    else state.visible.delete(id);
  }
  saveLayerPrefs();
  buildLayerBar();
  redrawTiles();
  lightbox.render();
}

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
  hide: el('editor-hide'),
  clean: el('editor-clean'),
}, {
  onSave: saveLabels,
  onClose: closeEditor,
  onClean: (viewKey, polygons) => api.cleanView(editor.studyId, viewKey, polygons),
  layers: { has: (id) => state.visible.has(id), set: setLayers },
});

// ── small helpers ──────────────────────────────────────────────────────────
function saveLayerPrefs() { savePref(LAYER_STORAGE_KEY, JSON.stringify([...state.visible])); }

function showError(message, tone = 'error') {
  ui.bannerText.textContent = message;
  ui.banner.dataset.tone = tone;
  ui.banner.hidden = false;
}

function clearError() { ui.banner.hidden = true; }

/** Remove `message` from the banner, and only that: a notice shown since then stays. */
function clearErrorIf(message) {
  if (!ui.banner.hidden && ui.bannerText.textContent === message) clearError();
}

/** A notice that goes away by itself, unless something else has taken the banner meanwhile. */
function showBriefly(message, tone, ms = 3000) {
  showError(message, tone);
  return setTimeout(() => clearErrorIf(message), ms);
}

function show(screen) {
  state.screen = screen;
  for (const [name, node] of Object.entries(ui.screens)) node.hidden = name !== screen;
  ui.reviewTools.hidden = screen !== 'review';
  if (screen !== 'review') {
    ui.reportPanel.hidden = true;
    ui.notesPop.hidden = true;
  }
}

/** Date and time, for a list where rows from different days sit together. */
function formatDateTime(seconds) {
  if (!seconds) return '—';
  return new Date(seconds * 1000).toLocaleString(locale(), {
    day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit',
  });
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
    return comparison.wide_gap ? t('cmp_wide', { gap: formatNumber(comparison.gap_mm) }) : t('cmp_agree');
  }
  return t(comparison.state === 'partial' ? 'cmp_partial' : 'cmp_none');
}

// ── boot: splash until the models are loaded ───────────────────────────────
async function boot() {
  try { localStorage.removeItem(OLD_REPORT_LANG_KEY); } catch (error) { /* blocked storage: nothing was kept */ }
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
  renderDatasetPicker();
  renderDatasetNav();
  renderPairingButtons();
  if (!colorEditing) buildColorGrid();
  renderStatus();
  renderRetention();
  relabelClear();
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
    input.addEventListener('change', () => setLayers([layer.id], input.checked));

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
    if (files.length) uploadPicked(files);
  });
  ui.chooseDataset.addEventListener('click', (event) => { event.stopPropagation(); ui.datasetInput.click(); });
  ui.datasetInput.addEventListener('change', () => {
    const files = [...ui.datasetInput.files];
    ui.datasetInput.value = '';
    if (files.length) openDataset(files);
  });
  ui.datasetPrev.addEventListener('click', () => stepStudy(-1));
  ui.datasetNext.addEventListener('click', () => stepStudy(1));
  ui.datasetPick.addEventListener('change', () => {
    ui.datasetPick.blur(); // PageDown must move the study, not the native list
    goToStudy(Number(ui.datasetPick.value));
  });
  ui.datasetClose.addEventListener('click', () => { closeDataset(); startNewStudy(); });

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
    if (files.length) uploadPicked(files);
  });
  window.addEventListener('dragover', (event) => event.preventDefault());
  window.addEventListener('drop', (event) => event.preventDefault());

  ui.uploadCancel.addEventListener('click', cancelUpload);
  ui.newStudy.addEventListener('click', () => { closeDataset(); startNewStudy(); });
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
    if (state.dataset && !isTyping(event.target) && !panelOpen()) {
      const forward = event.key === 'PageDown' || (event.shiftKey && event.key === 'ArrowRight');
      const back = event.key === 'PageUp' || (event.shiftKey && event.key === 'ArrowLeft');
      if (forward || back) { event.preventDefault(); stepStudy(forward ? 1 : -1); return; }
    }
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
    if (id && (!state.study || state.study.id !== id)) openStored(id);
  });
}

/** Focus is somewhere a key belongs to the field, not to the page. */
function isTyping(target) {
  return target instanceof HTMLInputElement || target instanceof HTMLSelectElement
    || target instanceof HTMLTextAreaElement || Boolean(target && target.isContentEditable);
}

/**
 * Something that covers the review or takes the keyboard is open: page-level shortcuts stay quiet. The criteria
 * drawer is not one of them: it sits under the board, next to the images, and is read while stepping through studies.
 */
function panelOpen() {
  return [el('editor'), el('lightbox'), ui.reportPanel, ui.notesPop, ui.layoutPop, ui.stylePop]
    .some((node) => node && !node.hidden);
}

// ── upload ─────────────────────────────────────────────────────────────────
const uploadLimit = () => (state.health && state.health.upload && state.health.upload.max_files) || 40;

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
  const limit = uploadLimit();
  const usable = usableFiles(files);

  // Prefer files whose path names a view, but only among things that could be images — a labels/ sidecar inside
  // RMLO/ matches the view too.
  const byView = usable.filter((file) => VIEW_IN_PATH.test(pathOf(file)));
  if (byView.length) return byView.slice(0, limit);
  const byExt = usable.filter((file) => DICOM_EXT.test(pathOf(file)));
  return (byExt.length ? byExt : usable).slice(0, limit);
}

/**
 * Why nothing of `files` can be sent, in words that name the files: pictures (PNG/JPG cannot be measured), empty
 * files (0 bytes; a folder dropped where files were expected also arrives as one), or nothing that could be DICOM.
 */
function nothingToSend(files) {
  const pictures = files.filter((file) => file.size > 0 && PICTURE.test(pathOf(file)));
  if (pictures.length) return t('only_dicom', { names: fileList(pictures) });
  const empty = files.filter((file) => file.size === 0);
  if (empty.length) return t(empty.length === 1 ? 'empty_file_1' : 'empty_files', { files: fileList(empty) });
  return t('no_dicom_files', { n: files.length });
}

/** A folder or a drop: the files worth sending, or a message saying why there are none. */
function uploadPicked(files) {
  const picked = pickLikelyDicoms(files);
  if (picked.length) upload(picked);
  else showError(nothingToSend(files));
}

// "Clear" deletes every study on the server, so it takes two deliberate clicks: the first counts what is stored
// and says so, the second (not part of a double click, within a few seconds) deletes exactly that many.
const CLEAR_ARM_MS = 8000;
const CLEAR_MIN_GAP_MS = 600;
let clearArmed = null;

let clearCounting = false;

function disarmClear() {
  if (clearArmed) clearTimeout(clearArmed.timer);
  clearArmed = null;
  ui.clearHistory.textContent = t('clear');
  ui.clearHistory.classList.remove('armed');
}

// The count includes studies a dataset analysed ahead that nobody opened: they are stored, though not listed.
function clearQuestion(total, unopened) {
  return t('clear_confirm', { n: total }) + (unopened ? ` ${t('clear_unopened', { k: unopened })}` : '');
}

/**
 * An armed "clear" in the new language: the button's static label was just reset to "Clear" (data-i18n), so the
 * armed label and its question are written again, still armed for the same count and the same time.
 */
function relabelClear() {
  if (!clearArmed) return;
  const { count, unopened, message: before } = clearArmed;
  const message = clearQuestion(count, unopened);
  ui.clearHistory.textContent = t('clear_arm', { n: count });
  if (!ui.banner.hidden && ui.bannerText.textContent === before) ui.bannerText.textContent = message;
  clearArmed.message = message;
}

async function clearHistory() {
  if (!clearArmed) {
    if (clearCounting) return; // a second first click while the first is still counting: one question, one timer
    clearCounting = true;
    let total;
    let unopened;
    try {
      ({ total, unopened } = await api.listStudies({ limit: 0 })); // the counts only: no record is read for them
    } catch (error) {
      showError(error.message);
      return;
    } finally {
      clearCounting = false;
    }
    if (!total) {
      showBriefly(t('clear_nothing'), 'info');
      return;
    }
    disarmClear(); // never two timers
    const message = clearQuestion(total, unopened);
    // When the time is up only this question is taken down (in whichever language it is shown by then): a notice
    // that appeared meanwhile stays.
    const timer = setTimeout(() => {
      const shown = clearArmed ? clearArmed.message : message;
      disarmClear();
      clearErrorIf(shown);
    }, CLEAR_ARM_MS);
    clearArmed = { count: total, unopened, at: Date.now(), timer, message };
    ui.clearHistory.textContent = t('clear_arm', { n: total });
    ui.clearHistory.classList.add('armed');
    showError(message);
    return;
  }
  if (Date.now() - clearArmed.at < CLEAR_MIN_GAP_MS) return; // the second half of a double click
  const { count: expected, message } = clearArmed;
  disarmClear();
  clearErrorIf(message);
  try {
    // The server refuses when the count changed since the first click, so nothing unseen is deleted.
    const { deleted } = await api.deleteAll(expected);
    if (state.dataset) {
      // The dataset list stays, as it was when the folder was opened: no record is left, so no study is opened,
      // stored or removed any more, and every study is analysed again when it is reached. Only an analysis still
      // running is kept waiting for: its record is stored after this clear and is used when it arrives.
      const dataset = state.dataset;
      for (const index of [...dataset.jobs.keys()]) {
        if (dataset.ids.has(index)) dataset.jobs.delete(index);
      }
      dataset.ids.clear();
      dataset.opened.clear();
      dataset.removed.clear();
      dataset.finished = false;
      renderDatasetPicker();
      renderDatasetNav();
    }
    if (state.study) startNewStudy();
    else await refreshHistory();
    showBriefly(deleted === 1 ? t('deleted_1') : t('deleted_n', { n: deleted }));
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
  if (!accepted.length) { showError(nothingToSend(files)); return; }
  const empty = files.filter((file) => file.size === 0);

  // A study uploaded on its own is not part of the open dataset: the dataset is left, as with "New study", so its
  // bar does not stay over a study it does not hold.
  closeDataset();
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
    // The study is shown; the files that were left out for being empty are named, so none goes missing silently.
    if (empty.length) showError(t('empty_skipped', { files: fileList(empty) }), 'info');
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

// ── dataset ────────────────────────────────────────────────────────────────
// A folder of many studies, reviewed one study at a time. Studies are analysed only when reached; the next one is
// analysed in the background while the current one is on screen, so moving on is usually instant.
//
// The page never deletes a record on its own account: another tab or another reviewer may have opened it. A study
// analysed ahead is stored as "not opened yet" and is not listed among the recent studies; it is listed from the
// moment any page puts it on screen for review (openStudy tells the server), by whatever route. A record nobody
// opens is removed by the retention window like any other. Leaving the dataset or the page changes nothing on the
// server, and an analysis still running then simply finishes.
//
// Every dataset study is stored under its study key (dataset.js). Opening the same folder again asks the server
// which of the keys it holds and shows those studies from their stored records, corrections included, without
// uploading them a second time; they carry the • mark until they are opened in this session. A study that was
// stored from an incomplete upload (a file failed in a way a retry can change) is not offered again unless a reviewer
// edited it: it is analysed afresh, and the new record replaces the old one.

async function openDataset(files) {
  if (state.busy) { showError(t('busy_wait')); return; }
  clearError();
  // The folder is read first: the dataset that is open stays open until the new one is known to hold a study.
  const back = () => show(state.study ? 'review' : 'landing');
  state.busy = true;
  ui.progressText.textContent = t('ds_scanning', { d: 0, n: files.length });
  show('processing');
  let dataset;
  let found = {};
  try {
    dataset = await buildDataset(files, (done, total) => {
      ui.progressText.textContent = t('ds_scanning', { d: done, n: total });
    });
    // The UID has done its work once the key is made: it is not kept.
    const salt = (state.health && state.health.dataset && state.health.dataset.key_salt) || '';
    dataset.studies = dataset.studies.map(({ name, files: list, uid }) => (
      { name, files: list, key: studyKey(salt, uid, list) }));
    const keys = dataset.studies.map((study) => study.key).filter(Boolean);
    if (keys.length) {
      // Without an answer every study is simply analysed when it is reached, as if nothing were stored.
      found = await api.lookupStudies(keys).then((reply) => (reply && reply.found) || {}, () => ({}));
    }
  } catch (error) {
    back();
    showError(t('ds_unreadable', { msg: (error && error.message) || String(error) }));
    return;
  } finally {
    state.busy = false;
  }
  if (!dataset.studies.length) {
    back();
    showError(t('ds_empty'));
    return;
  }
  closeDataset();
  state.dataset = {
    ...dataset,
    index: 0,                 // the study on screen (moves only once the next one is shown)
    cursor: 0,                // the last study asked for: Next / Previous step from here, also past one that failed
    target: null,             // the study being opened
    moving: false,
    finished: false,
    ids: new Map(),           // index -> stored study id
    jobs: new Map(),          // index -> analysis promise
    opened: new Set(),
    failed: new Set(),
    removed: new Set(),
    aborter: new AbortController(),
  };
  state.dataset.studies.forEach((study, index) => {
    const stored = study.key && found[study.key];
    if (typeof stored !== 'string') return;
    state.dataset.ids.set(index, stored);       // already on the server: opened from its record, not uploaded
    state.dataset.jobs.set(index, Promise.resolve(stored));
  });
  renderDatasetPicker();
  goToStudy(0);
}

/** Leave the dataset. Nothing is removed from the server: what was analysed ahead stays, not opened yet. */
function closeDataset() {
  const dataset = state.dataset;
  state.dataset = null;
  ui.datasetNav.hidden = true;
  if (dataset) dataset.aborter.abort(); // stops the study that is being opened, not an analysis on the server
}

/** Forget a study's stored record, so reaching it again analyses it again. */
function forgetStudy(dataset, index) {
  dataset.ids.delete(index);
  dataset.jobs.delete(index);
}

/** How many files of dataset study `index` are not sent, because the server takes only so many per study. */
function leftOut(dataset, index) {
  const study = dataset.studies[index];
  return study ? studyUpload(study.files, uploadLimit()).left : 0;
}

/** The stored study id of dataset study `index`, analysing it first if nobody has yet. */
function ensureStudy(index) {
  const dataset = state.dataset;
  if (!dataset || index < 0 || index >= dataset.studies.length) return Promise.resolve(null);
  if (!dataset.jobs.has(index)) {
    // The files were sorted out when the dataset was built; only the server's per-upload cap applies here, with
    // the files whose path names a view first, as in a single-study upload (goToStudy says when some stay behind).
    const study = dataset.studies[index];
    const { files, left } = studyUpload(study.files, uploadLimit());
    // Stored as "not opened yet": the reviewer may never get to it. openStudy says when it is on screen.
    const job = api.createStudy(files, { studyKey: study.key, unopened: true, leftOut: left }).then((record) => {
      if (state.dataset !== dataset || dataset.jobs.get(index) !== job) {
        // Nobody here is waiting for this record any more (the dataset was closed, or the study was forgotten
        // while it was being analysed). It is left where it is: stored, not opened yet, and found by its key when
        // the folder is opened again.
        throw new Error(t('cancelled'));
      }
      dataset.ids.set(index, record.id);
      updateDatasetOption(index);
      return record.id;
    });
    job.catch(() => { if (dataset.jobs.get(index) === job) dataset.jobs.delete(index); }); // tried again when reached
    dataset.jobs.set(index, job);
  }
  return dataset.jobs.get(index);
}

function stepStudy(step) {
  const dataset = state.dataset;
  if (!dataset || dataset.moving) return;
  // From the last study asked for, not from the one on screen: a study that fails is stepped over in either
  // direction instead of being tried again for ever, and Finish works when the last one failed.
  const target = dataset.cursor + step;
  if (target >= dataset.studies.length) { finishDataset(); return; }
  if (target >= 0) goToStudy(target);
}

async function goToStudy(index, retried = false) {
  const dataset = state.dataset;
  if (!dataset) return;
  // Every way out that does not move puts the bar back on the study it is on: the list would otherwise keep
  // showing the study that was picked while another one stays on screen.
  if (dataset.moving || !(index >= 0 && index < dataset.studies.length)) { renderDatasetNav(); return; }
  if (state.busy) { showError(t('busy_wait')); renderDatasetNav(); return; }
  dataset.moving = true;
  dataset.target = index;
  dataset.cursor = index;
  clearError();
  renderDatasetNav();
  ui.progressText.textContent = t('ds_analysing', { i: index + 1, n: dataset.studies.length });
  show('processing');

  let outcome = { ok: false };
  try {
    let studyId;
    try {
      studyId = await ensureStudy(index);
    } catch (error) {
      if (state.dataset !== dataset) return; // closed while it was analysed
      dataset.failed.add(index);
      updateDatasetOption(index);
      show(state.study ? 'review' : 'landing');
      showError(t('ds_failed', { i: index + 1, name: dataset.studies[index].name, msg: error.message }));
      return;
    }
    if (state.dataset !== dataset) return;
    outcome = await openStudy(studyId, { signal: dataset.aborter.signal });
  } finally {
    dataset.moving = false;
    dataset.target = null;
    if (state.dataset === dataset) renderDatasetNav();
  }
  if (state.dataset !== dataset) return;

  if (outcome.ok) {
    dataset.index = index;
    dataset.opened.add(index);
    dataset.failed.delete(index);
    dataset.removed.delete(index);
    updateDatasetOption(index);
    renderDatasetNav();
    const left = leftOut(dataset, index);
    if (left) {
      const total = dataset.studies[index].files.length;
      showError(t('ds_left_out', { n: total, limit: total - left, left }), 'info');
    }
    ensureStudy(index + 1).catch(() => {}); // analyse the next study while this one is read
    return;
  }
  if (outcome.status === 404) {
    // The stored record is gone (deleted, or past the retention window): analyse the files again, once.
    forgetStudy(dataset, index);
    if (!retried) { goToStudy(index, true); return; }
  }
  dataset.failed.add(index);
  updateDatasetOption(index);
  renderDatasetNav();
}

function finishDataset() {
  const dataset = state.dataset;
  if (!dataset) return;
  dataset.finished = true;
  renderDatasetNav();
  const total = dataset.studies.length;
  const counts = datasetCounts(dataset);
  const parts = ['opened', 'analysed', 'failed', 'removed', 'unseen']
    .filter((kind) => counts[kind]).map((kind) => t(`ds_part_${kind}`, { c: counts[kind] }));
  showError(counts.opened === total
    ? t('ds_finished_all', { n: total })
    : t('ds_finished_counts', { n: total, parts: parts.join(', ') }), 'info');
}

/**
 * What became of one study of the dataset. The list marks and the finish summary both read this, so they agree:
 * opened (✓, shown on screen in this session), analysed (•, a record is on the server — analysed ahead just now, or
 * kept from an earlier session — but it has not been opened in this one), failed (✕), removed (–), or unseen.
 */
function datasetState(dataset, index) {
  if (dataset.removed.has(index)) return 'removed';
  if (dataset.opened.has(index)) return 'opened';
  if (dataset.failed.has(index)) return 'failed';
  if (dataset.ids.has(index)) return 'analysed';
  return 'unseen';
}

function datasetCounts(dataset) {
  const counts = { opened: 0, analysed: 0, failed: 0, removed: 0, unseen: 0 };
  dataset.studies.forEach((_, index) => { counts[datasetState(dataset, index)] += 1; });
  return counts;
}

const DATASET_MARKS = { opened: ' ✓', analysed: ' •', failed: ' ✕', removed: ' –', unseen: '' };

function datasetOptionText(dataset, index) {
  const mark = DATASET_MARKS[datasetState(dataset, index)];
  return t('ds_option', { i: index + 1, n: dataset.studies.length, name: dataset.studies[index].name }) + mark;
}

/** The whole study list: built once per dataset and per language, so an open list is never rebuilt under the user. */
function renderDatasetPicker() {
  const dataset = state.dataset;
  if (!dataset) return;
  ui.datasetPick.title = t('ds_mark_title');
  ui.datasetLegend.textContent = t('ds_legend'); // what the marks of the list mean, without hovering
  ui.datasetPick.replaceChildren(...dataset.studies.map((_, index) => {
    const option = document.createElement('option');
    option.value = String(index);
    option.textContent = datasetOptionText(dataset, index);
    return option;
  }));
  ui.datasetPick.value = String(dataset.cursor);
}

function updateDatasetOption(index) {
  const dataset = state.dataset;
  const option = dataset && ui.datasetPick.options[index];
  if (option) option.textContent = datasetOptionText(dataset, index);
}

function renderDatasetNav() {
  const dataset = state.dataset;
  ui.datasetNav.hidden = !dataset;
  if (!dataset) return;
  const total = dataset.studies.length;
  const at = dataset.cursor;
  const last = at === total - 1;
  ui.datasetName.textContent = dataset.name;
  ui.datasetName.title = dataset.name;
  ui.datasetName.hidden = !dataset.name;
  const left = leftOut(dataset, at);
  ui.datasetSkipped.textContent = [
    dataset.skipped ? t('ds_skipped', { n: dataset.skipped }) : '',
    left ? t('ds_left_short', { n: dataset.studies[at].files.length, left }) : '',
  ].filter(Boolean).join(' · ');
  if (ui.datasetPick.value !== String(at)) ui.datasetPick.value = String(at);
  ui.datasetPick.disabled = dataset.moving;
  ui.datasetPrev.disabled = dataset.moving || at === 0;
  ui.datasetNext.disabled = dataset.moving || (last && dataset.finished);
  ui.datasetNext.title = last ? t('ds_finish') : t('ds_next_title');
  el('dataset-next-label').textContent = last ? t('ds_finish_btn') : t('ds_next_btn');
  el('dataset-next-mark').textContent = last ? '✓' : '›';
  let tone = '';
  let text = '';
  if (dataset.moving) text = t('ds_working', { i: dataset.target + 1 });
  else if (dataset.removed.has(at) && !state.study) { tone = 'error'; text = t('ds_state_removed'); }
  else if (dataset.failed.has(at)) { tone = 'error'; text = t('ds_state_failed'); }
  else if (dataset.finished) { tone = 'done'; text = t('ds_done', { o: datasetCounts(dataset).opened, n: total }); }
  else if (last) text = t('ds_last');
  ui.datasetState.textContent = text;
  if (tone) ui.datasetState.dataset.tone = tone;
  else delete ui.datasetState.dataset.tone;
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

/** Show a study record; false when `signal` was aborted while its views loaded (nothing is shown then). */
async function showStudy(study, signal) {
  await loadBitmaps(study);
  if (signal && signal.aborted) return false;
  // Another study starts on its first page (its views are not the last one's); the same study, saved, stays put.
  if (!state.study || state.study.id !== study.id) state.page = 0;
  state.study = study;
  state.report = null;

  ui.thresholdInput.value = String((study.rules || {}).threshold_mm ?? 10);
  renderReview(study);

  show('review');
  if (!ui.reportPanel.hidden) loadReport();
  if (location.hash.replace('#', '') !== study.id) history.replaceState(null, '', `#${study.id}`);
  redrawTiles();
  return true;
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
  // The card's own sentence for each breast that needs a look, so the card and the notes never word the same
  // situation differently: a verdict disagreement, a wide gap under the same verdict, or a single method.
  for (const side of (study.summary || {}).sides || []) {
    const comparison = side.comparison || {};
    if (comparison.state === 'agree' && !comparison.wide_gap) continue;
    const name = side.laterality === 'R' ? t('right_breast') : t('left_breast');
    const tone = comparison.state === 'disagree' || comparison.wide_gap ? 'warn' : 'info';
    notices.push({ text: `${name}: ${comparisonNote(comparison)}`, tone });
  }
  const unknownScale = study.images.filter((image) => !image.pixel_spacing_known);
  if (unknownScale.length) {
    notices.push({ tone: 'info', text: t('notice_spacing', { views: unknownScale.map((i) => i.view_key).join(', ') }) });
  }
  // A view measured without its nipple (the contour fallback): its verdict rests on an estimate.
  for (const viewKey of VIEW_ORDER) {
    const image = study.images.find((entry) => entry.view_key === viewKey);
    if (image && isEstimated(image)) notices.push({ tone: 'warn', text: t(`notice_estimated_${image.view}`, { view: viewKey }) });
  }
  // Files the upload stepped over are worth recording but are not findings; they sit at the bottom of the list
  // in a quieter style. Written in the reader's language from the codes stored with them; a record stored before
  // the codes were kept shows its English sentences.
  const coded = Array.isArray(study.problem_codes) && study.problem_codes.length === (study.problems || []).length
    ? study.problem_codes : [];
  (study.problems || []).forEach((problem, index) => {
    const entry = coded[index];
    const skipped = entry ? SKIP_CODES.has(entry.code) : /\bskipped\b/.test(problem);
    notices.push({ text: uploadNote(problem, entry), tone: skipped ? 'skip' : 'warn' });
  });

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

// Upload notes about a file that was stepped over by design (not a view of the study), shown in the quieter style.
const SKIP_CODES = new Set(['not_an_image', 'raster_image', 'duplicate_view']);

/** The segmentation route of the view was measured without a nipple (the contour fallback). */
const isEstimated = (image) => Boolean(((image.measures || {}).mask || {}).estimated);

function setNotes(open) {
  ui.notesPop.hidden = !open;
  ui.notesToggle.setAttribute('aria-expanded', String(open));
}

function renderQuad(study) {
  const byKey = new Map(study.images.map((image) => [image.view_key, image]));
  const layoutPages = pages(study);
  state.page = Math.min(Math.max(state.page, 0), layoutPages.length - 1);
  const shown = layoutPages[state.page];

  state.tiles.clear();
  const grid = gridOf(shown.length);
  ui.quad.style.gridTemplateColumns = `repeat(${grid.cols}, minmax(0, 1fr))`;
  ui.quad.style.gridTemplateRows = `repeat(${grid.rows}, minmax(0, 1fr))`;
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
      span.append(document.createTextNode(formatMm(measure.mm)));
      if (measure.estimated) {
        const mark = document.createElement('span');
        mark.className = 'est-mark';
        mark.textContent = t('est_short');
        span.append(document.createTextNode(' '), mark);
      }
    } else {
      span.classList.add('miss');
      span.append(document.createTextNode(reason(measure.reason)));
    }
    span.title = t(route === 'landmark' ? 'method_landmark' : 'method_mask')
      + (measure.ok && measure.estimated ? ` · ${t('est_title')}` : '');
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
  if (nipple && nipple.brightened) {
    badges.push({ text: t('badge_brightened'), tone: 'warn', title: t('badge_brightened_title') });
  }
  if (isEstimated(image)) badges.push({ text: t('badge_estimated'), tone: 'warn', title: t('badge_estimated_title') });
  return badges;
}

function renderRails(study) {
  const sides = new Map(((study.summary || {}).sides || []).map((side) => [side.laterality, side]));
  const mloQuality = (laterality) => {
    const image = study.images.find((entry) => entry.view === 'MLO' && entry.laterality === laterality);
    return image && (image.criteria || []).find((item) => item.id === 'mlo_quality');
  };
  renderRail(ui.railR, 'R', sides.get('R'), mloQuality('R'));
  renderRail(ui.railL, 'L', sides.get('L'), mloQuality('L'));
}

function renderRail(node, laterality, side, mloQuality) {
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

  const children = [
    head,
    buildVerdictCard(t('method_landmark'), 'landmark', side.landmark),
    buildVerdictCard(t('method_mask'), 'mask', side.mask),
    agreement,
  ];
  if (mloQuality) {
    // The MLO judged on its own (MLO-only criterion): good, poor or n/a. It never changes the ΔD verdicts above.
    const line = document.createElement('p');
    line.className = 'mlo-quality';
    line.dataset.grade = mloQuality.grade;
    line.title = criterionDetail(mloQuality);
    const label = document.createElement('span');
    label.textContent = t('mlo_quality_label');
    const value = document.createElement('strong');
    value.textContent = criterionValue(mloQuality);
    line.append(label, value);
    children.push(line);
  }
  node.replaceChildren(...children);
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
  value.textContent = formatNumber(result.delta_mm);
  const unit = document.createElement('span');
  unit.className = 'delta-unit';
  unit.textContent = 'mm';
  const label = document.createElement('span');
  label.className = 'delta-label';
  label.textContent = 'ΔD';
  delta.append(value, unit, label);

  const pairs = document.createElement('dl');
  pairs.className = 'pair';
  const estimated = result.estimated || [];
  for (const [term, description, view] of [
    [t('pnl_mlo'), formatMm(result.pnl_mm), 'MLO'],
    [t('wall_cc'), formatMm(result.cc_mm), 'CC'],
    [t('threshold'), `≤ ${formatMm(result.threshold_mm, null)}`, ''],
  ]) {
    const dt = document.createElement('dt');
    dt.textContent = term;
    const dd = document.createElement('dd');
    dd.textContent = description;
    if (view && estimated.includes(view)) {
      const mark = document.createElement('span');
      mark.className = 'est-mark';
      mark.textContent = t('est_short');
      mark.title = t('est_title');
      dd.append(document.createTextNode(' '), mark);
    }
    pairs.append(dt, dd);
  }

  card.append(delta, pairs);
  if (estimated.length) {
    // The verdict stands, and says what it rests on (the contour fallback).
    const note = document.createElement('p');
    note.className = 'card-estimated';
    note.textContent = t('card_estimated', { views: estimated.join(' + ') });
    card.appendChild(note);
  }
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

  const groups = [];
  for (const viewKey of VIEW_ORDER) {
    const image = study.images.find((entry) => entry.view_key === viewKey);
    if (image && (image.criteria || []).length) groups.push([viewKey, image.criteria]);
  }

  const counts = summary.counts || {};
  // Only the MLO quality is graded now, and it is good or poor: "acceptable" is named only if a study has one.
  ui.criteriaCounts.textContent = summary.graded
    ? t(counts.acceptable ? 'criteria_counts' : 'criteria_counts_gp',
      { g: counts.good || 0, a: counts.acceptable || 0, p: counts.poor || 0 })
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
      row.dataset.criterion = item.id;
      const detail = criterionDetail(item);
      if (detail) row.title = detail;

      const dot = document.createElement('span');
      dot.className = 'grade-dot';

      const name = document.createElement('span');
      name.className = 'criterion-name';
      name.append(document.createTextNode(criterionName(item)));

      const value = document.createElement('span');
      value.className = 'criterion-value';
      value.textContent = criterionValue(item);

      row.append(dot, name, value);
      list.appendChild(row);
    }
    card.appendChild(list);
    return card;
  }));
}

/** A criterion's value as shown: worded values (the MLO quality) are translated, numbers carry their unit. */
function criterionValue(item) {
  if (item.value === null || item.value === undefined) return gradeText(item.grade);
  if (item.id === 'mlo_quality') return t(`mlo_quality_${item.value}`, {}, String(item.value));
  if (typeof item.value !== 'number') return `${item.value}${item.unit || ''}`;
  return formatValue(item.value, item.unit); // with the criterion's own unit, whatever it is
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
      state.reportLang = code; // for this session: the next start follows the interface language again
      saveSessionPref(REPORT_LANG_KEY, code);
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
      if (row.criterion) line.dataset.criterion = row.criterion;
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
  if (!payload.images.length) { // nothing differs from the loaded labels
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
let historyAsked = 0;

async function refreshHistory() {
  historyAsked += 1;
  const asked = historyAsked;
  try {
    const { studies, unopened } = await api.listStudies();
    if (asked !== historyAsked) return; // a newer listing is on its way: this one may already be out of date
    state.history = studies;
    state.historyUnopened = unopened || 0;
    renderHistory(studies);
  } catch (error) {
    showError(error.message);
  }
}

function renderHistory(studies) {
  ui.recentEmpty.hidden = studies.length > 0;
  // Studies a dataset analysed ahead are stored but not listed until somebody opens them.
  ui.recentEmpty.textContent = state.historyUnopened
    ? t('no_recent_opened', { k: state.historyUnopened }) : t('no_recent');
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
  const label = document.createElement('span');
  label.className = 'history-label';
  label.textContent = String(row.id || '').slice(0, 6); // the same id the review footer shows
  body.append(views, label);

  button.append(dot, body);
  // ▲ only when the two methods reach different verdicts; a wide gap under the same verdict and a single method
  // get their own quieter marks (older servers send only "flagged").
  const kind = row.flag !== undefined ? row.flag : (row.flagged ? 'disagree' : null);
  const marks = { disagree: ['▲', 'flag_title'], wide: ['≈', 'flag_wide_title'], partial: ['–', 'flag_partial_title'] };
  if (marks[kind]) {
    const flag = document.createElement('span');
    flag.className = 'history-flag';
    flag.dataset.kind = kind;
    flag.textContent = marks[kind][0];
    flag.title = t(marks[kind][1]);
    button.appendChild(flag);
  }
  const time = document.createElement('span');
  time.className = 'history-time';
  time.textContent = formatDateTime(row.created);
  button.appendChild(time);

  button.addEventListener('click', () => openStored(row.id));
  item.appendChild(button);
  return item;
}

/**
 * Open a stored study the reader picked on its own: a row of the recent list, or an address. With a dataset open
 * its bar must not stay on another study: a study of the dataset is gone to through the dataset (the bar moves to
 * it, and it is marked as opened there), any other study leaves the dataset, as a single upload and "New study" do.
 */
function openStored(studyId) {
  const dataset = state.dataset;
  if (!dataset) return openStudy(studyId);
  if (state.busy) { showError(t('busy_wait')); return Promise.resolve({ ok: false }); } // nothing is left for nothing
  const held = [...dataset.ids].find(([, id]) => id === studyId);
  if (held) return goToStudy(held[0]);
  closeDataset();
  return openStudy(studyId);
}

/**
 * Fetch a stored study and put it on screen.
 * Resolves to { ok: true } once it is shown, or { ok: false, status?, aborted? } with the reason already displayed.
 */
async function openStudy(studyId, { signal } = {}) {
  if (state.busy) { showError(t('busy_wait')); return { ok: false }; }
  clearError();
  state.busy = true;
  ui.progressText.textContent = t('opening');
  show('processing');
  try {
    const study = await api.getStudy(studyId, { signal });
    if (signal && signal.aborted) return { ok: false, aborted: true };
    ui.progressText.textContent = t('loading_views');
    state.bitmaps.clear();
    const shown = await showStudy(study, signal);
    // On screen for review, by whatever route (dataset, recent list, address): the server lists it from now on.
    if (shown) api.markOpened(study.id).catch(() => {});
    return shown ? { ok: true } : { ok: false, aborted: true };
  } catch (error) {
    const aborted = Boolean(signal && signal.aborted);
    if (!aborted) {
      showError(error.message);
      show(state.study ? 'review' : 'landing');
    }
    return { ok: false, status: error.status, aborted };
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
    const deleted = state.study.id;
    await api.deleteStudy(deleted);
    const dataset = state.dataset;
    const entry = dataset && [...dataset.ids].find(([, id]) => id === deleted);
    startNewStudy();
    if (entry) {
      // Keep the dataset: mark the study removed and move on to the next one, if there is one.
      const [index] = entry;
      forgetStudy(dataset, index);
      dataset.removed.add(index);
      updateDatasetOption(index);
      renderDatasetNav();
      if (index === dataset.index && index + 1 < dataset.studies.length) goToStudy(index + 1);
    }
  } catch (error) {
    showError(error.message);
  }
}

boot();

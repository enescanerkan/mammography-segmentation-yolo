// Label editor: correct every contour and landmark the models produced. Stored lesion boxes (from the retired
// detector) are neither shown nor editable here; they are passed back unchanged on save.
//
// Which contours are drawn follows the review screen's layer toggles (breast tissue, pectoral muscle, nipple
// contour): one set of switches for every view of the study. "Hide contours" (H) switches the three off and back on.
// A hidden contour is only not drawn: it is not changed, and it cannot be edited until it is shown again (its handles
// would be invisible). The landmark points are always drawn and stay editable.
//
// Edits are held on a private copy of the study until Save, so leaving the editor
// never silently changes what the review screen showed. On save the server
// recomputes both measurement routes from the corrected shapes — a fixed contour
// has to move the millimetres, or correcting one would be theatre.

import { FRAME } from './overlay.js';
import { appearance, withAlpha } from './appearance.js';
import { t } from './i18n.js';

const MIN_ZOOM = 0.15;
const MAX_ZOOM = 16;
const MAX_CANVAS_SIDE = 6144;
const HIT_PX = 9;            // screen-pixel radius for grabbing a handle

// `name` is an i18n key; nameOf() resolves it in the current language.
export const CLASSES = [
  { id: 'breast-tissue', name: 'cls_breast', kind: 'polygon' },
  { id: 'pectoral', name: 'cls_pectoral', kind: 'polygon' },
  { id: 'nipple', name: 'cls_nipple', kind: 'polygon' },
  { id: 'point:nipple', name: 'cls_nipple_point', kind: 'point', key: 'nipple', style: 'nipple_point' },
  { id: 'point:pectoral_top', name: 'cls_pec_top', kind: 'point', key: 'pectoral_top' },
  { id: 'point:pectoral_bottom', name: 'cls_pec_bottom', kind: 'point', key: 'pectoral_bottom' },
];

const TOOLS = [
  { id: 'select', name: 'k_select', key: 'V', hint: 'hint_select' },
  { id: 'polygon', name: 'k_polygon', key: 'P', hint: 'hint_polygon' },
  { id: 'point', name: 'k_point', key: 'N', hint: 'hint_point' },
];

// The review screen's layer (overlay.js LAYERS) that shows each contour class.
export const LAYER_OF = { 'breast-tissue': 'breast', pectoral: 'pectoral', nipple: 'nipple' };

const EYE_OPEN = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M1 8s2.6-5 7-5 7 5 7 5-2.6 5-7 5-7-5-7-5z" fill="none" stroke="currentColor" stroke-width="1.4"/><circle cx="8" cy="8" r="2.2" fill="currentColor"/></svg>';
const EYE_SHUT = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M1 8s2.6-5 7-5 7 5 7 5-2.6 5-7 5-7-5-7-5z" fill="none" stroke="currentColor" stroke-width="1.4"/><path d="M2.5 13.5l11-11" stroke="currentColor" stroke-width="1.4"/></svg>';

const clamp = (value, low, high) => Math.min(Math.max(value, low), high);
const classById = (id) => CLASSES.find((entry) => entry.id === id);
const styleKey = (entry) => entry.style || entry.key || entry.label || entry.id;
const colorOf = (entry) => appearance.colorOf(styleKey(entry), '#ffffff');
const round = (value) => Math.round(value * 10) / 10;
const nameOf = (entry) => t(entry.name);

/**
 * The labels a save would send for one view, in a fixed key order. The "edited" flags a drag sets are left out, so a
 * vertex moved and moved back, or an edit undone, compares equal to the loaded labels.
 */
export function labelSignature(image) {
  const canonical = (value) => {
    if (Array.isArray(value)) return value.map(canonical);
    if (value && typeof value === 'object') {
      const out = {};
      for (const key of Object.keys(value).sort()) if (key !== 'edited') out[key] = canonical(value[key]);
      return out;
    }
    return value;
  };
  return JSON.stringify(canonical({ polygons: image.polygons || {}, points: image.points || {} }));
}

export class Editor {
  /**
   * `layers` is the review screen's layer set: `has(id)` says whether a layer is shown, `set(ids, on)` switches
   * layers (and keeps the review's toggles in step). `onClean(viewKey, polygons)` resolves to the server's answer,
   * `{ polygons, changed }` (the classes whose contour was replaced or removed, decided on label maps).
   */
  constructor(elements, { onSave, onClose, onClean, layers }) {
    this.el = elements;
    this.onSave = onSave;
    this.onClose = onClose;
    this.onClean = onClean;
    this.layers = layers || { has: () => true, set: () => {} };
    this.hiddenByToggle = null; // the contour layers "Hide contours" switched off, to switch them back on
    this.cleaning = false;

    this.ctx = this.el.canvas.getContext('2d');
    this.images = new Map();      // view_key -> editable label copy
    this.bitmaps = new Map();
    this.order = [];
    this.viewKey = null;
    this.tool = 'select';
    this.activeClass = 'breast-tissue';
    this.zoom = 1;
    this.offset = { x: 0, y: 0 };
    this.draft = null;
    this.drag = null;
    this.pan = null;
    this.undoStack = [];
    this.saved = new Map(); // view key -> labels signature as loaded; "dirty" means a view differs from it
    this.open = false;

    this._buildTools();
    this._bind();
    appearance.onChange(() => { if (this.open) this.render(); });
  }

  // ── lifecycle ────────────────────────────────────────────────────────────
  load(study, bitmaps, viewKey, boxOf) {
    this.images = new Map(study.images.map((image) => [image.view_key, {
      view_key: image.view_key,
      polygons: JSON.parse(JSON.stringify(image.polygons || {})),
      points: JSON.parse(JSON.stringify((image.landmarks || {}).points || {})),
      lesions: JSON.parse(JSON.stringify(image.lesions || [])), // passed back as stored, never edited
    }]));
    this.studyId = study.id;
    this.bitmaps = bitmaps;
    this.boxOf = boxOf || (() => [0, 0, FRAME, FRAME]);
    this.order = study.images.map((image) => image.view_key);
    this.undoStack = [];
    this.saved = new Map([...this.images].map(([key, image]) => [key, labelSignature(image)]));
    this.draft = null;
    this.viewKey = this.images.has(viewKey) ? viewKey : this.order[0];

    this.el.root.hidden = false;
    this.open = true;
    this._renderRail();
    this.fit();
    this.el.save.focus();
  }

  close() {
    this.open = false;
    this.el.root.hidden = true;
    this.draft = null;
    this.drag = null;
  }

  /** A view whose labels differ from the ones it was loaded with (an edit undone, or dragged back, is no change). */
  changed(image) {
    return Boolean(image) && labelSignature(image) !== this.saved.get(image.view_key);
  }

  /** Unsaved changes: some view differs from what was loaded. */
  get dirty() {
    return [...this.images.values()].some((image) => this.changed(image));
  }

  payload() {
    // Only the views whose labels differ from the loaded ones: the server marks what it receives as edited, and
    // an untouched view should keep its model-made status.
    return {
      images: [...this.images.values()].filter((image) => this.changed(image)).map((image) => ({
        view_key: image.view_key,
        polygons: image.polygons,
        landmarks: { points: image.points },
        lesions: image.lesions,
      })),
    };
  }

  get current() { return this.images.get(this.viewKey); }

  /** The class's contour is drawn (and so can be edited); landmark points always are. */
  shown(classId) {
    return !(classId in LAYER_OF) || this.layers.has(LAYER_OF[classId]);
  }

  /** No contour is drawn: the toggle offers to show them. */
  get contoursHidden() {
    return Object.keys(LAYER_OF).every((classId) => !this.shown(classId));
  }

  /** One class's contour on or off; a single change ends what "Hide contours" would restore. */
  toggleContour(classId) {
    if (!(classId in LAYER_OF)) return;
    this.layers.set([LAYER_OF[classId]], !this.shown(classId));
    this.hiddenByToggle = null;
    if (this.draft && this.draft.classId === classId) this.draft = null;
    this.drag = null;
    if (this.open) { this._renderRail(); this.render(); }
  }

  /** "Hide contours" / "Show contours": switch the three contour layers off, or back to what was on. */
  toggleContours() {
    if (!this.contoursHidden) {
      this.hiddenByToggle = Object.values(LAYER_OF).filter((layer) => this.layers.has(layer));
      this.layers.set(this.hiddenByToggle, false);
    } else {
      const restore = this.hiddenByToggle && this.hiddenByToggle.length ? this.hiddenByToggle : Object.values(LAYER_OF);
      this.hiddenByToggle = null;
      this.layers.set(restore, true);
    }
    this.draft = null;
    this.drag = null;
    if (this.open) { this._renderRail(); this.render(); }
  }

  /**
   * "Smooth / clean labels": the server's rules on this view's contours, kept as an unsaved edit (Undo reverts).
   * The answer goes to the view it was asked for, also when the reader has moved to another view meanwhile; it is
   * not applied over an edit made to that view while it was being computed, and the reader is told so.
   */
  async clean() {
    const image = this.current;
    if (!image || this.cleaning || !this.onClean) return;
    const { viewKey, studyId } = this;
    const sent = JSON.parse(JSON.stringify(image.polygons));
    this.cleaning = true;
    this._renderRail();
    let message = '';
    try {
      const reply = await this.onClean(viewKey, sent);
      const target = this.images.get(viewKey);
      const elsewhere = viewKey !== this.viewKey;
      if (!target || !reply || !this.open || this.studyId !== studyId) {
        // the editor was closed or loaded another study meanwhile: nothing to apply it to
      } else if (JSON.stringify(target.polygons) !== JSON.stringify(sent)) {
        message = t('ed_clean_stale', { view: viewKey });
      } else {
        const changed = (reply.changed || []).filter((name) => name in LAYER_OF);
        if (!changed.length) {
          message = elsewhere ? t('ed_clean_none_view', { view: viewKey }) : t('ed_clean_none');
        } else {
          this._snapshot(viewKey);
          const next = {};
          for (const [name, entry] of Object.entries(reply.polygons || {})) {
            next[name] = changed.includes(name) ? { ...entry, edited: true } : entry;
          }
          target.polygons = next;
          if (!elsewhere) this.draft = null;
          const classes = changed.map((name) => nameOf(classById(name))).join(', ');
          message = elsewhere ? t('ed_clean_done_view', { view: viewKey, classes }) : t('ed_clean_done', { classes });
        }
      }
    } catch (error) {
      message = error.message;
    } finally {
      this.cleaning = false;
      if (this.open) {
        this._renderRail();
        this.render();
        if (message) this.el.toolHint.textContent = message;
      }
    }
  }

  /** Rebuild every translated string (called when the interface language changes). */
  relabel() {
    this._buildTools();
    if (this.open) { this._renderRail(); this.render(); }
  }

  // ── chrome ───────────────────────────────────────────────────────────────
  _buildTools() {
    this.el.tools.replaceChildren(...TOOLS.map((tool) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'tool-btn';
      button.dataset.tool = tool.id;
      button.append(document.createTextNode(t(tool.name)));
      const kbd = document.createElement('kbd');
      kbd.textContent = tool.key;
      button.appendChild(kbd);
      button.addEventListener('click', () => this._setTool(tool.id));
      return button;
    }));
  }

  _renderRail() {
    this.el.viewName.textContent = this.viewKey || '—';
    this.el.dirty.hidden = !this.dirty;

    this.el.views.replaceChildren(...this.order.map((key) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'view-btn' + (key === this.viewKey ? ' on' : '');
      button.append(document.createTextNode(key));
      if (this.changed(this.images.get(key))) {
        const mark = document.createElement('span');
        mark.className = 'edited';
        mark.textContent = ' •';
        mark.title = t('edited_mark');
        button.appendChild(mark);
      }
      button.addEventListener('click', () => this._setView(key));
      return button;
    }));

    const image = this.current;
    this.el.classes.replaceChildren(...CLASSES.map((entry) => this._classItem(entry, image)));

    for (const button of this.el.tools.querySelectorAll('.tool-btn')) {
      button.classList.toggle('on', button.dataset.tool === this.tool);
    }
    const tool = TOOLS.find((entry) => entry.id === this.tool);
    const blocked = this.tool !== 'point' && !this.shown(this.activeClass);
    this.el.toolHint.textContent = blocked ? t('hint_hidden') : (tool ? t(tool.hint) : '');

    if (this.el.hide) {
      const hidden = this.contoursHidden;
      this.el.hide.textContent = t(hidden ? 'ed_show' : 'ed_hide');
      this.el.hide.setAttribute('aria-pressed', String(hidden));
      this.el.hide.classList.toggle('on', hidden);
    }
    if (this.el.clean) {
      this.el.clean.disabled = this.cleaning;
      this.el.clean.textContent = t(this.cleaning ? 'ed_cleaning' : 'ed_clean');
    }
  }

  _classItem(entry, image) {
    const item = document.createElement('li');

    const row = document.createElement('button');
    row.type = 'button';
    row.className = 'class-row' + (entry.id === this.activeClass ? ' on' : '');

    const swatch = document.createElement('span');
    swatch.className = 'swatch';
    swatch.style.background = colorOf(entry);

    const name = document.createElement('span');
    name.className = 'class-name';
    name.textContent = nameOf(entry);

    const state = document.createElement('span');
    state.className = 'class-state';
    const status = this._statusOf(image, entry);
    state.textContent = this.shown(entry.id) ? status.text : `${status.text} · ${t('st_hidden')}`;
    state.dataset.has = status.has ? 'yes' : 'no';
    if (!this.shown(entry.id)) row.classList.add('hidden-class');

    row.append(swatch, name, state);
    row.addEventListener('click', () => this._setClass(entry.id));
    item.appendChild(row);

    if (entry.id in LAYER_OF) {
      // One contour on or off (the review layer of that class); "Hide contours" still switches all three.
      const shown = this.shown(entry.id);
      const eye = document.createElement('button');
      eye.type = 'button';
      eye.className = 'class-eye' + (shown ? '' : ' off');
      eye.setAttribute('aria-pressed', String(!shown));
      eye.title = t(shown ? 'ed_hide_one' : 'ed_show_one', { name: nameOf(entry) });
      eye.setAttribute('aria-label', eye.title);
      eye.innerHTML = shown ? EYE_OPEN : EYE_SHUT;
      eye.addEventListener('click', () => this.toggleContour(entry.id));
      item.classList.add('class-item');
      item.appendChild(eye);
    }

    return item;
  }

  _statusOf(image, entry) {
    if (!image) return { text: '—', has: false };
    if (entry.kind === 'polygon') {
      const polygon = image.polygons[entry.id];
      const count = polygon && polygon.points ? polygon.points.length : 0;
      return { text: count ? t('st_pts', { n: count }) : t('st_none'), has: count > 0 };
    }
    if (entry.kind === 'point') {
      return { text: image.points[entry.key] ? t('st_set') : t('st_none'), has: Boolean(image.points[entry.key]) };
    }
    return { text: '—', has: false };
  }

  _setView(key) {
    if (!this.images.has(key) || key === this.viewKey) return;
    this.viewKey = key;
    this.draft = null;
    this._renderRail();
    this.fit();
  }

  _setClass(id) {
    this.activeClass = id;
    const entry = classById(id);
    this.draft = null;
    if (entry && this.tool !== 'select') this.tool = entry.kind;
    this._renderRail();
    this.render();
  }

  _setTool(id) {
    this.tool = id;
    this.draft = null;
    // Keep the class and the tool talking about the same kind of shape.
    const entry = classById(this.activeClass);
    if (id !== 'select' && entry && entry.kind !== id) {
      const match = CLASSES.find((candidate) => candidate.kind === id);
      if (match) this.activeClass = match.id;
    }
    this._renderRail();
    this.render();
  }

  // ── history ──────────────────────────────────────────────────────────────
  _snapshot(viewKey = this.viewKey) {
    const image = this.images.get(viewKey);
    if (!image) return;
    this.undoStack.push({
      view_key: viewKey,
      polygons: JSON.parse(JSON.stringify(image.polygons)),
      points: JSON.parse(JSON.stringify(image.points)),
    });
    if (this.undoStack.length > 80) this.undoStack.shift();
  }

  undo() {
    const previous = this.undoStack.pop();
    if (!previous) return;
    const image = this.images.get(previous.view_key);
    if (!image) return;
    image.polygons = previous.polygons;
    image.points = previous.points;
    if (previous.view_key !== this.viewKey) {
      this.viewKey = previous.view_key;
      this.fit();
    }
    this.draft = null;
    this._renderRail();
    this.render();
  }

  revertView(original) {
    if (!original) return;
    this._snapshot();
    const image = this.current;
    image.polygons = JSON.parse(JSON.stringify(original.polygons || {}));
    image.points = JSON.parse(JSON.stringify((original.landmarks || {}).points || {}));
    this.draft = null;
    this._renderRail();
    this.render();
  }

  // ── geometry helpers ─────────────────────────────────────────────────────
  _toImage(clientX, clientY) {
    const rect = this.el.canvas.getBoundingClientRect();
    return {
      x: clamp((clientX - rect.left) / this.zoom, 0, FRAME),
      y: clamp((clientY - rect.top) / this.zoom, 0, FRAME),
    };
  }

  _hitRadius() { return HIT_PX / this.zoom; }

  _findVertex(point) {
    const image = this.current;
    const radius = this._hitRadius();
    for (const entry of CLASSES) {
      // Only the active class draws its handles, so only it can be grabbed; the
      // nipple contour sits on the skin line, within reach of breast vertices. A hidden contour has none.
      if (entry.kind !== 'polygon' || entry.id !== this.activeClass || !this.shown(entry.id)) continue;
      const polygon = image.polygons[entry.id];
      if (!polygon) continue;
      for (let i = 0; i < polygon.points.length; i += 1) {
        const [x, y] = polygon.points[i];
        if (Math.hypot(x - point.x, y - point.y) <= radius) return { classId: entry.id, index: i };
      }
    }
    return null;
  }

  _findPoint(point) {
    const image = this.current;
    const radius = this._hitRadius() * 1.2;
    for (const entry of CLASSES) {
      if (entry.kind !== 'point') continue;
      const position = image.points[entry.key];
      if (!position) continue;
      if (Math.hypot(position[0] - point.x, position[1] - point.y) <= radius) {
        return { key: entry.key };
      }
    }
    return null;
  }

  _findEdge(point) {
    const image = this.current;
    const radius = this._hitRadius();
    for (const entry of CLASSES) {
      if (entry.kind !== 'polygon' || entry.id !== this.activeClass || !this.shown(entry.id)) continue;
      const polygon = image.polygons[entry.id];
      if (!polygon || polygon.points.length < 2) continue;
      const points = polygon.points;
      for (let i = 0; i < points.length; i += 1) {
        if (distanceToSegment(point, points[i], points[(i + 1) % points.length]) <= radius) {
          return { classId: entry.id, index: i + 1 };
        }
      }
    }
    return null;
  }

  // ── input ────────────────────────────────────────────────────────────────
  _bind() {
    this.el.back.addEventListener('click', () => this.onClose());
    this.el.undo.addEventListener('click', () => this.undo());
    this.el.save.addEventListener('click', () => this.onSave());
    this.el.zoomIn.addEventListener('click', () => this.zoomBy(1.25));
    this.el.zoomOut.addEventListener('click', () => this.zoomBy(1 / 1.25));
    this.el.zoomFit.addEventListener('click', () => this.fit());
    if (this.el.hide) this.el.hide.addEventListener('click', () => this.toggleContours());
    if (this.el.clean) this.el.clean.addEventListener('click', () => this.clean());

    this.el.stage.addEventListener('wheel', (event) => {
      if (!this.open) return;
      event.preventDefault();
      const rect = this.el.stage.getBoundingClientRect();
      this.zoomBy(event.deltaY < 0 ? 1.12 : 1 / 1.12,
                  event.clientX - rect.left, event.clientY - rect.top);
    }, { passive: false });

    this.el.canvas.addEventListener('pointerdown', (event) => this._onDown(event));
    this.el.canvas.addEventListener('pointermove', (event) => this._onMove(event));
    this.el.canvas.addEventListener('pointerup', (event) => this._onUp(event));
    this.el.canvas.addEventListener('pointercancel', (event) => this._onUp(event));
    // The right button pans (below); it never opens the browser's menu over the image.
    this.el.stage.addEventListener('contextmenu', (event) => event.preventDefault());

    this.el.canvas.addEventListener('dblclick', (event) => {
      event.preventDefault();
      if (this.tool === 'polygon' && this.draft) this._commitPolygon();
      else this.fit();                     // a double-click anywhere else re-fits
    });
    this.el.stage.addEventListener('dblclick', (event) => {
      if (event.target === this.el.stage) this.fit();
    });

    // Right- or middle-drag pans wherever the pointer is, so editing (the left button) never fights panning. A right
    // click that does not move pans by nothing and does nothing else.
    this.el.stage.addEventListener('pointerdown', (event) => {
      if (event.button !== 1 && event.button !== 2) return;
      event.preventDefault();
      this.pan = { x: event.clientX, y: event.clientY };
      this.el.stage.classList.add('panning');
      this.el.stage.setPointerCapture(event.pointerId);
    });
    this.el.stage.addEventListener('pointermove', (event) => {
      if (!this.pan) return;
      this.offset.x += event.clientX - this.pan.x;
      this.offset.y += event.clientY - this.pan.y;
      this.pan = { x: event.clientX, y: event.clientY };
      this.render();
    });
    const endPan = () => {
      if (!this.pan) return;
      this.pan = null;
      this.el.stage.classList.remove('panning');
    };
    this.el.stage.addEventListener('pointerup', endPan);
    this.el.stage.addEventListener('pointercancel', endPan);

    document.addEventListener('keydown', (event) => this._onKey(event));
    window.addEventListener('resize', () => { if (this.open) this.render(); });
  }

  _onKey(event) {
    if (!this.open) return;
    if (event.target instanceof HTMLInputElement) return;
    const meta = event.ctrlKey || event.metaKey;

    if (meta && event.key.toLowerCase() === 'z') { event.preventDefault(); this.undo(); return; }
    if (event.key === 'Escape') {
      if (this.draft) { this.draft = null; this.render(); }
      else this.onClose();
      return;
    }
    if (event.key === 'Enter' && this.tool === 'polygon' && this.draft) {
      event.preventDefault();
      this._commitPolygon();
      return;
    }
    if (meta || event.altKey) return;
    if (event.key.toLowerCase() === 'h') { event.preventDefault(); this.toggleContours(); return; }
    const byKey = { v: 'select', p: 'polygon', n: 'point' }[event.key.toLowerCase()];
    if (byKey) this._setTool(byKey);
  }

  _onDown(event) {
    if (!this.open || event.button !== 0) return;
    const point = this._toImage(event.clientX, event.clientY);
    const image = this.current;
    this.el.canvas.setPointerCapture(event.pointerId);

    if (this.tool === 'select') {
      const vertex = this._findVertex(point);
      if (vertex) {
        if (event.altKey) {
          const polygon = image.polygons[vertex.classId];
          if (polygon.points.length > 3) {
            this._snapshot();
            polygon.points.splice(vertex.index, 1);
            polygon.edited = true;
            this._renderRail();
            this.render();
          }
          return;
        }
        this._snapshot();
        this.drag = { type: 'vertex', ...vertex };
        return;
      }

      const landmark = this._findPoint(point);
      if (landmark) {
        this._snapshot();
        if (event.altKey) {
          delete image.points[landmark.key];
          this._renderRail();
          this.render();
          return;
        }
        this.drag = { type: 'point', key: landmark.key };
        return;
      }

      if (event.shiftKey) {
        const edge = this._findEdge(point);
        if (edge) {
          this._snapshot();
          const polygon = image.polygons[edge.classId];
          polygon.points.splice(edge.index, 0, [round(point.x), round(point.y)]);
          polygon.edited = true;
          this._renderRail();
          this.render();
        }
        return;
      }

      return;
    }

    if (this.tool === 'polygon') {
      const entry = classById(this.activeClass);
      if (!entry || entry.kind !== 'polygon' || !this.shown(entry.id)) return; // a hidden contour is not redrawn
      if (!this.draft) this.draft = { classId: entry.id, points: [] };
      const first = this.draft.points[0];
      if (first && this.draft.points.length >= 3
          && Math.hypot(first[0] - point.x, first[1] - point.y) <= this._hitRadius() * 1.6) {
        this._commitPolygon();
        return;
      }
      this.draft.points.push([round(point.x), round(point.y)]);
      this.render();
      return;
    }

    if (this.tool === 'point') {
      const entry = classById(this.activeClass);
      if (!entry || entry.kind !== 'point') return;
      this._snapshot();
      if (event.altKey) delete image.points[entry.key];
      else image.points[entry.key] = [round(point.x), round(point.y)];
      this._renderRail();
      this.render();
    }
  }

  _onMove(event) {
    const point = this._toImage(event.clientX, event.clientY);

    if (!this.drag) {
      if (this.tool === 'polygon' && this.draft) {
        this.draft.hover = point;
        this.render();
      }
      return;
    }

    const image = this.current;
    if (this.drag.type === 'vertex') {
      image.polygons[this.drag.classId].points[this.drag.index] = [round(point.x), round(point.y)];
      image.polygons[this.drag.classId].edited = true;
    } else if (this.drag.type === 'point') {
      image.points[this.drag.key] = [round(point.x), round(point.y)];
    }
    this.render();
  }

  _onUp(event) {
    if (this.el.canvas.hasPointerCapture(event.pointerId)) {
      this.el.canvas.releasePointerCapture(event.pointerId);
    }
    if (!this.drag) return;

    this.drag = null;
    this._renderRail(); // the unsaved-changes mark and the per-view dots follow the labels, not the gesture
    this.render();
  }

  _commitPolygon() {
    if (!this.draft || this.draft.points.length < 3) { this.draft = null; this.render(); return; }
    this._snapshot();
    this.current.polygons[this.draft.classId] = {
      points: this.draft.points.map(([x, y]) => [round(x), round(y)]),
      confidence: 1,
      rescued: false,
      edited: true,
    };
    this.draft = null;
    this._renderRail();
    this.render();
  }

  // ── view transform ───────────────────────────────────────────────────────
  fit() {
    // Fit the tissue, not the padded square: the canvas still holds the whole
    // frame so a contour can be drawn anywhere, but the reviewer starts zoomed
    // in on the part that has anything on it.
    const rect = this.el.stage.getBoundingClientRect();
    const [x1, y1, x2, y2] = this.boxOf ? this.boxOf(this.viewKey) : [0, 0, FRAME, FRAME];
    const scale = Math.min((rect.width - 40) / (x2 - x1), (rect.height - 40) / (y2 - y1));
    this.zoom = clamp(scale, MIN_ZOOM, MAX_ZOOM);
    this.offset = {
      x: ((FRAME - x1 - x2) / 2) * this.zoom,
      y: ((FRAME - y1 - y2) / 2) * this.zoom,
    };
    this.render();
  }

  zoomBy(factor, anchorX, anchorY) {
    const rect = this.el.stage.getBoundingClientRect();
    const cx = anchorX === undefined ? rect.width / 2 : anchorX;
    const cy = anchorY === undefined ? rect.height / 2 : anchorY;
    const next = clamp(this.zoom * factor, MIN_ZOOM, MAX_ZOOM);
    if (next === this.zoom) return;

    const ratio = next / this.zoom;
    this.offset.x += (rect.width / 2 + this.offset.x - cx) * (ratio - 1);
    this.offset.y += (rect.height / 2 + this.offset.y - cy) * (ratio - 1);
    this.zoom = next;
    this.render();
  }

  // ── drawing ──────────────────────────────────────────────────────────────
  render() {
    const image = this.current;
    if (!image) return;
    const bitmap = this.bitmaps.get(this.viewKey);

    const dpr = window.devicePixelRatio || 1;
    const side = FRAME * this.zoom;
    // A full-resolution buffer at high zoom is a 100M-pixel redraw on every pointer
    // move, and past Chromium's canvas area limit it draws nothing at all. By then
    // one image pixel spans several screen pixels, so the cap costs no detail.
    const density = Math.min(dpr, MAX_CANVAS_SIDE / side);
    const canvas = this.el.canvas;
    canvas.width = Math.round(side * density);
    canvas.height = Math.round(side * density);
    canvas.style.width = `${side}px`;
    canvas.style.height = `${side}px`;
    canvas.style.transform = `translate(${this.offset.x}px, ${this.offset.y}px)`;
    canvas.style.cursor = this.tool === 'select' ? 'default' : 'crosshair';

    const scale = this.zoom * density;
    const unit = 1 / (this.zoom * dpr);     // line widths stay in screen pixels
    const ctx = this.ctx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.setTransform(scale, 0, 0, scale, 0, 0);
    ctx.imageSmoothingEnabled = this.zoom < 3;
    if (bitmap) {
      ctx.filter = appearance.imageFilter();
      ctx.drawImage(bitmap, 0, 0, FRAME, FRAME);
      ctx.filter = 'none';
    }
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    this._drawPolygons(ctx, image, unit);
    this._drawPoints(ctx, image, unit);
    this._drawDraft(ctx, unit);

    this.el.zoomLevel.textContent = `${Math.round(this.zoom * 100)}%`;
  }

  _drawPolygons(ctx, image, unit) {
    for (const entry of CLASSES) {
      if (entry.kind !== 'polygon' || !this.shown(entry.id)) continue;
      const polygon = image.polygons[entry.id];
      if (!polygon || polygon.points.length < 3) continue;
      const active = entry.id === this.activeClass;
      const color = colorOf(entry);

      ctx.beginPath();
      polygon.points.forEach(([x, y], index) => (index ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      ctx.closePath();
      ctx.fillStyle = withAlpha(color, appearance.fillAlpha * (active ? 1 : 0.45));
      ctx.fill();
      ctx.strokeStyle = active ? color : withAlpha(color, 0.6);
      ctx.lineWidth = (active ? appearance.lineWidth + 0.2 : appearance.lineWidth - 0.4) * unit;
      ctx.stroke();

      if (active) {
        for (const [x, y] of polygon.points) {
          ctx.beginPath();
          ctx.rect(x - 4 * unit, y - 4 * unit, 8 * unit, 8 * unit);
          ctx.fillStyle = '#0b0e12';
          ctx.fill();
          ctx.strokeStyle = color;
          ctx.lineWidth = 1.6 * unit;
          ctx.stroke();
        }
      }
    }
  }

  _drawPoints(ctx, image, unit) {
    for (const entry of CLASSES) {
      if (entry.kind !== 'point') continue;
      const position = image.points[entry.key];
      if (!position) continue;
      const [x, y] = position;
      const color = colorOf(entry);
      const radius = 5 * unit;

      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = 'rgba(0,0,0,.75)';
      ctx.lineWidth = 1.5 * unit;
      ctx.stroke();

      if (entry.id === this.activeClass) {
        ctx.beginPath();
        ctx.arc(x, y, radius * 2.4, 0, Math.PI * 2);
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.2 * unit;
        ctx.setLineDash([3 * unit, 3 * unit]);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }
  }

  _drawDraft(ctx, unit) {
    if (!this.draft || !this.draft.points.length) return;
    const entry = classById(this.draft.classId);
    const color = entry ? colorOf(entry) : '#ffffff';
    ctx.beginPath();
    this.draft.points.forEach(([x, y], index) => (index ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    if (this.draft.hover) ctx.lineTo(this.draft.hover.x, this.draft.hover.y);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2 * unit;
    ctx.setLineDash([6 * unit, 4 * unit]);
    ctx.stroke();
    ctx.setLineDash([]);
    for (const [x, y] of this.draft.points) {
      ctx.beginPath();
      ctx.arc(x, y, 3.2 * unit, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    }
  }

  _label(ctx, text, x, y, color, unit) {
    const size = 12 * unit;
    ctx.font = `500 ${size}px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace`;
    ctx.textBaseline = 'bottom';
    const width = ctx.measureText(text).width;
    ctx.fillStyle = 'rgba(8, 10, 13, .78)';
    ctx.fillRect(x - 4 * unit, y - size - 3 * unit, width + 8 * unit, size + 6 * unit);
    ctx.fillStyle = color;
    ctx.fillText(text, x, y);
  }
}

function distanceToSegment(point, a, b) {
  const abx = b[0] - a[0];
  const aby = b[1] - a[1];
  const lengthSquared = abx * abx + aby * aby;
  if (lengthSquared === 0) return Math.hypot(point.x - a[0], point.y - a[1]);
  const t = clamp(((point.x - a[0]) * abx + (point.y - a[1]) * aby) / lengthSquared, 0, 1);
  return Math.hypot(point.x - (a[0] + t * abx), point.y - (a[1] + t * aby));
}

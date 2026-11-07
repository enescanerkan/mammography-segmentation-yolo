// Label editor: correct every contour, landmark and box the models produced.
//
// Edits are held on a private copy of the study until Save, so leaving the editor
// never silently changes what the review screen showed. On save the server
// recomputes both measurement routes from the corrected shapes — a fixed contour
// has to move the millimetres, or correcting one would be theatre.

import { FRAME, lesionName } from './overlay.js';
import { appearance, lighten, withAlpha } from './appearance.js';
import { t } from './i18n.js';

const MIN_ZOOM = 0.15;
const MAX_ZOOM = 16;
const MAX_CANVAS_SIDE = 6144;
const HIT_PX = 9;            // screen-pixel radius for grabbing a handle
const PASTE_OFFSET = 14;     // image px, so a pasted copy is visibly its own box

// `name` is an i18n key; nameOf() resolves it in the current language.
export const CLASSES = [
  { id: 'breast-tissue', name: 'cls_breast', kind: 'polygon' },
  { id: 'pectoral', name: 'cls_pectoral', kind: 'polygon' },
  { id: 'nipple', name: 'cls_nipple', kind: 'polygon' },
  { id: 'point:nipple', name: 'cls_nipple_point', kind: 'point', key: 'nipple', style: 'nipple_point' },
  { id: 'point:pectoral_top', name: 'cls_pec_top', kind: 'point', key: 'pectoral_top' },
  { id: 'point:pectoral_bottom', name: 'cls_pec_bottom', kind: 'point', key: 'pectoral_bottom' },
  { id: 'box:Mass', name: 'cls_mass', kind: 'box', label: 'Mass' },
  { id: 'box:Calcification', name: 'cls_calc', kind: 'box', label: 'Calcification' },
];

const TOOLS = [
  { id: 'select', name: 'k_select', key: 'V', hint: 'hint_select' },
  { id: 'polygon', name: 'k_polygon', key: 'P', hint: 'hint_polygon' },
  { id: 'box', name: 'k_box', key: 'B', hint: 'hint_box' },
  { id: 'point', name: 'k_point', key: 'N', hint: 'hint_point' },
];

const clamp = (value, low, high) => Math.min(Math.max(value, low), high);
const classById = (id) => CLASSES.find((entry) => entry.id === id);
const styleKey = (entry) => entry.style || entry.key || entry.label || entry.id;
const colorOf = (entry) => appearance.colorOf(styleKey(entry), '#ffffff');
const round = (value) => Math.round(value * 10) / 10;
const nameOf = (entry) => t(entry.name);

let idCounter = 0;
const nextId = () => `x${Date.now().toString(36)}${(idCounter += 1).toString(36)}`;

export class Editor {
  constructor(elements, { onSave, onClose }) {
    this.el = elements;
    this.onSave = onSave;
    this.onClose = onClose;

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
    this.selected = null;         // { type: 'lesion', id }
    this.hovered = null;
    this.clipboard = null;
    this.undoStack = [];
    this.dirty = false;
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
      lesions: (image.lesions || []).map((lesion) => ({
        ...JSON.parse(JSON.stringify(lesion)),
        id: lesion.id || nextId(),
      })),
    }]));
    this.studyId = study.id;
    this.bitmaps = bitmaps;
    this.boxOf = boxOf || (() => [0, 0, FRAME, FRAME]);
    this.order = study.images.map((image) => image.view_key);
    this.undoStack = [];
    this.dirty = false;
    this.draft = null;
    this.selected = null;
    this.hovered = null;
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

  payload() {
    // Only the views the reviewer changed: the server marks whatever it receives as
    // edited, and an untouched view should keep its model-made status.
    return {
      images: [...this.images.values()].filter((image) => image.__touched).map((image) => ({
        view_key: image.view_key,
        polygons: image.polygons,
        landmarks: { points: image.points },
        lesions: image.lesions,
      })),
    };
  }

  get current() { return this.images.get(this.viewKey); }

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
      if (this.images.get(key) && this.images.get(key).__touched) {
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
    this.el.toolHint.textContent = tool ? t(tool.hint) : '';
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
    state.textContent = status.text;
    state.dataset.has = status.has ? 'yes' : 'no';

    row.append(swatch, name, state);
    row.addEventListener('click', () => this._setClass(entry.id));
    item.appendChild(row);

    // Boxes are the one class that can hold several instances, so they get a
    // numbered list: without it "3" in the status column is unhelpful when the
    // reviewer wants to remove exactly one of them.
    if (entry.kind === 'box' && image) {
      const instances = image.lesions
        .map((lesion, index) => ({ lesion, index }))
        .filter(({ lesion }) => lesion.label === entry.label);
      if (instances.length) {
        const list = document.createElement('ul');
        list.className = 'instance-list';
        instances.forEach(({ lesion }, ordinal) => {
          list.appendChild(this._instanceRow(entry, lesion, ordinal + 1));
        });
        item.appendChild(list);
      }
    }
    return item;
  }

  _instanceRow(entry, lesion, ordinal) {
    const row = document.createElement('li');
    row.className = 'instance-row'
      + (this.selected && this.selected.id === lesion.id ? ' on' : '')
      + (this.hovered && this.hovered.id === lesion.id ? ' hover' : '');

    const pick = document.createElement('button');
    pick.type = 'button';
    pick.className = 'instance-pick';
    const dot = document.createElement('span');
    dot.className = 'instance-dot';
    dot.style.background = colorOf(entry);
    const label = document.createElement('span');
    label.textContent = `${nameOf(entry)} ${ordinal}`;
    const size = document.createElement('span');
    size.className = 'instance-size';
    const [x1, y1, x2, y2] = lesion.box;
    size.textContent = `${Math.round(x2 - x1)}×${Math.round(y2 - y1)}`;
    pick.append(dot, label, size);
    pick.addEventListener('click', () => {
      this.selected = { type: 'lesion', id: lesion.id };
      this._renderRail();
      this.render();
    });
    pick.addEventListener('pointerenter', () => this._setHover({ type: 'lesion', id: lesion.id }));
    pick.addEventListener('pointerleave', () => this._setHover(null));

    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'instance-remove';
    remove.title = t('delete_box');
    remove.setAttribute('aria-label', t('delete_named', { name: `${nameOf(entry)} ${ordinal}` }));
    remove.textContent = '✕';
    remove.addEventListener('click', (event) => {
      event.stopPropagation();
      this._removeLesion(lesion.id);
    });

    row.append(pick, remove);
    return row;
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
    const count = image.lesions.filter((lesion) => lesion.label === entry.label).length;
    return { text: count ? String(count) : t('st_none'), has: count > 0 };
  }

  _setHover(target) {
    const before = this.hovered && this.hovered.id;
    const after = target && target.id;
    if (before === after) return;
    this.hovered = target;
    this._renderRail();
    this.render();
  }

  _setView(key) {
    if (!this.images.has(key) || key === this.viewKey) return;
    this.viewKey = key;
    this.draft = null;
    this.selected = null;
    this.hovered = null;
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
  _snapshot() {
    const image = this.current;
    if (!image) return;
    this.undoStack.push({
      view_key: this.viewKey,
      polygons: JSON.parse(JSON.stringify(image.polygons)),
      points: JSON.parse(JSON.stringify(image.points)),
      lesions: JSON.parse(JSON.stringify(image.lesions)),
    });
    if (this.undoStack.length > 80) this.undoStack.shift();
    image.__touched = true;
    this.dirty = true;
    this.el.dirty.hidden = false;
  }

  undo() {
    const previous = this.undoStack.pop();
    if (!previous) return;
    const image = this.images.get(previous.view_key);
    if (!image) return;
    image.polygons = previous.polygons;
    image.points = previous.points;
    image.lesions = previous.lesions;
    if (previous.view_key !== this.viewKey) {
      this.viewKey = previous.view_key;
      this.fit();
    }
    this.draft = null;
    this.selected = null;
    this._renderRail();
    this.render();
  }

  revertView(original) {
    if (!original) return;
    this._snapshot();
    const image = this.current;
    image.polygons = JSON.parse(JSON.stringify(original.polygons || {}));
    image.points = JSON.parse(JSON.stringify((original.landmarks || {}).points || {}));
    image.lesions = (original.lesions || []).map((lesion) => ({
      ...JSON.parse(JSON.stringify(lesion)), id: lesion.id || nextId(),
    }));
    this.draft = null;
    this.selected = null;
    this._renderRail();
    this.render();
  }

  // ── lesion helpers ───────────────────────────────────────────────────────
  _lesionById(id) { return this.current.lesions.find((lesion) => lesion.id === id); }

  _removeLesion(id) {
    const image = this.current;
    const index = image.lesions.findIndex((lesion) => lesion.id === id);
    if (index < 0) return;
    this._snapshot();
    image.lesions.splice(index, 1);
    if (this.selected && this.selected.id === id) this.selected = null;
    if (this.hovered && this.hovered.id === id) this.hovered = null;
    this._renderRail();
    this.render();
  }

  _copySelection() {
    if (!this.selected) return;
    const lesion = this._lesionById(this.selected.id);
    if (lesion) this.clipboard = JSON.parse(JSON.stringify(lesion));
  }

  _paste() {
    if (!this.clipboard) return;
    this._snapshot();
    const [x1, y1, x2, y2] = this.clipboard.box;
    const shift = (value) => clamp(value + PASTE_OFFSET, 0, FRAME);
    const copy = {
      ...this.clipboard,
      id: nextId(),
      edited: true,
      box: [shift(x1), shift(y1), shift(x2), shift(y2)],
    };
    this.current.lesions.push(copy);
    this.selected = { type: 'lesion', id: copy.id };
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
      // nipple contour sits on the skin line, within reach of breast vertices.
      if (entry.kind !== 'polygon' || entry.id !== this.activeClass) continue;
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
      if (entry.kind !== 'polygon' || entry.id !== this.activeClass) continue;
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

  _findLesion(point) {
    const image = this.current;
    for (let i = image.lesions.length - 1; i >= 0; i -= 1) {
      const [x1, y1, x2, y2] = image.lesions[i].box;
      if (point.x >= x1 && point.x <= x2 && point.y >= y1 && point.y <= y2) return image.lesions[i];
    }
    return null;
  }

  _findLesionCorner(point) {
    const radius = this._hitRadius();
    for (let i = this.current.lesions.length - 1; i >= 0; i -= 1) {
      const lesion = this.current.lesions[i];
      const [x1, y1, x2, y2] = lesion.box;
      const corners = [[x1, y1, 'nw'], [x2, y1, 'ne'], [x1, y2, 'sw'], [x2, y2, 'se']];
      for (const [cx, cy, corner] of corners) {
        if (Math.hypot(cx - point.x, cy - point.y) <= radius) return { lesion, corner };
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
    this.el.canvas.addEventListener('pointerleave', () => this._setHover(null));
    this.el.canvas.addEventListener('contextmenu', (event) => event.preventDefault());

    this.el.canvas.addEventListener('dblclick', (event) => {
      event.preventDefault();
      if (this.tool === 'polygon' && this.draft) this._commitPolygon();
      else this.fit();                     // a double-click anywhere else re-fits
    });
    this.el.stage.addEventListener('dblclick', (event) => {
      if (event.target === this.el.stage) this.fit();
    });

    // Middle-drag pans wherever the pointer is, so editing never fights panning.
    this.el.stage.addEventListener('pointerdown', (event) => {
      if (event.button !== 1) return;
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
    if (meta && event.key.toLowerCase() === 'c') { event.preventDefault(); this._copySelection(); return; }
    if (meta && event.key.toLowerCase() === 'v') { event.preventDefault(); this._paste(); return; }
    if (meta && event.key.toLowerCase() === 'd') {
      event.preventDefault();
      this._copySelection();
      this._paste();
      return;
    }
    if (event.key === 'Delete' || event.key === 'Backspace') {
      if (this.selected) { event.preventDefault(); this._removeLesion(this.selected.id); }
      return;
    }
    if (event.key === 'Escape') {
      if (this.draft) { this.draft = null; this.render(); }
      else if (this.selected) { this.selected = null; this._renderRail(); this.render(); }
      else this.onClose();
      return;
    }
    if (event.key === 'Enter' && this.tool === 'polygon' && this.draft) {
      event.preventDefault();
      this._commitPolygon();
      return;
    }
    const byKey = { v: 'select', p: 'polygon', b: 'box', n: 'point' }[event.key.toLowerCase()];
    if (byKey) this._setTool(byKey);
  }

  _onDown(event) {
    if (!this.open || event.button !== 0) return;
    const point = this._toImage(event.clientX, event.clientY);
    const image = this.current;
    this.el.canvas.setPointerCapture(event.pointerId);

    if (this.tool === 'select') {
      const corner = this._findLesionCorner(point);
      if (corner) {
        this._snapshot();
        this.selected = { type: 'lesion', id: corner.lesion.id };
        // The opposite corner stays where it was at grab time, even if the dragged
        // one crosses over it and back.
        const [x1, y1, x2, y2] = corner.lesion.box;
        const anchor = [corner.corner.includes('w') ? x2 : x1, corner.corner.includes('n') ? y2 : y1];
        this.drag = { type: 'lesion-resize', id: corner.lesion.id, corner, anchor };
        this._renderRail();
        return;
      }

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

      const lesion = this._findLesion(point);
      if (lesion) {
        if (event.altKey) { this._removeLesion(lesion.id); return; }
        this._snapshot();
        this.selected = { type: 'lesion', id: lesion.id };
        this.drag = { type: 'lesion-move', id: lesion.id, grab: point, box: [...lesion.box] };
        this._renderRail();
        this.render();
        return;
      }

      if (this.selected) { this.selected = null; this._renderRail(); this.render(); }
      return;
    }

    if (this.tool === 'polygon') {
      const entry = classById(this.activeClass);
      if (!entry || entry.kind !== 'polygon') return;
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

    if (this.tool === 'box') {
      const entry = classById(this.activeClass);
      if (!entry || entry.kind !== 'box') return;
      if (event.altKey) {
        const lesion = this._findLesion(point);
        if (lesion) this._removeLesion(lesion.id);
        return;
      }
      this.drag = { type: 'box-draw', label: entry.label, start: point, end: point };
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
        return;
      }
      const lesion = this._findLesion(point) || (this._findLesionCorner(point) || {}).lesion;
      this._setHover(lesion ? { type: 'lesion', id: lesion.id } : null);
      return;
    }

    const image = this.current;
    if (this.drag.type === 'vertex') {
      image.polygons[this.drag.classId].points[this.drag.index] = [round(point.x), round(point.y)];
      image.polygons[this.drag.classId].edited = true;
    } else if (this.drag.type === 'point') {
      image.points[this.drag.key] = [round(point.x), round(point.y)];
    } else if (this.drag.type === 'box-draw') {
      this.drag.end = point;
    } else if (this.drag.type === 'lesion-move') {
      const lesion = this._lesionById(this.drag.id);
      if (lesion) {
        const [x1, y1, x2, y2] = this.drag.box;
        let dx = point.x - this.drag.grab.x;
        let dy = point.y - this.drag.grab.y;
        dx = clamp(dx, -x1, FRAME - x2);
        dy = clamp(dy, -y1, FRAME - y2);
        lesion.box = [x1 + dx, y1 + dy, x2 + dx, y2 + dy].map(round);
        lesion.edited = true;
      }
    } else if (this.drag.type === 'lesion-resize') {
      const lesion = this._lesionById(this.drag.id);
      if (lesion) {
        const [ax, ay] = this.drag.anchor;
        const next = [ax, ay, point.x, point.y];
        lesion.box = [Math.min(next[0], next[2]), Math.min(next[1], next[3]),
                      Math.max(next[0], next[2]), Math.max(next[1], next[3])].map(round);
        lesion.edited = true;
      }
    }
    this.render();
  }

  _onUp(event) {
    if (this.el.canvas.hasPointerCapture(event.pointerId)) {
      this.el.canvas.releasePointerCapture(event.pointerId);
    }
    if (!this.drag) return;

    if (this.drag.type === 'box-draw') {
      const { start, end, label } = this.drag;
      const box = [Math.min(start.x, end.x), Math.min(start.y, end.y),
                   Math.max(start.x, end.x), Math.max(start.y, end.y)];
      if (box[2] - box[0] >= 4 && box[3] - box[1] >= 4) {
        this._snapshot();
        const lesion = { id: nextId(), label, box: box.map(round), confidence: 1, edited: true };
        this.current.lesions.push(lesion);
        this.selected = { type: 'lesion', id: lesion.id };
        this._renderRail();
      }
    } else if (this.drag.type === 'lesion-resize' || this.drag.type === 'lesion-move') {
      this._renderRail();
    }
    this.drag = null;
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
    this._drawLesions(ctx, image, unit);
    this._drawDraft(ctx, unit);

    this.el.zoomLevel.textContent = `${Math.round(this.zoom * 100)}%`;
  }

  _drawPolygons(ctx, image, unit) {
    for (const entry of CLASSES) {
      if (entry.kind !== 'polygon') continue;
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

  _drawLesions(ctx, image, unit) {
    const counters = new Map();
    for (const lesion of image.lesions) {
      const ordinal = (counters.get(lesion.label) || 0) + 1;
      counters.set(lesion.label, ordinal);

      const entry = CLASSES.find((candidate) => candidate.label === lesion.label);
      const base = entry ? colorOf(entry) : '#ff5fb0';
      const isSelected = this.selected && this.selected.id === lesion.id;
      const isHovered = this.hovered && this.hovered.id === lesion.id;
      const color = isHovered || isSelected ? lighten(base, isSelected ? 0.45 : 0.3) : base;

      const [x1, y1, x2, y2] = lesion.box;
      if (isHovered || isSelected) {
        ctx.fillStyle = withAlpha(base, 0.14);
        ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
      }
      ctx.beginPath();
      ctx.rect(x1, y1, x2 - x1, y2 - y1);
      ctx.strokeStyle = color;
      ctx.lineWidth = (isSelected ? 2.6 : isHovered ? 2.2 : 1.8) * unit;
      ctx.setLineDash(isSelected ? [] : [5 * unit, 4 * unit]);
      ctx.stroke();
      ctx.setLineDash([]);

      if (isSelected) {
        for (const [cx, cy] of [[x1, y1], [x2, y1], [x1, y2], [x2, y2]]) {
          ctx.beginPath();
          ctx.rect(cx - 4 * unit, cy - 4 * unit, 8 * unit, 8 * unit);
          ctx.fillStyle = '#0b0e12';
          ctx.fill();
          ctx.strokeStyle = color;
          ctx.lineWidth = 1.6 * unit;
          ctx.stroke();
        }
      }
      this._label(ctx, `${lesionName(lesion.label)} ${ordinal}`, x1, y1 - 5 * unit, color, unit);
    }
  }

  _drawDraft(ctx, unit) {
    if (this.drag && this.drag.type === 'box-draw') {
      const { start, end } = this.drag;
      ctx.beginPath();
      ctx.rect(Math.min(start.x, end.x), Math.min(start.y, end.y),
               Math.abs(end.x - start.x), Math.abs(end.y - start.y));
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.6 * unit;
      ctx.setLineDash([5 * unit, 4 * unit]);
      ctx.stroke();
      ctx.setLineDash([]);
    }

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

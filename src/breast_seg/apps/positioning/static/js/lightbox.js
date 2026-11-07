// Full-screen viewer: one image, zoomable and pannable, with the same overlay
// renderer the grid tiles use.

import { drawOverlay, FRAME } from './overlay.js';
import { appearance } from './appearance.js';

const MIN_ZOOM = 0.2;
const MAX_ZOOM = 12;
const MAX_CANVAS_SIDE = 6144;

export class Lightbox {
  constructor(elements, getVisibleLayers) {
    this.el = elements;
    this.getVisibleLayers = getVisibleLayers;

    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d');
    this.el.stage.appendChild(this.canvas);

    this.image = null;
    this.bitmap = null;
    this.box = [0, 0, FRAME, FRAME];
    this.zoom = 1;
    this.offset = { x: 0, y: 0 };
    this.drag = null;
    this.open = false;

    this._bind();
    appearance.onChange(() => { if (this.open) this.render(); });
  }

  _bind() {
    this.el.close.addEventListener('click', () => this.hide());
    this.el.zoomIn.addEventListener('click', () => this.zoomBy(1.25));
    this.el.zoomOut.addEventListener('click', () => this.zoomBy(1 / 1.25));
    this.el.zoomReset.addEventListener('click', () => this.fit());

    this.el.root.addEventListener('click', (event) => {
      if (event.target === this.el.root) this.hide();
    });

    document.addEventListener('keydown', (event) => {
      if (!this.open) return;
      if (event.key === 'Escape') this.hide();
      else if (event.key === '+' || event.key === '=') this.zoomBy(1.25);
      else if (event.key === '-') this.zoomBy(1 / 1.25);
      else if (event.key === '0') this.fit();
    });

    // A double-click anywhere on the stage puts zoom and pan back to a clean fit.
    this.el.stage.addEventListener('dblclick', (event) => {
      event.preventDefault();
      this.fit();
    });

    this.el.stage.addEventListener('wheel', (event) => {
      if (!this.open) return;
      event.preventDefault();
      const rect = this.el.stage.getBoundingClientRect();
      this.zoomBy(event.deltaY < 0 ? 1.12 : 1 / 1.12,
                  event.clientX - rect.left, event.clientY - rect.top);
    }, { passive: false });

    this.el.stage.addEventListener('contextmenu', (event) => event.preventDefault());  // right-drag pans
    this.el.stage.addEventListener('pointerdown', (event) => {
      if (!this.open) return;
      this.drag = { x: event.clientX, y: event.clientY };
      this.el.stage.classList.add('panning');
      this.el.stage.setPointerCapture(event.pointerId);
    });
    this.el.stage.addEventListener('pointermove', (event) => {
      if (!this.drag) return;
      this.offset.x += event.clientX - this.drag.x;
      this.offset.y += event.clientY - this.drag.y;
      this.drag = { x: event.clientX, y: event.clientY };
      this.render();
    });
    const endDrag = (event) => {
      if (!this.drag) return;
      this.drag = null;
      this.el.stage.classList.remove('panning');
      if (event.pointerId !== undefined && this.el.stage.hasPointerCapture(event.pointerId)) {
        this.el.stage.releasePointerCapture(event.pointerId);
      }
    };
    this.el.stage.addEventListener('pointerup', endDrag);
    this.el.stage.addEventListener('pointercancel', endDrag);

    this._onResize = () => { if (this.open) this.render(); };
    window.addEventListener('resize', this._onResize);
  }

  show(image, bitmap, box) {
    this.image = image;
    this.bitmap = bitmap;
    this.box = box || [0, 0, FRAME, FRAME];
    this.el.title.textContent = image.view_key;
    this.el.root.hidden = false;
    this.open = true;
    this.fit();
    this.el.close.focus();
  }

  hide() {
    this.open = false;
    this.el.root.hidden = true;
    this.image = null;
    this.bitmap = null;
  }

  fit() {
    const rect = this.el.stage.getBoundingClientRect();
    const [x1, y1, x2, y2] = this.box;
    const scale = Math.min((rect.width - 32) / (x2 - x1), (rect.height - 32) / (y2 - y1));
    this.zoom = Math.max(MIN_ZOOM, scale);
    this.offset = { x: 0, y: 0 };
    this.render();
  }

  zoomBy(factor, anchorX, anchorY) {
    const rect = this.el.stage.getBoundingClientRect();
    const cx = anchorX === undefined ? rect.width / 2 : anchorX;
    const cy = anchorY === undefined ? rect.height / 2 : anchorY;

    const next = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, this.zoom * factor));
    if (next === this.zoom) return;

    // Keep the point under the cursor fixed while the scale changes.
    const centreX = rect.width / 2 + this.offset.x;
    const centreY = rect.height / 2 + this.offset.y;
    const ratio = next / this.zoom;
    this.offset.x += (centreX - cx) * (ratio - 1);
    this.offset.y += (centreY - cy) * (ratio - 1);
    this.zoom = next;
    this.render();
  }

  render() {
    if (!this.image || !this.bitmap) return;

    const dpr = window.devicePixelRatio || 1;
    const [x1, y1, x2, y2] = this.box;
    const cssWidth = (x2 - x1) * this.zoom;
    const cssHeight = (y2 - y1) * this.zoom;

    // Same cap as the editor: past the canvas area limit the buffer draws nothing.
    const density = Math.min(dpr, MAX_CANVAS_SIDE / Math.max(cssWidth, cssHeight));
    this.canvas.width = Math.round(cssWidth * density);
    this.canvas.height = Math.round(cssHeight * density);
    this.canvas.style.width = `${cssWidth}px`;
    this.canvas.style.height = `${cssHeight}px`;
    // The stage centres its child; a transform is enough to pan without relayout.
    this.canvas.style.transform = `translate(${this.offset.x}px, ${this.offset.y}px)`;

    const scale = this.zoom * density;
    const ctx = this.ctx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.setTransform(scale, 0, 0, scale, -x1 * scale, -y1 * scale);
    ctx.imageSmoothingEnabled = this.zoom < 3;
    ctx.filter = appearance.imageFilter();
    ctx.drawImage(this.bitmap, 0, 0, FRAME, FRAME);
    ctx.filter = 'none';
    drawOverlay(ctx, this.image, this.getVisibleLayers(), this.zoom * dpr);  // line widths in screen px

    this.el.zoomLevel.textContent = `${Math.round(this.zoom * 100)}%`;
  }
}

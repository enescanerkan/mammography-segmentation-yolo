// Per-class colour and fill opacity, shared by the review overlay and the editor
// so a colour changed in one place is the colour drawn in the other.
//
// Kept in localStorage because it is a per-reader preference, not study data: a
// radiologist who wants the pectoral fill at 5% should not have to set it again
// on the next study, and nobody else's screen should change because they did.

const STORAGE_KEY = 'mammo.appearance.v1';

const DEFAULTS = {
  colors: {
    'breast-tissue': '#4d9cff',
    pectoral: '#ff9f2e',
    nipple: '#2ee08a',
    nipple_point: '#2ee08a',
    pectoral_top: '#ffd07a',
    pectoral_bottom: '#ffd07a',
    Mass: '#ff5fb0',
    Calcification: '#ff8fd0',
    measure_landmark: '#ffffff',
    measure_mask: '#c77dff',
  },
  fillAlpha: 0.16,
  lineWidth: 2,
  brightness: 1,
  contrast: 1,
  invert: false,
};

function clone(value) { return JSON.parse(JSON.stringify(value)); }

export const appearance = {
  colors: clone(DEFAULTS.colors),
  fillAlpha: DEFAULTS.fillAlpha,
  lineWidth: DEFAULTS.lineWidth,
  brightness: 1,
  contrast: 1,
  invert: false,
  listeners: new Set(),

  load() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return this;
      const saved = JSON.parse(raw);
      if (saved && typeof saved === 'object') {
        for (const [key, value] of Object.entries(saved.colors || {})) {
          if (key in this.colors && /^#[0-9a-f]{6}$/i.test(value)) this.colors[key] = value;
        }
        if (Number.isFinite(saved.fillAlpha)) {
          this.fillAlpha = Math.min(0.6, Math.max(0, saved.fillAlpha));
        }
        if (Number.isFinite(saved.lineWidth)) {
          this.lineWidth = Math.min(4, Math.max(1, saved.lineWidth));
        }
        if (Number.isFinite(saved.brightness)) {
          this.brightness = Math.min(2.5, Math.max(0.3, saved.brightness));
        }
        if (Number.isFinite(saved.contrast)) {
          this.contrast = Math.min(2.5, Math.max(0.3, saved.contrast));
        }
        this.invert = Boolean(saved.invert);
      }
    } catch (error) {
      /* unreadable preference: the defaults above are a fine place to be */
    }
    return this;
  },

  save() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({
        colors: this.colors, fillAlpha: this.fillAlpha, lineWidth: this.lineWidth,
        brightness: this.brightness, contrast: this.contrast, invert: this.invert,
      }));
    } catch (error) { /* blocked storage: the session still honours the change */ }
    for (const listener of this.listeners) listener();
  },

  reset() {
    this.colors = clone(DEFAULTS.colors);
    this.fillAlpha = DEFAULTS.fillAlpha;
    this.lineWidth = DEFAULTS.lineWidth;
    this.resetWindow();
  },

  resetWindow() {
    this.brightness = DEFAULTS.brightness;
    this.contrast = DEFAULTS.contrast;
    this.invert = DEFAULTS.invert;
    this.save();
  },

  /**
   * Window/level as a canvas filter string.
   *
   * Applied to the image draw only — the overlay must keep its own colours, so
   * every caller sets this, draws the bitmap, then clears the filter before the
   * contours go down.
   */
  imageFilter() {
    const parts = [];
    if (this.brightness !== 1) parts.push(`brightness(${this.brightness.toFixed(3)})`);
    if (this.contrast !== 1) parts.push(`contrast(${this.contrast.toFixed(3)})`);
    if (this.invert) parts.push('invert(1)');
    return parts.length ? parts.join(' ') : 'none';
  },

  colorOf(key, fallback = '#ffffff') {
    return this.colors[key] || fallback;
  },

  onChange(listener) { this.listeners.add(listener); },
};

export function withAlpha(hex, alpha) {
  const value = parseInt(String(hex).slice(1), 16);
  if (Number.isNaN(value)) return `rgba(255, 255, 255, ${alpha})`;
  return `rgba(${(value >> 16) & 255}, ${(value >> 8) & 255}, ${value & 255}, ${alpha})`;
}

/** Lighten a colour toward white — used to make the hovered shape read as hovered. */
export function lighten(hex, amount = 0.35) {
  const value = parseInt(String(hex).slice(1), 16);
  if (Number.isNaN(value)) return hex;
  const mix = (channel) => Math.round(channel + (255 - channel) * amount);
  const r = mix((value >> 16) & 255);
  const g = mix((value >> 8) & 255);
  const b = mix(value & 255);
  return `rgb(${r}, ${g}, ${b})`;
}

appearance.load();

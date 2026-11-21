// Drawing the model output on top of a 640x640 render.
//
// One function serves both the grid tiles and the zoomed viewer: the caller passes how many device pixels one
// image pixel occupies, and every stroke width and marker radius is divided by it, so lines stay the same
// thickness on screen at any zoom instead of turning into slabs.

import { appearance, withAlpha } from './appearance.js';
import { formatMm, t } from './i18n.js';

export const FRAME = 640;

export const LAYERS = [
  { id: 'breast', label: 'c_breast', key: 'breast-tissue', kind: 'polygon', source: 'breast-tissue' },
  { id: 'pectoral', label: 'c_pectoral', key: 'pectoral', kind: 'polygon', source: 'pectoral' },
  { id: 'nipple', label: 'c_nipple', key: 'nipple', kind: 'polygon', source: 'nipple' },
  { id: 'points', label: 'layer_points', key: 'nipple_point', kind: 'points' },
  { id: 'measure_landmark', label: 'c_measure_landmark', key: 'measure_landmark', kind: 'measure', route: 'landmark' },
  { id: 'measure_mask', label: 'c_measure_mask', key: 'measure_mask', kind: 'measure', route: 'mask' },
];

export const layerName = (layer) => t(layer.label);
export const layerColor = (layer) => appearance.colorOf(layer.key);

/**
 * @param {CanvasRenderingContext2D} ctx  already translated/scaled to image space
 * @param {object} image                  one entry of study.images
 * @param {Set<string>} visible           layer ids to draw
 * @param {number} scale                  device pixels per image pixel
 */
export function drawOverlay(ctx, image, visible, scale) {
  const unit = 1 / Math.max(scale, 0.0001);
  ctx.save();
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';

  for (const layer of LAYERS) {
    if (!visible.has(layer.id)) continue;
    if (layer.kind === 'polygon') drawPolygon(ctx, image, layer, unit);
    else if (layer.kind === 'points') drawPoints(ctx, image, unit);
    else if (layer.kind === 'measure') drawMeasure(ctx, image, layer, unit);
  }

  ctx.restore();
}

function drawPolygon(ctx, image, layer, unit) {
  const entry = image.polygons && image.polygons[layer.source];
  const points = entry && entry.points;
  if (!points || points.length < 3) return;
  const color = layerColor(layer);

  ctx.beginPath();
  points.forEach(([x, y], index) => (index ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.closePath();
  ctx.fillStyle = withAlpha(color, appearance.fillAlpha);
  ctx.fill();
  ctx.strokeStyle = color;
  ctx.lineWidth = appearance.lineWidth * unit;
  ctx.stroke();
}

function drawPoints(ctx, image, unit) {
  const points = (image.landmarks && image.landmarks.points) || {};
  const radius = 5 * unit;

  for (const [name, position] of Object.entries(points)) {
    const [x, y] = position;
    const color = appearance.colorOf(name === 'nipple' ? 'nipple_point' : name);

    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = 'rgba(0, 0, 0, .75)';
    ctx.lineWidth = 1.5 * unit;
    ctx.stroke();

    // Crosshair: the fill alone is hard to place precisely against bright tissue.
    ctx.beginPath();
    for (const [dx, dy] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
      ctx.moveTo(x + dx * radius * 1.1, y + dy * radius * 1.1);
      ctx.lineTo(x + dx * radius * 2.1, y + dy * radius * 2.1);
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.4 * unit;
    ctx.stroke();

    label(ctx, t(`pt_${name}`, {}, name), x + radius * 2.4, y - radius * 1.6, color, unit);
  }
}

/** What the MLO quality of this view was judged from (640 frame), when it was graded. */
function qualityBasis(image) {
  const item = (image.criteria || []).find((entry) => entry.id === 'mlo_quality');
  return item && item.basis ? item.basis : null;
}

/** The fitted pectoral line, dashed. */
function drawPectoralLine(ctx, line, color, unit) {
  ctx.beginPath();
  ctx.moveTo(line[0][0], line[0][1]);
  ctx.lineTo(line[1][0], line[1][1]);
  ctx.strokeStyle = withAlpha(color, 0.75);
  ctx.lineWidth = 1.8 * unit;
  ctx.setLineDash([7 * unit, 5 * unit]);
  ctx.stroke();
  ctx.setLineDash([]);
}

/**
 * The pectoral line and the perpendicular from the nipple the MLO quality used. Dashed and without a millimetre
 * label: it shows what the verdict rests on, it is not a measured distance.
 */
function drawBasis(ctx, basis, color, unit) {
  drawPectoralLine(ctx, basis.line, color, unit);
  ctx.strokeStyle = withAlpha(color, 0.75);
  ctx.beginPath();
  ctx.moveTo(basis.nipple[0], basis.nipple[1]);
  ctx.lineTo(basis.foot[0], basis.foot[1]);
  ctx.lineWidth = 1.6 * unit;
  ctx.setLineDash([3 * unit, 4 * unit]);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.arc(basis.nipple[0], basis.nipple[1], 3 * unit, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
}

function drawMeasure(ctx, image, layer, unit) {
  const measure = image.measures && image.measures[layer.route];
  const color = layerColor(layer);
  if (!measure || !measure.ok) {
    // The MLO quality is judged from the line the segmentation route fitted, so it is drawn with that route. (When
    // the route measured, the quality used the very nipple and line of the distance drawn below.)
    const basis = layer.route === 'mask' ? qualityBasis(image) : null;
    // No distance was measured (no nipple mask), yet the MLO quality was graded from the pectoral line and the
    // nipple it fell back to: draw those, so the reviewer sees what the verdict is based on. With no nipple at all
    // only the pectoral angle was read off the line, and the line alone is drawn.
    if (basis) drawBasis(ctx, basis, color, unit);
    else if (measure && measure.line) drawPectoralLine(ctx, measure.line, color, unit);
    return;
  }

  // The pectoral line the perpendicular is dropped onto, when the route has one.
  if (measure.line) drawPectoralLine(ctx, measure.line, color, unit);

  // Measured without a nipple (the contour fallback): the distance is dashed and starts at an open ring on the
  // tissue contour, and its label says it is an estimate. The MLO quality still judged the nipple it fell back to
  // (the pose point); that perpendicular is drawn as well when it starts elsewhere.
  const estimated = Boolean(measure.estimated);
  const basis = estimated && layer.route === 'mask' ? qualityBasis(image) : null;
  if (basis && (basis.nipple[0] !== measure.from[0] || basis.nipple[1] !== measure.from[1])) {
    drawBasis(ctx, basis, color, unit);
  }

  const [x1, y1] = measure.from;
  const [x2, y2] = measure.to;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.2 * unit;
  if (estimated) ctx.setLineDash([8 * unit, 4 * unit]);
  ctx.stroke();
  ctx.setLineDash([]);

  [[x1, y1], [x2, y2]].forEach(([x, y], index) => {
    ctx.beginPath();
    if (estimated && index === 0) {
      ctx.arc(x, y, 4.5 * unit, 0, Math.PI * 2);
      ctx.lineWidth = 1.8 * unit;
      ctx.stroke();
      return;
    }
    ctx.arc(x, y, 3 * unit, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  });

  const text = estimated ? `${formatMm(measure.mm)} ${t('est_short')}` : formatMm(measure.mm);
  label(ctx, text, (x1 + x2) / 2 + 8 * unit, (y1 + y2) / 2 - 8 * unit, color, unit);
}

export function label(ctx, text, x, y, color, unit) {
  const size = 12 * unit;
  ctx.font = `500 ${size}px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace`;
  ctx.textBaseline = 'bottom';
  const width = ctx.measureText(text).width;
  const padX = 4 * unit;
  const padY = 3 * unit;

  ctx.fillStyle = 'rgba(8, 10, 13, .78)';
  ctx.fillRect(x - padX, y - size - padY, width + padX * 2, size + padY * 2);
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
}

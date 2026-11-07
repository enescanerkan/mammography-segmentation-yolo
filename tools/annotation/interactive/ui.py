"""Gradio UI layer (orchestration only).

This module knows how to:
  - lay out the Gradio Blocks,
  - convert click events into PromptState mutations,
  - call the inference pipeline,
  - persist masks **per tissue** across tissue switches,
  - render all stored masks + current prompts simultaneously.

It does NOT know how the model loads, what an "ignore disk" is, or how
class competition is computed — those live in their own modules.
"""

from __future__ import annotations

from typing import Optional

import gradio as gr
import numpy as np

from .inference import InferencePipeline, InferenceSettings, PredictionResult
from .prompts import (
    TISSUE_PRESETS,
    BoxPrompt,
    PromptState,
    TissuePreset,
    ensure_rgb,
    render_overlay,
    tissue_by_label,
)


MODE_POSITIVE = "Pozitif Nokta (foreground)"
MODE_IGNORE = "Yoksay Noktası (ignore — hard subtract)"
MODE_BOX = "Bounding Box (iki tık ile)"


_LIVE_BOX_HEAD = """
<style>
  #mammo_input { position: relative; }
  #mammo_input .mammo-preview-canvas {
    position: absolute;
    pointer-events: none;
    z-index: 50;
  }
</style>
<script>
(function () {
  // Live rubber-band box preview between the two clicks in BOX mode.
  // Pure visual overlay; Python still receives the final two clicks via Gradio's
  // standard `select` event.
  const POLL_MS = 250;
  const COLOR_BAND = 'rgba(220, 220, 60, 0.95)';
  const COLOR_GUIDE = 'rgba(220, 220, 60, 0.55)';
  const COLOR_FILL = 'rgba(220, 220, 60, 0.12)';

  const state = { firstCorner: null, attachedImg: null, canvas: null };

  function isBoxMode() {
    const root = document.querySelector('#mode_radio');
    if (!root) return false;
    const sel = root.querySelector('input[type="radio"]:checked');
    if (!sel) return false;
    const lab = sel.closest('label');
    return !!(lab && lab.innerText && lab.innerText.indexOf('Bounding Box') !== -1);
  }

  function clearCanvas() {
    if (!state.canvas) return;
    state.canvas.getContext('2d').clearRect(0, 0, state.canvas.width, state.canvas.height);
  }

  function ensureCanvas(container) {
    if (state.canvas && state.canvas.parentElement === container) return state.canvas;
    if (state.canvas) state.canvas.remove();
    const c = document.createElement('canvas');
    c.className = 'mammo-preview-canvas';
    container.appendChild(c);
    state.canvas = c;
    return c;
  }

  function syncCanvasSize(img) {
    if (!state.canvas) return;
    const imgRect = img.getBoundingClientRect();
    const parentRect = state.canvas.parentElement.getBoundingClientRect();
    state.canvas.width = Math.max(1, Math.round(imgRect.width));
    state.canvas.height = Math.max(1, Math.round(imgRect.height));
    state.canvas.style.width = imgRect.width + 'px';
    state.canvas.style.height = imgRect.height + 'px';
    state.canvas.style.left = (imgRect.left - parentRect.left) + 'px';
    state.canvas.style.top = (imgRect.top - parentRect.top) + 'px';
  }

  function drawPreview(x, y) {
    if (!state.canvas || !state.firstCorner) return;
    const ctx = state.canvas.getContext('2d');
    const W = state.canvas.width, H = state.canvas.height;
    ctx.clearRect(0, 0, W, H);
    const fx = state.firstCorner.x, fy = state.firstCorner.y;
    const x0 = Math.min(fx, x), y0 = Math.min(fy, y);
    const x1 = Math.max(fx, x), y1 = Math.max(fy, y);
    const w = x1 - x0, h = y1 - y0;
    ctx.fillStyle = COLOR_FILL;
    ctx.fillRect(x0, y0, w, h);
    ctx.strokeStyle = COLOR_BAND;
    ctx.lineWidth = 2;
    ctx.setLineDash([10, 6]);
    ctx.strokeRect(x0, y0, w, h);
    ctx.setLineDash([]);
    ctx.strokeStyle = COLOR_GUIDE;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, y); ctx.lineTo(W, y);
    ctx.moveTo(x, 0); ctx.lineTo(x, H);
    ctx.stroke();
    ctx.fillStyle = COLOR_BAND;
    ctx.font = '13px ui-monospace, Menlo, monospace';
    const dims = Math.round(w) + ' x ' + Math.round(h);
    ctx.fillText(dims, Math.min(x + 8, W - 60), Math.max(y - 8, 14));
  }

  function attach(img) {
    if (state.attachedImg === img) return;
    state.attachedImg = img;
    state.firstCorner = null;

    const container = document.querySelector('#mammo_input');
    if (!container) return;
    ensureCanvas(container);
    setTimeout(() => syncCanvasSize(img), 30);

    img.addEventListener('click', (e) => {
      if (!isBoxMode()) {
        state.firstCorner = null;
        clearCanvas();
        return;
      }
      const r = img.getBoundingClientRect();
      const x = e.clientX - r.left;
      const y = e.clientY - r.top;
      if (state.firstCorner === null) {
        state.firstCorner = { x: x, y: y };
        drawPreview(x, y);
      } else {
        state.firstCorner = null;
        clearCanvas();
      }
    }, true);

    img.addEventListener('mousemove', (e) => {
      if (!state.firstCorner) return;
      const r = img.getBoundingClientRect();
      drawPreview(e.clientX - r.left, e.clientY - r.top);
    });

    img.addEventListener('mouseleave', () => {
      // keep firstCorner; just hide the live cursor lines
      if (!state.firstCorner) return;
      // redraw without live point
      drawPreview(state.firstCorner.x, state.firstCorner.y);
    });

    img.addEventListener('load', () => {
      state.firstCorner = null;
      clearCanvas();
      setTimeout(() => syncCanvasSize(img), 30);
    });

    try {
      new ResizeObserver(() => syncCanvasSize(img)).observe(img);
    } catch (e) {}

    const modeRoot = document.querySelector('#mode_radio');
    if (modeRoot) {
      modeRoot.querySelectorAll('input[type="radio"]').forEach((r) => {
        r.addEventListener('change', () => {
          state.firstCorner = null;
          clearCanvas();
        });
      });
    }
  }

  function poll() {
    const img = document.querySelector('#mammo_input img');
    if (img && img.complete && img.naturalWidth > 0) attach(img);
    setTimeout(poll, POLL_MS);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', poll);
  } else {
    poll();
  }
})();
</script>
"""


_HEADER_MD = """\
# MedSAM2 — İnteraktif Mamografi Segmentasyonu

Sol tıklayarak nokta veya bounding box ekle. Maske her tıklamada otomatik güncellenir.

- **Pozitif Nokta**: dokunun *içine* tıkla. Birden fazla atılabilir.
- **Yoksay Noktası (ignore)**: bu noktanın etrafı **kesinlikle** maske dışında kalır
  (R yarıçaplı disk maskeden mekanik çıkarılır + modele soft negatif olarak verilir).
- **Bounding Box**: *iki tık* ile dikdörtgen çiz. 1) bir köşeye tıkla → büyük sarı
  X belirir, 2) mouse'u oynat → kutu **canlı önizleme** halinde takip eder
  (CVAT/labelImg gibi), 3) karşı köşeye tıkla → maske otomatik üretilir.

**Çoklu doku**: her doku tipinde ayrı maske oluştur. Doku tipini değiştirdiğinde
önceki maske ekrana yapışık kalır; yeni tipte ayrı çizim yaparsın.
**Model class-agnostic**: doku tipi yalnızca rengi ve class-competition hedefini
belirler — modele hangi dokuyu istediğini söylemez. Doğru bölgeye işaret etmek
sana kalmış.
"""


def _empty_prompt_state() -> dict:
    return {
        "positive": [],
        "ignore": [],
        "box": {"x0": None, "y0": None, "x1": None, "y1": None, "finished": False},
    }


def _empty_session_state() -> dict:
    """Top-level session state: prompts + per-tissue masks dictionary."""
    return {
        "prompts": _empty_prompt_state(),
        "masks": {},
    }


def _prompts_from_dict(d: dict) -> PromptState:
    s = PromptState()
    s.positive = [tuple(p) for p in d.get("positive", [])]
    s.ignore = [tuple(p) for p in d.get("ignore", [])]
    b = d.get("box") or {}
    s.box = BoxPrompt(
        x0=b.get("x0"), y0=b.get("y0"),
        x1=b.get("x1"), y1=b.get("y1"),
        finished=bool(b.get("finished", False)),
    )
    return s


def _prompts_to_dict(s: PromptState) -> dict:
    return {
        "positive": list(s.positive),
        "ignore": list(s.ignore),
        "box": {
            "x0": s.box.x0, "y0": s.box.y0,
            "x1": s.box.x1, "y1": s.box.y1,
            "finished": s.box.finished,
        },
    }


def _apply_click(state: PromptState, mode: str, x: int, y: int) -> None:
    if mode == MODE_POSITIVE:
        state.positive.append((float(x), float(y)))
    elif mode == MODE_IGNORE:
        state.ignore.append((float(x), float(y)))
    elif mode == MODE_BOX:
        if state.box.finished or state.box.x0 is None:
            state.box.reset()
            state.box.x0, state.box.y0 = float(x), float(y)
        else:
            state.box.x1, state.box.y1 = float(x), float(y)
            state.box.finished = True


def _info_md(
    session: dict,
    settings: InferenceSettings,
    tissue: TissuePreset,
    result: Optional[PredictionResult],
) -> str:
    s = _prompts_from_dict(session["prompts"])
    lines = []
    lines.append(f"**Aktif doku:** {tissue.label}")
    lines.append(f"**Pozitif noktalar:** {len(s.positive)}")
    lines.append(
        f"**Ignore noktaları:** {len(s.ignore)}  (R = {settings.ignore_radius_px} px → "
        f"disk başına ~{int(round(3.14159 * settings.ignore_radius_px ** 2))} px)"
    )
    box = s.box.to_xyxy()
    if box is not None:
        x0, y0, x1, y1 = (int(v) for v in box.tolist())
        lines.append(
            f"**Box (xyxy):** [{x0}, {y0}, {x1}, {y1}]  "
            f"({x1 - x0}×{y1 - y0} = {(x1 - x0) * (y1 - y0)} px alan)"
        )
    elif s.box.first_corner_only():
        lines.append(
            f"**Box:** ilk köşe ({int(s.box.x0)}, {int(s.box.y0)}) — ikinci tık bekliyor"
        )
    else:
        lines.append("**Box:** —")

    stored = session.get("masks", {}) or {}
    if stored:
        names = []
        for k, m in stored.items():
            if m is None:
                continue
            preset = TISSUE_PRESETS.get(k)
            label = preset.label if preset else k
            names.append(f"{label} ({int(m.sum())}px)")
        if names:
            lines.append("**Kaydedilmiş maskeler:** " + ", ".join(names))

    if result is None:
        lines.append("\n*Bu doku için maske henüz yok (en az 1 prompt ekle).*")
        return "\n\n".join(lines)

    lines.append("")
    lines.append(f"**Aktif maske pikselleri:** {result.pixel_count:,}")
    lines.append(f"**SAM skor:** {result.score:.3f}")
    if result.used_implicit_box and result.used_box is not None:
        b = result.used_box.astype(int).tolist()
        lines.append(f"**Implicit box:** {b}")
    if result.notes:
        lines.append("")
        for n in result.notes:
            lines.append(f"- {n}")
    return "\n\n".join(lines)


def _settings_from_inputs(
    ignore_radius: int,
    use_implicit_box: bool,
    use_class_competition: bool,
    use_soft_negative: bool,
) -> InferenceSettings:
    return InferenceSettings(
        ignore_radius_px=int(ignore_radius),
        use_implicit_box=bool(use_implicit_box),
        use_class_competition=bool(use_class_competition),
        use_soft_negative=bool(use_soft_negative),
    )


def _predict_and_render(
    rgb: np.ndarray,
    session: dict,
    tissue: TissuePreset,
    settings: InferenceSettings,
    pipeline: InferencePipeline,
) -> tuple[np.ndarray, dict, Optional[PredictionResult]]:
    """Run pipeline, store result mask under tissue key, render all masks."""
    state = _prompts_from_dict(session["prompts"])
    masks = dict(session.get("masks", {}) or {})

    result = pipeline.run(rgb, state, tissue, settings, masks_by_tissue=masks)
    if result is not None:
        masks[tissue.key] = result.mask

    overlay = render_overlay(
        rgb, state, masks,
        current_tissue_key=tissue.key,
        ignore_radius=settings.ignore_radius_px,
    )
    session = {"prompts": session["prompts"], "masks": masks}
    return overlay, session, result


def _render_only(
    rgb: np.ndarray,
    session: dict,
    tissue: TissuePreset,
    settings: InferenceSettings,
) -> np.ndarray:
    """Render canvas WITHOUT running the model (used after tissue switch)."""
    state = _prompts_from_dict(session["prompts"])
    masks = session.get("masks", {}) or {}
    return render_overlay(
        rgb, state, masks,
        current_tissue_key=tissue.key,
        ignore_radius=settings.ignore_radius_px,
    )


def build_app(pipeline: InferencePipeline) -> gr.Blocks:
    """Build the Gradio Blocks app. The ``pipeline`` is captured by reference."""

    custom_css = """
    .info-panel { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }
    .gradio-container { max-width: 1480px !important; }
    """

    with gr.Blocks(
        theme=gr.themes.Soft(primary_hue="indigo"),
        css=custom_css,
        title="MedSAM2 — Mamografi",
        head=_LIVE_BOX_HEAD,
    ) as demo:
        gr.Markdown(_HEADER_MD)

        original_rgb_st = gr.State(value=None)
        session_st = gr.State(value=_empty_session_state())
        last_result_st = gr.State(value=None)

        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                gr.Markdown("### 1) Tıklama Modu")
                mode = gr.Radio(
                    choices=[MODE_POSITIVE, MODE_IGNORE, MODE_BOX],
                    value=MODE_BOX,
                    label="Sol tık ne yapacak?",
                    elem_id="mode_radio",
                )

                gr.Markdown("### 2) Doku Sınıfı (renk + class-competition)")
                tissue_label = gr.Radio(
                    choices=[p.label for p in TISSUE_PRESETS.values()],
                    value=TISSUE_PRESETS["breast"].label,
                    label="Hangi dokuyu çiziyorsun?",
                    info="Her doku tipi ayrı maske olarak saklanır. Tipi değiştirsen "
                         "eski maske kaybolmaz, sadece aktif çizimi sıfırlar.",
                )

                with gr.Accordion("3) Gelişmiş Ayarlar", open=False):
                    ignore_radius = gr.Slider(
                        minimum=4, maximum=80, step=1, value=20,
                        label="Ignore disk yarıçapı (px)",
                        info="Her ignore noktasının etrafından bu yarıçap kadar piksel mekanik silinir.",
                    )
                    use_implicit_box = gr.Checkbox(
                        value=True,
                        label="Implicit box (önerilir)",
                        info="Box yoksa pozitif noktaların etrafına otomatik kutu türet. "
                             "Fine-tune dağılımına hizalar.",
                    )
                    use_class_competition = gr.Checkbox(
                        value=True,
                        label="Class competition (önceden çizilen pectoral'i breast'ten çıkar)",
                        info="ÖNCE pectoral maskeyi çiz; SONRA breast tissue'ya geçtiğinde "
                             "üst üste binen pectoral pikselleri otomatik düşürülür. "
                             "Pectoral çizilmediyse no-op.",
                    )
                    use_soft_negative = gr.Checkbox(
                        value=True,
                        label="Modele soft negatif besle",
                        info="Hard subtract zaten yapılır; bu modele de ipucu verir (belt + suspenders).",
                    )

                gr.Markdown("### 4) Aksiyonlar")
                with gr.Row():
                    btn_undo_pos = gr.Button("↶ Son pozitif", size="sm")
                    btn_undo_ign = gr.Button("↶ Son ignore", size="sm")
                with gr.Row():
                    btn_reset_box = gr.Button("□ Box reset", size="sm")
                    btn_clear_tissue = gr.Button("⨯ Bu doku maskesini sil", size="sm")
                btn_clear_all = gr.Button("⨯⨯ TÜM maskeleri temizle", variant="secondary")
                btn_rerun = gr.Button("⟳ Yeniden tahminle", variant="primary")

            with gr.Column(scale=3):
                with gr.Row():
                    img_input = gr.Image(
                        label="Görüntü — buraya tıkla",
                        type="numpy",
                        interactive=True,
                        height=620,
                        elem_id="mammo_input",
                    )
                    img_output = gr.Image(
                        label="Tahmin (canlı) — tüm maskeler üst üste",
                        type="numpy",
                        interactive=False,
                        height=620,
                        elem_id="mammo_output",
                    )
                info_md = gr.Markdown("*Henüz görüntü yok.*", elem_classes=["info-panel"])

        # --- Event handlers ---------------------------------------------------

        def on_new_image(img):
            rgb = ensure_rgb(img)
            session = _empty_session_state()
            return rgb, session, rgb, None, "*Yeni görüntü yüklendi. Tıklayarak prompt eklemeye başla.*"

        img_input.change(
            on_new_image,
            inputs=[img_input],
            outputs=[original_rgb_st, session_st, img_output, last_result_st, info_md],
        )

        def on_click(
            rgb,
            session,
            mode_v,
            tissue_v,
            ignore_r,
            implicit_box_v,
            class_comp_v,
            soft_neg_v,
            evt: gr.SelectData,
        ):
            if rgb is None:
                return None, _empty_session_state(), None, "*Önce görüntü yükle.*"

            idx = evt.index
            if not (isinstance(idx, (list, tuple)) and len(idx) >= 2):
                return None, session, None, "*Tıklama koordinatı okunamadı.*"
            x, y = int(idx[0]), int(idx[1])

            state = _prompts_from_dict(session["prompts"])
            _apply_click(state, mode_v, x, y)
            session = {"prompts": _prompts_to_dict(state), "masks": session.get("masks", {})}

            tissue = tissue_by_label(tissue_v)
            settings = _settings_from_inputs(ignore_r, implicit_box_v, class_comp_v, soft_neg_v)
            overlay, session, result = _predict_and_render(rgb, session, tissue, settings, pipeline)
            return overlay, session, result, _info_md(session, settings, tissue, result)

        img_input.select(
            on_click,
            inputs=[
                original_rgb_st, session_st, mode, tissue_label,
                ignore_radius, use_implicit_box, use_class_competition, use_soft_negative,
            ],
            outputs=[img_output, session_st, last_result_st, info_md],
        )

        def on_rerun(rgb, session, tissue_v, ignore_r, implicit_box_v, class_comp_v, soft_neg_v):
            if rgb is None:
                return None, session, None, "*Önce görüntü yükle.*"
            tissue = tissue_by_label(tissue_v)
            settings = _settings_from_inputs(ignore_r, implicit_box_v, class_comp_v, soft_neg_v)
            overlay, session, result = _predict_and_render(rgb, session, tissue, settings, pipeline)
            return overlay, session, result, _info_md(session, settings, tissue, result)

        rerun_inputs = [
            original_rgb_st, session_st, tissue_label,
            ignore_radius, use_implicit_box, use_class_competition, use_soft_negative,
        ]
        rerun_outputs = [img_output, session_st, last_result_st, info_md]

        btn_rerun.click(on_rerun, inputs=rerun_inputs, outputs=rerun_outputs)
        for ctl in (ignore_radius, use_implicit_box, use_class_competition, use_soft_negative):
            ctl.change(on_rerun, inputs=rerun_inputs, outputs=rerun_outputs)

        def on_tissue_change(rgb, session, tissue_v, ignore_r, implicit_box_v, class_comp_v, soft_neg_v):
            """Tissue switched: reset prompts (start fresh for new tissue) but keep all stored masks."""
            if rgb is None:
                return session, None, None, "*Önce görüntü yükle.*"
            session = {"prompts": _empty_prompt_state(), "masks": session.get("masks", {})}
            tissue = tissue_by_label(tissue_v)
            settings = _settings_from_inputs(ignore_r, implicit_box_v, class_comp_v, soft_neg_v)
            overlay = _render_only(rgb, session, tissue, settings)
            return session, overlay, None, _info_md(session, settings, tissue, None)

        tissue_label.change(
            on_tissue_change,
            inputs=[
                original_rgb_st, session_st, tissue_label,
                ignore_radius, use_implicit_box, use_class_competition, use_soft_negative,
            ],
            outputs=[session_st, img_output, last_result_st, info_md],
        )

        def on_clear_all(rgb):
            if rgb is None:
                return _empty_session_state(), None, None, "*Görüntü yok.*"
            return _empty_session_state(), rgb, None, "*Tüm promptlar ve maskeler temizlendi.*"

        btn_clear_all.click(
            on_clear_all,
            inputs=[original_rgb_st],
            outputs=[session_st, img_output, last_result_st, info_md],
        )

        def on_clear_tissue(rgb, session, tissue_v, ignore_r, implicit_box_v, class_comp_v, soft_neg_v):
            if rgb is None:
                return session, None, None, "*Görüntü yok.*"
            tissue = tissue_by_label(tissue_v)
            masks = dict(session.get("masks", {}) or {})
            masks.pop(tissue.key, None)
            session = {"prompts": _empty_prompt_state(), "masks": masks}
            settings = _settings_from_inputs(ignore_r, implicit_box_v, class_comp_v, soft_neg_v)
            overlay = _render_only(rgb, session, tissue, settings)
            return session, overlay, None, _info_md(session, settings, tissue, None)

        btn_clear_tissue.click(
            on_clear_tissue,
            inputs=[
                original_rgb_st, session_st, tissue_label,
                ignore_radius, use_implicit_box, use_class_competition, use_soft_negative,
            ],
            outputs=[session_st, img_output, last_result_st, info_md],
        )

        def _undo_kind(kind: str):
            def _fn(rgb, session, tissue_v, ignore_r, implicit_box_v, class_comp_v, soft_neg_v):
                if rgb is None:
                    return session, None, None, "*Görüntü yok.*"
                state = _prompts_from_dict(session["prompts"])
                state.pop_last(kind)
                session = {"prompts": _prompts_to_dict(state), "masks": session.get("masks", {})}
                tissue = tissue_by_label(tissue_v)
                settings = _settings_from_inputs(ignore_r, implicit_box_v, class_comp_v, soft_neg_v)
                overlay, session, result = _predict_and_render(rgb, session, tissue, settings, pipeline)
                return session, overlay, result, _info_md(session, settings, tissue, result)
            return _fn

        undo_inputs = [
            original_rgb_st, session_st, tissue_label,
            ignore_radius, use_implicit_box, use_class_competition, use_soft_negative,
        ]
        undo_outputs = [session_st, img_output, last_result_st, info_md]

        btn_undo_pos.click(_undo_kind("positive"), inputs=undo_inputs, outputs=undo_outputs)
        btn_undo_ign.click(_undo_kind("ignore"), inputs=undo_inputs, outputs=undo_outputs)
        btn_reset_box.click(_undo_kind("box"), inputs=undo_inputs, outputs=undo_outputs)

    return demo

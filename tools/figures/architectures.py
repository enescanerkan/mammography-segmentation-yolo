"""Draw the architecture figures of docs/figures/architectures (light + dark SVG per model).

    python tools/figures/architectures.py            # writes docs/figures/architectures/*.svg

Every shape and parameter count below was read from the models themselves: forward hooks on a 640 x 640 input
for the YOLO checkpoints, SegFormer (``nvidia/mit-b3`` config of the trained model), the nnU-Net plans of the
trained model, U-Net / AttUNet (``breast_seg.training.nets``) and the MedSAM2 checkpoint. Shapes are
channels x height x width for one image.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from svgdiagram import Diagram, save  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "docs" / "figures" / "architectures"


def _rows(y0: float, pitch: float, n: int):
    return [y0 + i * pitch for i in range(n)]


# ── YOLO (one layout for YOLO26 and YOLOv8, pose / seg / detect heads) ─────────
def yolo(name: str, title: str, subtitle: str, backbone, neck, heads, proto, output, notes, legend_extra=(),
         legend_kinds=("io", "enc", "dec", "head", "attn")):
    """``backbone``: rows (title, shape, tag, kind); P3 / P4 / P5 are the rows tagged so.
    ``neck``: titles + shapes of the four neck nodes (td4, td3, bu4, bu5). ``heads``: per-level head lines."""
    W, pitch, bh = 1290, 50, 40
    d = Diagram(W, 900, title, subtitle)
    ys = _rows(112, pitch, len(backbone))
    d.panel(20, 76, 360, ys[-1] + bh + 18 - 76, "Backbone")
    boxes = []
    for (t, shape, tag, kind), y in zip(backbone, ys):
        boxes.append(d.block(40, y, 320, bh, t, shape, kind, tag))
    for a, b in zip(boxes, boxes[1:]):
        d.arrow(a.b, b.t)
    idx = {tag: i for i, (_, _, tag, _) in enumerate(backbone) if tag in ("P3", "P4", "P5")}
    p3, p4, p5 = (boxes[idx[k]] for k in ("P3", "P4", "P5"))

    d.panel(400, 76, 480, ys[-1] + bh + 18 - 76, "Neck", "top-down (FPN) · bottom-up (PAN)")
    nw, nh = 215, 50
    (t4, s4), (t3, s3), (u4, su4), (u5, su5) = neck
    td4 = d.block(418, p4.y - 5, nw, nh, t4, s4, "dec")
    td3 = d.block(418, p3.y - 5, nw, nh, t3, s3, "dec", "P3'")
    bu4 = d.block(650, p4.y - 5, nw, nh, u4, su4, "dec", "P4'")
    bu5 = d.block(650, p5.y - 5, nw, nh, u5, su5, "dec", "P5'")
    # top-down
    d.arrow(p5.r, (526, p5.r[1]), td4.b, label="", head=True)
    d.arrow(p4.r, td4.l, skip=True, label="concat")
    d.arrow(td4.t, td3.b)
    d.arrow(p3.r, td3.l, skip=True, label="concat")
    # bottom-up
    d.arrow(td3.r, (757, td3.r[1]), bu4.t)
    d.arrow(td4.r, bu4.l, skip=True)
    d.arrow(bu4.b, bu5.t)
    d.arrow((526, p5.r[1]), bu5.l, skip=True, label="concat", label_at=0.5)

    d.panel(900, 76, 370, ys[-1] + bh + 18 - 76, "Head")
    hw, hh = 210, 54
    hx = 912
    h3 = d.block(hx, td3.y - 2, hw, hh, heads[0][0], heads[0][1], "head", lines=heads[0][2:])
    h4 = d.block(hx, bu4.y - 2, hw, hh, heads[1][0], heads[1][1], "head", lines=heads[1][2:])
    h5 = d.block(hx, bu5.y - 2, hw, hh, heads[2][0], heads[2][1], "head", lines=heads[2][2:])
    d.arrow(td3.r, (880, td3.r[1]), (880, h3.l[1]), h3.l)
    d.arrow(bu4.r, h4.l)
    d.arrow(bu5.r, h5.l)
    if proto:
        pr = d.block(hx, td3.y - 100, hw, 50, proto[0], proto[1], "head")
        d.arrow((890, td3.r[1]), (890, pr.l[1]), pr.l)
    # merge column
    mx = 1140
    for h in (h3, h4, h5):
        d.arrow(h.r, (mx, h.r[1]), head=False)
    d.arrow((mx, h3.r[1]), (mx, h5.r[1]), head=False)
    out = d.block(1040, ys[-1] + 62, 220, 92, output[0], output[1], "io", lines=output[2:])
    d.arrow((mx, h5.r[1]), (mx, out.y))
    if proto:
        d.arrow(pr.r, (1250, pr.r[1]), (1250, out.y), dashed=True)

    y = ys[-1] + bh + 40
    for line in notes:
        d.text(28, y, line, 11.5)
        y += 17
    y = max(y, out.y + out.h) + 22
    d.legend(28, y, list(legend_kinds), (("skip", "lateral (concat)"), *legend_extra))
    d.height = y + 22
    return d


YOLO26_BACKBONE = [
    ("Input", "3 × 640 × 640", "", "io"),
    ("Conv 3×3, stride 2", "64 × 320 × 320", "/2", "enc"),
    ("Conv 3×3, stride 2", "128 × 160 × 160", "/4", "enc"),
    ("C3k2 ×{n}", "256 × 160 × 160", "", "enc"),
    ("Conv 3×3, stride 2", "256 × 80 × 80", "/8", "enc"),
    ("C3k2 ×{n}", "512 × 80 × 80", "P3", "enc"),
    ("Conv 3×3, stride 2", "512 × 40 × 40", "/16", "enc"),
    ("C3k2 ×{n} (C3k blocks)", "512 × 40 × 40", "P4", "enc"),
    ("Conv 3×3, stride 2", "512 × 20 × 20", "/32", "enc"),
    ("C3k2 ×{n} (C3k blocks)", "512 × 20 × 20", "", "enc"),
    ("SPPF (5×5 max-pool ×3)", "512 × 20 × 20", "", "enc"),
    ("C2PSA ×{n} (self-attention)", "512 × 20 × 20", "P5", "attn"),
]


def yolo26_backbone(n: int):
    return [(t.format(n=n), s, tag, k) for t, s, tag, k in YOLO26_BACKBONE]


def yolo26_neck(n: int):
    return [("upsample ×2, concat P4, C3k2 ×%d" % n, "512 × 40 × 40"),
            ("upsample ×2, concat P3, C3k2 ×%d" % n, "256 × 80 × 80"),
            ("conv s2, concat, C3k2 ×%d" % n, "512 × 40 × 40"),
            ("conv s2, concat P5, C3k2 (PSA)", "512 × 20 × 20")]


def fig_pose() -> Diagram:
    heads = [("Pose26 head · /8", "80 × 80 cells", "box 4 · score 1 · keypoints K×3"),
             ("Pose26 head · /16", "40 × 40 cells", "box 4 · score 1 · keypoints K×3"),
             ("Pose26 head · /32", "20 × 20 cells", "box 4 · score 1 · keypoints K×3")]
    output = ("Top 300 boxes, no NMS", "300 × (6 + 3K)", "highest-scoring 'Breast' box", "MLO: nipple, pectoral top,",
              "pectoral bottom · CC: nipple")
    notes = ["Ultralytics YOLO26l-pose, one class ('Breast') whose box carries the keypoints; input: wavelet640 canvas "
             "(NLM, db4 detail boost, CLAHE).",
             "Keypoints K = 3 for MLO (kpt_shape [3, 3]), K = 1 for CC ([1, 3]); each keypoint is (x, y, visibility). "
             "One-to-one assignment: the head ranks the 8 400 cells",
             "(80² + 40² + 20²) and keeps the top 300, so no non-maximum suppression is needed. Box regression is "
             "direct (reg_max 1, no DFL). Parameters: 27.93 M (MLO), 27.93 M (CC).",
             "l scale: width 1.0, depth 1.0 (C3k2 / C2PSA blocks repeated twice), at most 512 channels."]
    return yolo("yolo26l_pose", "Landmark models · YOLO26l-pose (MLO and CC)",
                "3 × 640 × 640 wavelet canvas  →  breast box + keypoints · 27.9 M parameters",
                yolo26_backbone(2), yolo26_neck(2), heads, None, output, notes)


def fig_seg() -> Diagram:
    heads = [("Segment26 head · /8", "80 × 80 cells", "box 4 · class 3 · coef 32"),
             ("Segment26 head · /16", "40 × 40 cells", "box 4 · class 3 · coef 32"),
             ("Segment26 head · /32", "20 × 20 cells", "box 4 · class 3 · coef 32")]
    proto = ("Mask prototypes (from P3')", "32 × 160 × 160")
    output = ("Top 300 instances, no NMS", "300 × (6 + 32)", "mask = coef · prototypes", "→ 160 × 160 → 640 × 640",
              "pectoral · breast tissue · nipple")
    notes = ["Ultralytics YOLO26m-seg: instance segmentation of pectoral muscle, breast tissue and nipple on the "
             "baseline640 canvas (MLO and CC in one model).",
             "Each of the 8 400 cells predicts a box, 3 class scores and 32 mask coefficients; the instance mask is the "
             "coefficients times 32 prototype masks (sigmoid, cropped to the box).",
             "One-to-one assignment keeps the top 300 instances without NMS; box regression is direct (reg_max 1). "
             "m scale: width 1.0, depth 0.5 (blocks once), at most 512 channels.",
             "Parameters: 26.97 M. Dataset generations v1–v5 use this network (v4 is released). YOLO26l-seg "
             "(benchmark): same layout with the blocks repeated twice, 31.38 M."]
    return yolo("yolo26m_seg", "Segmentation model · YOLO26m-seg (v1–v5, v4 released)",
                "3 × 640 × 640 canvas  →  instances of pectoral, breast tissue, nipple · 27.0 M parameters",
                yolo26_backbone(1), yolo26_neck(1), heads, proto, output, notes)


def fig_yolov8() -> Diagram:
    backbone = [
        ("Input", "3 × 640 × 640", "", "io"),
        ("Conv 3×3, stride 2", "48 × 320 × 320", "/2", "enc"),
        ("Conv 3×3, stride 2", "96 × 160 × 160", "/4", "enc"),
        ("C2f ×2", "96 × 160 × 160", "", "enc"),
        ("Conv 3×3, stride 2", "192 × 80 × 80", "/8", "enc"),
        ("C2f ×4", "192 × 80 × 80", "P3", "enc"),
        ("Conv 3×3, stride 2", "384 × 40 × 40", "/16", "enc"),
        ("C2f ×4", "384 × 40 × 40", "P4", "enc"),
        ("Conv 3×3, stride 2", "576 × 20 × 20", "/32", "enc"),
        ("C2f ×2", "576 × 20 × 20", "", "enc"),
        ("SPPF (5×5 max-pool ×3)", "576 × 20 × 20", "P5", "enc"),
    ]
    neck = [("upsample ×2, concat P4, C2f ×2", "384 × 40 × 40"),
            ("upsample ×2, concat P3, C2f ×2", "192 × 80 × 80"),
            ("conv s2, concat, C2f ×2", "384 × 40 × 40"),
            ("conv s2, concat P5, C2f ×2", "576 × 20 × 20")]
    heads = [("Segment head · /8", "80 × 80 cells", "box 4×16 (DFL) · cls · coef 32"),
             ("Segment head · /16", "40 × 40 cells", "box 4×16 (DFL) · cls · coef 32"),
             ("Segment head · /32", "20 × 20 cells", "box 4×16 (DFL) · cls · coef 32")]
    proto = ("Mask prototypes (from P3')", "32 × 160 × 160")
    output = ("NMS over 8 400 cells", "instances", "mask = coef · prototypes", "→ 160 × 160 → 640 × 640",
              "(detect head: boxes only)")
    notes = ["Ultralytics YOLOv8m with a decoupled anchor-free head: box sides as 16-bin distributions (DFL), class "
             "scores and, for segmentation, 32 mask coefficients per cell.",
             "Detections are filtered by non-maximum suppression. m scale: width 0.75, depth 0.67, at most 768 "
             "channels.",
             "YOLOv8m-seg (segmentation benchmark, 3 classes): 27.24 M parameters. YOLOv8l-seg: width 1.0, depth 1.0 "
             "(64-128-256-512-512 channels, C2f ×3/6/6/3), 45.94 M.",
             "Lesion detector: the same YOLOv8m backbone and neck with the Detect head (boxes + 2 classes, Mass / "
             "Calcification), trained at 1024 × 1024, 25.86 M."]
    return yolo("yolov8m", "Benchmark backbone and lesion detector · YOLOv8m (seg / detect)",
                "3 × 640 × 640 canvas  →  instances (seg) or boxes (detect) · 27.2 M (seg) / 25.9 M (detect)",
                backbone, neck, heads, proto, output, notes, legend_kinds=("io", "enc", "dec", "head"))


# ── U-shaped networks ──────────────────────────────────────────────────────────
def u_net(name, title, subtitle, enc, bottom, dec, head, notes, input_block, gates=False, extra_head=None,
          deep_supervision=None, legend=("io", "enc", "dec", "head"), down="", up=""):
    """``enc`` / ``dec``: (title, shape) per level, top to bottom / bottom to top; ``bottom``: (title, shape)."""
    n = len(enc)
    pitch, bh, bw, step = 68, 46, 250, 22
    W = 1240 if deep_supervision else 1120
    d = Diagram(W, 900, title, subtitle)
    ys = _rows(178, pitch, n)
    inp = d.block(40, 84, bw, 44, *input_block, kind="io")
    enc_right = 40 + (n - 1) * step + bw
    d.panel(20, 146, enc_right + 20 - 20, ys[-1] + bh + 14 - 146, "Encoder")
    encs = [d.block(40 + i * step, y, bw, bh, t, s, "enc") for i, ((t, s), y) in enumerate(zip(enc, ys))]
    d.arrow(inp.b, encs[0].t)
    for a, b in zip(encs, encs[1:]):
        d.arrow((a.x + 30, a.y + a.h), (a.x + 30, b.y), label=down, label_dx=8, label_dy=4, label_anchor="start")
    limit = W - (190 if deep_supervision else 40)
    dx = limit - (n - 1) * step - bw
    d.panel(dx - 20, 146, limit + 20 - (dx - 20), ys[-1] + bh + 14 - 146, "Decoder")
    by = ys[-1] + pitch + 10
    bot = d.block(W / 2 - 160, by, 320, bh, bottom[0], bottom[1], "enc")
    d.arrow((encs[-1].x + 60, encs[-1].y + bh), (encs[-1].x + 60, bot.l[1]), bot.l)
    decs = []
    for i, ((t, s), y) in enumerate(zip(dec, list(reversed(ys)))):
        decs.append(d.block(dx + (n - 1 - i) * step, y, bw, bh, t, s, "dec"))
    d.arrow(bot.r, (decs[0].x + bw - 60, bot.r[1]), (decs[0].x + bw - 60, decs[0].y + bh))
    for a, b in zip(decs, decs[1:]):
        d.arrow((a.x + bw - 30, a.y), (a.x + bw - 30, b.y + bh), label=up, label_dx=-8, label_dy=4,
                label_anchor="end")
    for e, dd in zip(encs, reversed(decs)):
        if gates:
            g = d.op((e.r[0] + dd.l[0]) / 2, e.r[1], "×", 12, "attn")
            d.arrow(e.r, g.l, skip=True, label="skip x")
            d.arrow(g.r, dd.l, skip=True, label="α · x")
            d.text(g.c[0], g.c[1] + 26, "gate α(x, g)", 10.5, "middle")
        else:
            d.arrow(e.r, dd.l, skip=True, label="skip (concat)")
    top = decs[-1]
    if extra_head:
        hb = d.block(top.x + bw - 250, 84, 250, 44, head[0], head[1], "head")
        eb = d.block(top.x + bw - 250 - 270, 84, 250, 44, extra_head[0], extra_head[1], "head")
        jx, jy = hb.b[0], 137
        d.arrow((jx, top.y), hb.b)
        d.arrow((jx, jy), (eb.b[0], jy), eb.b)
    else:
        hb = d.block(top.x + bw - 250, 84, 250, 44, head[0], head[1], "head")
        d.arrow((hb.b[0], top.y), hb.b)
    if deep_supervision:
        for dd, (t, s) in zip(decs[:-1], deep_supervision):
            ds = d.block(W - 170, dd.y + 6, 150, 34, t, s, "head")
            d.arrow(dd.r, ds.l, dashed=True)
    y = by + bh + 34
    for line in notes:
        d.text(28, y, line, 11.5)
        y += 17
    extra = (("skip", "skip connection"),) + ((("dashed", "deep-supervision output"),) if deep_supervision else ())
    d.legend(28, y + 10, list(legend), extra)
    d.height = y + 34
    return d


def fig_unet() -> Diagram:
    enc = [("DoubleConv", "64 × 640 × 640"), ("DoubleConv", "128 × 320 × 320"), ("DoubleConv", "256 × 160 × 160"),
           ("DoubleConv", "512 × 80 × 80")]
    dec = [("up-conv, concat, DoubleConv", "512 × 80 × 80"), ("up-conv, concat, DoubleConv", "256 × 160 × 160"),
           ("up-conv, concat, DoubleConv", "128 × 320 × 320"), ("up-conv, concat, DoubleConv", "64 × 640 × 640")]
    notes = ["Plain U-Net (Ronneberger 2015), trained from scratch: DoubleConv = (3×3 conv, BatchNorm, ReLU) ×2; "
             "2×2 max-pool down, 2×2 transposed convolution up,",
             "skip features concatenated before each decoder DoubleConv. Input: the single-channel canvas scaled to "
             "[0, 1]. Output: 4-class logits (background, pectoral,",
             "breast tissue, nipple). Parameters: 31.04 M. The reference baseline without attention, residual blocks "
             "or pre-training."]
    return u_net("unet", "Semantic baseline · U-Net", "1 × 640 × 640 canvas  →  4 × 640 × 640 class logits · 31.0 M "
                 "parameters", enc, ("DoubleConv (bottleneck)", "1024 × 40 × 40"), dec, ("1×1 conv", "4 × 640 × 640"),
                 notes, ("Input", "1 × 640 × 640 (grey, 0–1)"), down="max-pool 2×2", up="up-conv 2×2")


def fig_attunet() -> Diagram:
    enc = [("DoubleConv", "64 × 640 × 640"), ("DoubleConv", "128 × 320 × 320"), ("DoubleConv", "256 × 160 × 160"),
           ("DoubleConv", "512 × 80 × 80")]
    dec = [("up-conv, gate, concat, DoubleConv", "512 × 80 × 80"), ("up-conv, gate, concat, DoubleConv", "256 × 160 × 160"),
           ("up-conv, gate, concat, DoubleConv", "128 × 320 × 320"), ("up-conv, gate, concat, DoubleConv", "64 × 640 × 640")]
    notes = ["Attention U-Net (Oktay 2018) with CoordConv input: the grey canvas plus normalised x and y coordinate "
             "channels. Each skip is multiplied by an attention gate",
             "α = σ(ψ(ReLU(W_g·g + W_x·x))) computed from the decoder signal g (1×1 convolutions, 256 / 128 / 64 / 32 "
             "intermediate channels) before concatenation.",
             "Two 1×1 heads on the last decoder features: 4-class segmentation and one heatmap per landmark, read out "
             "by a spatial soft-argmax. Landmark arm: landmark loss only;",
             "multitask arm: + segmentation loss through the shared network. Parameters: 31.39 M."]
    return u_net("attunet", "Landmark / multitask model · AttUNet",
                 "3 × 640 × 640 (grey + x + y)  →  4-class segmentation + 3 landmark heatmaps · 31.4 M parameters",
                 enc, ("DoubleConv (bottleneck)", "1024 × 40 × 40"), dec,
                 ("Landmark head 1×1 → soft-argmax", "3 heatmaps → 3 × (x, y)"), notes,
                 ("Input (CoordConv)", "3 × 640 × 640 (grey, x, y)"), gates=True,
                 extra_head=("Segmentation head 1×1", "4 × 640 × 640"), legend=("io", "enc", "dec", "head", "attn"),
                 down="max-pool 2×2", up="up-conv 2×2")


def fig_nnunet() -> Diagram:
    enc = [("stem conv + 1 res. block", "32 × 640 × 640"), ("3 residual blocks", "64 × 320 × 320"),
           ("4 residual blocks", "128 × 160 × 160"), ("6 residual blocks", "256 × 80 × 80"),
           ("6 residual blocks", "512 × 40 × 40"), ("6 residual blocks", "512 × 20 × 20")]
    dec = [("up-conv, concat, conv", "512 × 20 × 20"), ("up-conv, concat, conv", "512 × 40 × 40"),
           ("up-conv, concat, conv", "256 × 80 × 80"), ("up-conv, concat, conv", "128 × 160 × 160"),
           ("up-conv, concat, conv", "64 × 320 × 320"), ("up-conv, concat, conv", "32 × 640 × 640")]
    ds = [("deep supervision", "4 × 20 × 20"), ("deep supervision", "4 × 40 × 40"), ("deep supervision", "4 × 80 × 80"),
          ("deep supervision", "4 × 160 × 160"), ("deep supervision", "4 × 320 × 320")]
    notes = ["nnU-Net v2 residual-encoder U-Net, ResEnc-M preset, 2D configuration, plans of the trained model "
             "(patch = the whole 640 × 640 canvas, batch 12).",
             "Residual block = (3×3 conv, InstanceNorm, LeakyReLU) → (3×3 conv, InstanceNorm) + shortcut, LeakyReLU; "
             "the first block of a stage has stride 2 and an",
             "average-pool + 1×1 shortcut. Decoder: 2×2 transposed convolution, concatenated skip, one (3×3 conv, "
             "InstanceNorm, LeakyReLU). Every decoder level has a 1×1",
             "segmentation output (deep supervision during training; the full-resolution one at inference). Input: z-score "
             "normalised grey canvas. Parameters: 105.92 M."]
    return u_net("nnunet", "Semantic benchmark · nnU-Net ResEnc-M (2D)",
                 "1 × 640 × 640 canvas  →  4 × 640 × 640 class logits · 105.9 M parameters", enc,
                 ("6 residual blocks (stride 2)", "512 × 10 × 10"), dec, ("1×1 conv (output)", "4 × 640 × 640"), notes,
                 ("Input", "1 × 640 × 640 (z-score)"), deep_supervision=ds, down="stride 2", up="ConvT 2×2")


# ── SegFormer ──────────────────────────────────────────────────────────────────
def fig_segformer() -> Diagram:
    W = 1240
    d = Diagram(W, 700, "Semantic segmentation · SegFormer-B3",
                "3 × 640 × 640 canvas  →  4 × 640 × 640 class logits · 47.2 M parameters (encoder 44.1 M, decoder 3.2 M)")
    inp = d.block(40, 92, 300, 44, "Input", "3 × 640 × 640 (grey ×3, ImageNet norm.)", "io")
    d.panel(20, 150, 420, 420, "MiT-B3 encoder", "hierarchical transformer")
    stages = [
        ("Stage 1 · patch embed 7×7, s4", "64 × 160 × 160", "3 blocks · 1 head · reduction R = 8"),
        ("Stage 2 · patch embed 3×3, s2", "128 × 80 × 80", "4 blocks · 2 heads · R = 4"),
        ("Stage 3 · patch embed 3×3, s2", "320 × 40 × 40", "18 blocks · 5 heads · R = 2"),
        ("Stage 4 · patch embed 3×3, s2", "512 × 20 × 20", "3 blocks · 8 heads · R = 1"),
    ]
    sb = []
    for i, (t, s, ln) in enumerate(stages):
        sb.append(d.block(40, 182 + i * 96, 380, 66, t, s, "attn", f"/{4 * 2 ** i}", lines=[ln]))
    d.arrow(inp.b, sb[0].t)
    for a, b in zip(sb, sb[1:]):
        d.arrow(a.b, b.t)
    d.panel(470, 150, 750, 420, "All-MLP decoder")
    mlps = []
    for i, b in enumerate(sb):
        m = d.block(500, b.y + 9, 230, 48, f"Linear {stages[i][1].split(' ')[0]} → 768", "↑ to 768 × 160 × 160", "dec")
        d.arrow(b.r, m.l)
        mlps.append(m)
    cat = d.op(790, 345, "C", 14, "dec")
    for m in mlps:
        d.arrow(m.r, (760, m.r[1]), (760, 345), cat.l if m is mlps[1] or m is mlps[2] else cat.l, head=False)
    d.arrow((760, 345), cat.l)
    fuse = d.block(830, 300, 170, 90, "Fuse", "3072 → 768", "dec", lines=["1×1 conv, BN, ReLU", "dropout 0.1"])
    d.arrow(cat.r, fuse.l)
    cls = d.block(1030, 300, 170, 90, "Classifier", "1×1 conv → 4", "head", lines=["4 × 160 × 160", "(1/4 resolution)"])
    d.arrow(fuse.r, cls.l)
    out = d.block(1030, 470, 170, 66, "Output", "4 × 640 × 640", "io", lines=["bilinear ×4"])
    d.arrow(cls.b, out.t)
    notes = ["Each encoder block: efficient self-attention (keys and values spatially reduced by R with a strided "
             "convolution) and a Mix-FFN (MLP ratio 4 with a 3×3 depth-wise",
             "convolution instead of positional encodings); overlapping patch embeddings between stages. "
             "Configuration of the trained model (nvidia/mit-b3: depths 3-4-18-3,",
             "widths 64-128-320-512, drop path 0.1). Decoder: every stage projected to 768 channels, upsampled to "
             "1/4 resolution, concatenated and fused; logits upsampled",
             "bilinearly to the canvas. Prediction averages the softmax with the horizontally flipped pass."]
    y = 600
    for line in notes:
        d.text(28, y, line, 11.5)
        y += 17
    d.legend(28, y + 12, ["io", "attn", "dec", "head"])
    d.height = y + 36
    return d


# ── MedSAM2 ────────────────────────────────────────────────────────────────────
def fig_medsam2() -> Diagram:
    W = 1290
    d = Diagram(W, 760, "Prompted segmentation (annotation toolkit) · MedSAM2 fine-tune of SAM2 Hiera-tiny",
                "3 × 1024 × 1024 image + box  →  1 × 256 × 256 mask logits · 31.4 M parameters used (38.9 M in the "
                "checkpoint)")
    inp = d.block(30, 96, 190, 50, "Input image", "3 × 1024 × 1024", "io", lines=["raw1024"])
    d.panel(240, 80, 640, 300, "Image encoder", "Hiera-T trunk 26.8 M + FPN neck 0.4 M")
    pe = d.block(260, 110, 130, 50, "Patch embed", "7×7, s4", "enc")
    d.arrow(inp.r, pe.l)
    stages = [("Stage 1 · 1 block", "96 × 256 × 256", "window 8"), ("Stage 2 · 2 blocks", "192 × 128 × 128", "window 4"),
              ("Stage 3 · 7 blocks", "384 × 64 × 64", "window 14 · 3 global"), ("Stage 4 · 2 blocks", "768 × 32 × 32",
                                                                                 "window 7")]
    sx = [410, 410, 650, 650]
    sy = [110, 200, 110, 200]
    sbs = []
    for (t, s, ln), x, y in zip(stages, sx, sy):
        sbs.append(d.block(x, y, 210, 60, t, s, "attn", lines=[ln]))
    d.arrow(pe.r, sbs[0].l)
    d.arrow(sbs[0].b, sbs[1].t, label="Q-pool ×2", label_dx=8, label_dy=4, label_anchor="start")
    d.arrow(sbs[1].r, (635, sbs[1].r[1]), (635, sbs[2].l[1]), sbs[2].l)
    d.arrow(sbs[2].b, sbs[3].t, label="Q-pool ×2", label_dx=8, label_dy=4, label_anchor="start")
    fpn = d.block(410, 290, 450, 70, "FPN neck · 1×1 lateral → 256, top-down at the two coarse levels",
                  "levels kept: 256 × 256², 256 × 128², 256 × 64²", "dec", lines=["coarsest (32²) dropped (scalp 1)"])
    d.arrow((sbs[0].x, sbs[0].y + 48), (399, sbs[0].y + 48), (399, fpn.l[1]), fpn.l)
    d.arrow((sbs[1].b[0], sbs[1].y + sbs[1].h), (sbs[1].b[0], fpn.y))
    d.arrow(sbs[2].r, (872, sbs[2].r[1]), (872, fpn.r[1]), fpn.r)
    d.arrow((sbs[3].b[0], sbs[3].y + sbs[3].h), (sbs[3].b[0], fpn.y))
    emb = d.block(240, 410, 290, 60, "Image embedding", "256 × 64 × 64", "enc", lines=["64² level + no-memory embedding"])
    hr = d.block(560, 410, 320, 60, "High-res features (conv_s0, conv_s1)", "32 × 256²  ·  64 × 128²", "dec")
    d.arrow((470, fpn.y + fpn.h), (470, 390), (385, 390), emb.t)
    d.arrow((720, fpn.y + fpn.h), hr.t)
    d.panel(30, 500, 500, 150, "Prompt encoder (frozen)")
    de = d.block(262, 528, 250, 54, "Dense prompt + positional enc.", "256 × 64 × 64 each", "frozen",
                 lines=["added to / used with the embedding"])
    sp = d.block(262, 594, 250, 40, "Sparse tokens", "3 × 256 (2 corners + pad)", "frozen")
    box = d.block(50, 586, 175, 56, "Box prompt", "(x0, y0, x1, y1)", "cond", lines=["2 corners, labels 2 / 3"])
    d.arrow(box.r, sp.l)
    d.arrow(de.l, (247, de.l[1]), (247, emb.y + emb.h))
    d.panel(560, 500, 710, 210, "Mask decoder", "4.2 M")
    tok = d.block(580, 536, 200, 60, "Output tokens", "obj-score · IoU · 4 mask", "head", lines=["+ prompt tokens"])
    tw = d.block(810, 530, 210, 72, "Two-way transformer ×2", "256-d · 8 heads · MLP 2048", "attn",
                 lines=["tokens ↔ image attention"])
    d.arrow(sp.r, (545, sp.r[1]), (545, tok.l[1]), tok.l)
    d.arrow(tok.r, tw.l)
    d.arrow(emb.r, (538, emb.r[1]), (538, 486), (915, 486), tw.t)
    up = d.block(1050, 530, 200, 72, "Upscaling", "ConvT ×2 → 64 × 128²", "dec", lines=["+ s1, ConvT ×2 → 32 × 256²",
                                                                                         "+ s0"])
    d.arrow(tw.r, up.l)
    d.arrow(hr.r, (1150, hr.r[1]), up.t)
    hyp = d.block(810, 632, 210, 58, "Hypernetwork MLP", "mask token → 32 weights", "head", lines=["IoU + object-score heads"])
    out = d.block(1050, 632, 200, 58, "Mask logits", "1 × 256 × 256", "io", lines=["single mask (multimask off)"])
    d.arrow(tw.b, hyp.t)
    d.arrow(hyp.r, out.l)
    d.arrow(up.b, out.t)
    notes = ["SAM2 (Hiera-tiny, configuration sam2_hiera_t) fine-tuned through the MedSAM2 branch of the MedSAM code "
             "base: image encoder and mask decoder trained, prompt encoder frozen.",
             "The box enters as its two corners (point labels 2 and 3). Hiera blocks use windowed attention (window "
             "8 / 4 / 14 / 7 per stage) except three global blocks in stage 3;",
             "query pooling halves the resolution between stages. The high-resolution FPN levels skip the "
             "transformer and are added during upscaling. Memory attention and memory",
             "encoder (7.5 M, for video) are in the checkpoint but not used for single images. Target: one binary "
             "256 × 256 mask per structure and image."]
    y = 735
    for line in notes:
        d.text(28, y, line, 11.5)
        y += 17
    d.legend(28, y + 12, ["io", "enc", "attn", "dec", "head", "cond", "frozen"])
    d.height = y + 36
    return d


FIGURES = {
    "yolo26l_pose": fig_pose,
    "yolo26m_seg": fig_seg,
    "segformer_b3": fig_segformer,
    "nnunet_resenc_m": fig_nnunet,
    "unet": fig_unet,
    "attunet": fig_attunet,
    "yolov8m": fig_yolov8,
    "medsam2": fig_medsam2,
}


def main(argv=None) -> int:
    names = (argv if argv is not None else sys.argv[1:]) or list(FIGURES)
    for name in names:
        for path in save(FIGURES[name](), OUT, name):
            print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

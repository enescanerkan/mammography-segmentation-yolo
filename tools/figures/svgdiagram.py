"""A small SVG block-diagram writer for the architecture figures (light and dark variants from one drawing).

Shapes are recorded once and rendered per theme, so ``docs/figures/architectures/<name>.svg`` and
``<name>_dark.svg`` always show the same diagram. Colours follow the applications' Clinical Light / Dark palette.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape

FONT = "Inter, 'Segoe UI', -apple-system, Helvetica, Arial, sans-serif"
MONO = "'JetBrains Mono', 'SFMono-Regular', Consolas, 'Liberation Mono', monospace"

# kind -> (fill, stroke) per theme
THEMES: Dict[str, Dict] = {
    "light": {
        "bg": "#FFFFFF", "panel": "#F6F7F9", "panel_line": "#E3E6EC", "ink": "#14213D", "muted": "#5B6478",
        "arrow": "#6B7488", "skip": "#0E7C7B",
        "kinds": {
            "io": ("#F3F1EC", "#8A93A6"), "enc": ("#E7ECF5", "#2A3B66"), "dec": ("#E1F0EF", "#0E7C7B"),
            "head": ("#F8EDDB", "#B9771A"), "attn": ("#EEE8F6", "#6B4FA0"), "cond": ("#F6E3E0", "#B03A2E"),
            "op": ("#FFFFFF", "#6B7488"), "frozen": ("#EEF0F3", "#8A93A6"),
        },
    },
    "dark": {
        "bg": "#0D1117", "panel": "#131B2B", "panel_line": "#253049", "ink": "#E6EAF2", "muted": "#9BA6BB",
        "arrow": "#8F9AB0", "skip": "#4FB8B6",
        "kinds": {
            "io": ("#1A2233", "#7E899E"), "enc": ("#18264A", "#8FA8D8"), "dec": ("#0E3331", "#4FB8B6"),
            "head": ("#36290F", "#E0A54A"), "attn": ("#2A2240", "#A98BDB"), "cond": ("#3B1E1B", "#E07A6E"),
            "op": ("#131B2B", "#8F9AB0"), "frozen": ("#1A2030", "#6E788C"),
        },
    },
}

LEGEND_LABELS = {
    "io": "input / output", "enc": "encoder / backbone", "dec": "decoder / neck", "head": "head",
    "attn": "attention / transformer", "cond": "conditioning", "frozen": "frozen / not used",
}

Point = Tuple[float, float]


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def l(self) -> Point:  # noqa: E743
        return (self.x, self.y + self.h / 2)

    @property
    def r(self) -> Point:
        return (self.x + self.w, self.y + self.h / 2)

    @property
    def t(self) -> Point:
        return (self.x + self.w / 2, self.y)

    @property
    def b(self) -> Point:
        return (self.x + self.w / 2, self.y + self.h)

    @property
    def c(self) -> Point:
        return (self.x + self.w / 2, self.y + self.h / 2)

    def at(self, fx: float, fy: float) -> Point:
        return (self.x + fx * self.w, self.y + fy * self.h)


@dataclass
class Diagram:
    width: float
    height: float
    title: str
    subtitle: str = ""
    _back: List[tuple] = field(default_factory=list)
    _items: List[tuple] = field(default_factory=list)
    _front: List[tuple] = field(default_factory=list)

    # ── primitives ─────────────────────────────────────────────────────────
    def panel(self, x, y, w, h, label: str = "", sub: str = "") -> Box:
        self._back.append(("panel", x, y, w, h, label, sub))
        return Box(x, y, w, h)

    def block(self, x, y, w, h, title: str, sub: str = "", kind: str = "enc", tag: str = "",
              lines: Sequence[str] = ()) -> Box:
        """A layer / stage: bold title, a shape line (monospace) and optional extra lines; ``tag`` at the right."""
        self._items.append(("block", x, y, w, h, title, sub, kind, tag, tuple(lines)))
        return Box(x, y, w, h)

    def op(self, x, y, label: str, r: float = 11, kind: str = "op") -> Box:
        """A small circular operator (concat, add, multiply) centred at (x, y)."""
        self._items.append(("op", x, y, r, label, kind))
        return Box(x - r, y - r, 2 * r, 2 * r)

    def arrow(self, *points: Point, dashed: bool = False, skip: bool = False, label: str = "",
              label_at: float = 0.5, label_dy: float = -5, head: bool = True, color_kind: Optional[str] = None,
              label_dx: float = 0, label_anchor: str = "middle"):
        self._front.append(("arrow", tuple(points), dashed, skip, label, label_at, label_dy, head, color_kind,
                            label_dx, label_anchor))

    def text(self, x, y, s: str, size: float = 11.5, anchor: str = "start", muted: bool = True, bold: bool = False,
             mono: bool = False, italic: bool = False):
        self._front.append(("text", x, y, s, size, anchor, muted, bold, mono, italic))

    def legend(self, x, y, kinds: Sequence[str], extra: Sequence[Tuple[str, str]] = ()):
        """Colour keys, then (style, label) extras: style ``skip`` (dashed teal) or ``arrow``."""
        self._front.append(("legend", x, y, tuple(kinds), tuple(extra)))

    # ── rendering ──────────────────────────────────────────────────────────
    def render(self, theme: str = "light") -> str:
        th = THEMES[theme]
        out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width:.0f}" height="{self.height:.0f}" '
               f'viewBox="0 0 {self.width:.0f} {self.height:.0f}" font-family="{FONT}">']
        out.append("<defs>")
        for name, color in (("a", th["arrow"]), ("s", th["skip"])):
            out.append(f'<marker id="ah-{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
                       f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{color}"/>'
                       "</marker>")
        for kind, (_, stroke) in th["kinds"].items():
            out.append(f'<marker id="ah-{kind}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
                       f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{stroke}"/>'
                       "</marker>")
        out.append("</defs>")
        out.append(f'<rect width="100%" height="100%" rx="10" fill="{th["bg"]}"/>')
        out.append(self._t(24, 34, self.title, 19, "start", th["ink"], bold=True))
        if self.subtitle:
            out.append(self._t(24, 56, self.subtitle, 12.5, "start", th["muted"]))
        for item in self._back:
            out.append(self._render(item, th))
        for item in self._items:
            out.append(self._render(item, th))
        for item in self._front:
            out.append(self._render(item, th))
        out.append("</svg>")
        return "\n".join(s for s in out if s)

    def _t(self, x, y, s, size, anchor, color, bold=False, mono=False, italic=False) -> str:
        weight = ' font-weight="600"' if bold else ""
        family = f' font-family="{MONO}"' if mono else ""
        style = ' font-style="italic"' if italic else ""
        return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{color}"'
                f"{weight}{family}{style}>{escape(s)}</text>")

    def _render(self, item, th) -> str:
        kind = item[0]
        if kind == "panel":
            _, x, y, w, h, label, sub = item
            s = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{th["panel"]}" '
                 f'stroke="{th["panel_line"]}"/>']
            if label:
                s.append(self._t(x + 12, y + 19, label.upper(), 10.5, "start", th["muted"], bold=True))
            if sub:
                s.append(self._t(x + w - 12, y + 19, sub, 10.5, "end", th["muted"]))
            return "\n".join(s)
        if kind == "block":
            _, x, y, w, h, title, sub, k, tag, lines = item
            fill, stroke = th["kinds"][k]
            s = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}" '
                 f'stroke-width="1.2"/>']
            rows = [(title, "title")] + ([(sub, "sub")] if sub else []) + [(ln, "line") for ln in lines]
            heights = {"title": 14.5, "sub": 13.5, "line": 13}
            total = sum(heights[r] for _, r in rows)
            cy = y + (h - total) / 2 + 11
            for text, role in rows:
                if role == "title":
                    s.append(self._t(x + 10, cy, text, 12.5, "start", th["ink"], bold=True))
                elif role == "sub":
                    s.append(self._t(x + 10, cy, text, 11, "start", th["muted"], mono=True))
                else:
                    s.append(self._t(x + 10, cy, text, 11, "start", th["muted"]))
                cy += heights[role]
            if tag:
                s.append(self._t(x + w - 9, y + h - 8, tag, 10.5, "end", stroke, bold=True))
            return "\n".join(s)
        if kind == "op":
            _, x, y, r, label, k = item
            fill, stroke = th["kinds"][k]
            return (f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>'
                    + self._t(x, y + 4.5, label, 13, "middle", th["ink"], bold=True))
        if kind == "arrow":
            _, pts, dashed, skip, label, label_at, label_dy, head, ck, label_dx, label_anchor = item
            if ck:
                color, marker = th["kinds"][ck][1], f"ah-{ck}"
            elif skip:
                color, marker = th["skip"], "ah-s"
            else:
                color, marker = th["arrow"], "ah-a"
            d = "M" + " L".join(f"{px:.1f},{py:.1f}" for px, py in pts)
            dash = ' stroke-dasharray="5 4"' if dashed or skip else ""
            mk = f' marker-end="url(#{marker})"' if head else ""
            s = [f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.4"{dash}{mk}/>']
            if label:
                (x0, y0), (x1, y1) = _segment_at(pts, label_at)
                lx, ly = x0 + (x1 - x0) * 0.5, y0 + (y1 - y0) * 0.5
                s.append(self._t(lx + label_dx, ly + label_dy, label, 10.5, label_anchor, th["muted"]))
            return "\n".join(s)
        if kind == "text":
            _, x, y, s_, size, anchor, muted, bold, mono, italic = item
            return self._t(x, y, s_, size, anchor, th["muted"] if muted else th["ink"], bold=bold, mono=mono,
                           italic=italic)
        if kind == "legend":
            _, x, y, kinds, extra = item
            s, cx = [], x
            for k in kinds:
                fill, stroke = th["kinds"][k]
                s.append(f'<rect x="{cx}" y="{y - 10}" width="14" height="12" rx="3" fill="{fill}" stroke="{stroke}"/>')
                label = LEGEND_LABELS[k]
                s.append(self._t(cx + 20, y, label, 11, "start", th["muted"]))
                cx += 20 + 6.4 * len(label) + 22
            for style, label in extra:
                color = th["skip"] if style == "skip" else th["arrow"]
                dash = ' stroke-dasharray="5 4"' if style in ("skip", "dashed") else ""
                s.append(f'<path d="M{cx},{y - 4} L{cx + 26},{y - 4}" stroke="{color}" stroke-width="1.4"{dash}/>')
                s.append(self._t(cx + 32, y, label, 11, "start", th["muted"]))
                cx += 32 + 6.4 * len(label) + 22
            return "\n".join(s)
        raise ValueError(kind)


def _segment_at(pts: Sequence[Point], frac: float) -> Tuple[Point, Point]:
    """The polyline segment that holds the point at ``frac`` of the total length."""
    import math

    lengths = [math.dist(a, b) for a, b in zip(pts, pts[1:])]
    target, acc = frac * sum(lengths), 0.0
    for (a, b), seg in zip(zip(pts, pts[1:]), lengths):
        if acc + seg >= target:
            return a, b
        acc += seg
    return pts[-2], pts[-1]


def save(diagram: Diagram, out_dir, name: str) -> List[str]:
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for theme, suffix in (("light", ""), ("dark", "_dark")):
        path = out_dir / f"{name}{suffix}.svg"
        path.write_text(diagram.render(theme) + "\n", encoding="utf-8")
        written.append(str(path))
    return written

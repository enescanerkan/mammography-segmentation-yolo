"""Zoomable annotation canvas (QGraphicsView).

Modes
-----
- BOX  : two-click box (CVAT-style live rubber-band preview; corners
         remain draggable after finalize).
- POINT: left = positive, right = ignore. Both modes also accept point
         clicks so refinement after a box draw is always available.
- EDIT_POLYGON : the active tissue's mask contour is shown as a
         draggable polygon (drag vertex / click edge to insert / Alt+click
         to delete). On every change, the polygon rasterizes back into a
         mask and is broadcast via ``polygonMutated``.

Mouse / keyboard
----------------
- Ctrl + Wheel             : zoom at cursor
- Middle-click drag        : pan
- Double-click background  : fit to view (reset zoom)
- Left click               : positive point  /  box corner
- Right click              : ignore point
- Drag any vertex / handle : move (live update on release)
- Alt + click vertex/point : delete
- Esc                      : cancel in-progress box first corner
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import QLineF, QPoint, QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QWheelEvent,
)
from PyQt5.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
)

from interactive.prompts import TISSUE_PRESETS, BoxPrompt, PromptState

from .image_io import apply_window_level
from .theme import (
    ACCENT,
    ALERT,
    SUCCESS,
    TEXT_PRIMARY,
    TISSUE_RGB,
)


# Z layering
Z_IMAGE = 0
Z_MASK = 1
Z_POLYGON_EDGE = 40
Z_BOX = 50
Z_IGNORE_DISK = 60
Z_POINT = 100
Z_HANDLE = 110


MODE_IDLE = "idle"           # nothing selected — clicks do not annotate
MODE_POINT = "point"
MODE_BOX = "box"
MODE_EDIT_POLYGON = "edit_polygon"
MODE_DRAW_POLYGON = "draw_polygon"

# Hit tolerance for "is the click near an existing point?" — in scene pixels.
POINT_HIT_TOL = 18.0
EDGE_HIT_TOL = 8.0


# ─── Graphics items ──────────────────────────────────────────────────────


class DraggablePoint(QGraphicsEllipseItem):
    """Click-and-drag positive/ignore point marker."""

    def __init__(self, kind: str, radius_px: float, color: QColor) -> None:
        super().__init__(QRectF(-radius_px, -radius_px, 2 * radius_px, 2 * radius_px))
        self.kind = kind  # "positive" | "ignore"
        self.setBrush(QBrush(color))
        pen = QPen(QColor("#0a1014"))
        pen.setWidthF(1.6)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setAcceptHoverEvents(True)
        self.setCursor(QCursor(Qt.OpenHandCursor))
        self.setZValue(Z_POINT)

    def mousePressEvent(self, event):
        self.setCursor(QCursor(Qt.ClosedHandCursor))
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.setCursor(QCursor(Qt.OpenHandCursor))
        super().mouseReleaseEvent(event)
        view = self.scene().views()[0] if self.scene() and self.scene().views() else None
        if isinstance(view, MammoCanvas):
            view._on_point_released(self)


class BoxHandle(QGraphicsEllipseItem):
    """Corner handle for a finalized box. ``corner`` ∈ {nw, ne, sw, se}.

    The handle uses a re-entry guard (``_suppress_drag_callback``) because
    its sibling handles snap to derived positions during a drag; without
    the guard, the peer ``setPos`` calls would recursively trigger
    ``itemChange`` → handler → more peer ``setPos`` → SystemError.
    """

    def __init__(self, corner: str, radius_px: float = 6.5) -> None:
        super().__init__(QRectF(-radius_px, -radius_px, 2 * radius_px, 2 * radius_px))
        self.corner = corner
        self._suppress_drag_callback = False
        self.setBrush(QBrush(QColor(TEXT_PRIMARY)))
        pen = QPen(QColor(ACCENT))
        pen.setWidthF(2.0)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setAcceptHoverEvents(True)
        self.setCursor(QCursor(_handle_cursor(corner)))
        self.setZValue(Z_HANDLE)

    def itemChange(self, change, value):
        if (
            change == QGraphicsItem.ItemPositionChange
            and not self._suppress_drag_callback
        ):
            scene = self.scene()
            if scene is not None and scene.views():
                view = scene.views()[0]
                if isinstance(view, MammoCanvas):
                    clamped = view._on_box_handle_dragging(self, value)
                    if clamped is not None:
                        return clamped
        return super().itemChange(change, value)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        scene = self.scene()
        if scene is None or not scene.views():
            return
        view = scene.views()[0]
        if isinstance(view, MammoCanvas):
            view._on_box_handle_released(self)


class PolyBoxHandle(QGraphicsEllipseItem):
    """Corner handle used while editing a polygon that happens to be an
    axis-aligned 4-vertex rectangle (detector output / box-converted manual
    box). Dragging maintains the rectangle invariant — the opposite corner
    stays put, the two adjacent corners snap to the new ``x`` / ``y``."""

    def __init__(self, corner: str, radius_px: float = 7.0) -> None:
        super().__init__(QRectF(-radius_px, -radius_px, 2 * radius_px, 2 * radius_px))
        self.corner = corner
        self._suppress_drag_callback = False
        self.setBrush(QBrush(QColor(TEXT_PRIMARY)))
        pen = QPen(QColor(ACCENT))
        pen.setWidthF(2.2)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setAcceptHoverEvents(True)
        self.setCursor(QCursor(_handle_cursor(corner)))
        self.setZValue(Z_HANDLE)

    def itemChange(self, change, value):
        if (
            change == QGraphicsItem.ItemPositionChange
            and not self._suppress_drag_callback
        ):
            scene = self.scene()
            if scene is not None and scene.views():
                view = scene.views()[0]
                if isinstance(view, MammoCanvas):
                    clamped = view._on_polybox_corner_dragging(self, value)
                    if clamped is not None:
                        return clamped
        return super().itemChange(change, value)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        scene = self.scene()
        if scene is None or not scene.views():
            return
        view = scene.views()[0]
        if isinstance(view, MammoCanvas):
            view._on_polybox_corner_released(self)


class PolygonVertex(QGraphicsEllipseItem):
    """Draggable vertex for the polygon edit mode."""

    def __init__(self, idx: int, color: QColor, radius_px: float = 5.5) -> None:
        super().__init__(QRectF(-radius_px, -radius_px, 2 * radius_px, 2 * radius_px))
        self.idx = idx
        self.setBrush(QBrush(QColor(TEXT_PRIMARY)))
        pen = QPen(color)
        pen.setWidthF(2.0)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setAcceptHoverEvents(True)
        self.setCursor(QCursor(Qt.OpenHandCursor))
        self.setZValue(Z_POINT)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            view = self.scene().views()[0] if self.scene() and self.scene().views() else None
            if isinstance(view, MammoCanvas):
                view._on_polygon_vertex_dragging(self, value)
        return super().itemChange(change, value)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        view = self.scene().views()[0] if self.scene() and self.scene().views() else None
        if isinstance(view, MammoCanvas):
            view._on_polygon_vertex_released(self)


# ─── Canvas ──────────────────────────────────────────────────────────────


class MammoCanvas(QGraphicsView):
    """The interactive image canvas."""

    promptsMutated = pyqtSignal()
    polygonMutated = pyqtSignal(str)        # tissue_key whose polygon changed
    polygonDrawCompleted = pyqtSignal(str)  # tissue_key whose draw was just closed
    activePolygonRequested = pyqtSignal(int)  # back_idx the user clicked on
    polygonRenameRequested = pyqtSignal(int)  # back_idx of polygon to rename
    polygonContextMenuRequested = pyqtSignal(int, QPoint)  # back_idx, screen pos
    backgroundClickedInEdit = pyqtSignal(float, float)  # scene x, y — to allow MainWindow to chain a new box
    zoomChanged = pyqtSignal(float)
    windowLevelChanged = pyqtSignal(float, float)  # window, level

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("canvas")
        self.setRenderHints(QPainter.SmoothPixmapTransform | QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        # Image + overlays
        self._image_item: Optional[QGraphicsPixmapItem] = None
        self._image_hw: tuple[int, int] = (0, 0)
        self._mask_items: dict[str, QGraphicsPixmapItem] = {}

        # Prompt graphics
        self._point_items: list[DraggablePoint] = []
        self._ignore_disk_items: list[QGraphicsEllipseItem] = []
        self._box_rect_item: Optional[QGraphicsRectItem] = None
        self._box_handles: list[BoxHandle] = []
        self._box_preview_item: Optional[QGraphicsRectItem] = None
        self._first_corner_marker: Optional[QGraphicsPathItem] = None

        # State
        self._mode: str = MODE_BOX
        self._active_tissue: str = "breast"
        self._state: PromptState = PromptState()
        self._ignore_radius_px: int = 20

        # Polygon edit
        self._poly_vertices: list[PolygonVertex] = []
        self._poly_path_item: Optional[QGraphicsPathItem] = None
        self._poly_fill_item: Optional[QGraphicsPolygonItem] = None
        self._poly_points: list[tuple[float, float]] = []
        self._suppress_polygon_signals: bool = False  # for batch updates
        # Box-style edit (4-vertex axis-aligned rectangle). When True the
        # polygon is rendered with corner handles instead of free
        # vertices, and edge-click vertex insertion is disabled — so the
        # box stays a box.
        self._edit_as_box: bool = False
        self._polybox_handles: list[PolyBoxHandle] = []
        # Class-label pill pinned to the polygon's top-left corner.
        self._poly_label_item: Optional[QGraphicsTextItem] = None

        # Manual polygon draw (MODE_DRAW_POLYGON)
        self._draw_pts: list[tuple[float, float]] = []
        self._draw_path_item: Optional[QGraphicsPathItem] = None
        self._draw_dot_items: list[QGraphicsEllipseItem] = []

        # Sibling polygon outlines — drawn for every polygon-instance of
        # the active tissue OTHER than the one currently being edited.
        # Used so the user can SEE every detection at once and click on a
        # sibling to switch the edit cursor to it (multi-instance UX).
        self._other_outline_items: list[QGraphicsPathItem] = []
        # Geometry mirror so we can hit-test clicks without rebuilding the
        # QPainterPath each time. Tuples are (back_idx, polygon_pts).
        self._other_outline_data: list[tuple[int, list[tuple[float, float]]]] = []

        # Pan
        self._panning: bool = False
        self._pan_last: QPoint = QPoint()
        self._box_drag_in_progress: bool = False  # block re-emit during corner drag

        # Window/Level (DICOM-style brightness/contrast)
        self._base_rgb: Optional[np.ndarray] = None  # unmodified pixels
        self._window: float = 0.0
        self._level: float = 0.0
        self._wl_dragging: bool = False
        self._wl_start_pos: QPoint = QPoint()
        self._wl_start_window: float = 0.0
        self._wl_start_level: float = 0.0
        self._wl_drag_threshold_px: int = 4

        # Cached mask arrays so we can re-rasterize pixmaps when the user
        # changes the overlay opacity at runtime.
        self._mask_arrays: dict[str, np.ndarray] = {}
        self._mask_alpha: int = 38  # 0..255 — 15 % default (matches the side-panel slider)

        # Whether right-click in BOX / POINT modes drops a SAM-style
        # "ignore" point. Manual mode has no such concept — MainWindow
        # toggles this off there to keep right-click reserved for the
        # context menu.
        self._allow_right_click_ignore: bool = True

    # ─── Public API ────────────────────────────────────────────────────

    def load_image(self, rgb: np.ndarray) -> None:
        # Build the pixmap FIRST so that if the image is unsupported we
        # raise before mutating any state — keeps the current image
        # visible while the caller decides what to do.
        pixmap = _rgb_to_pixmap(rgb)

        # Wipe scene + every dangling reference. ``scene.clear()`` deletes
        # the underlying C++ objects; we need to clear every Python attr
        # that pointed at one of them, otherwise subsequent code touches
        # a freed item and raises ``RuntimeError: wrapped C/C++ object …
        # has been deleted``.
        self._scene.clear()
        self._image_item = None
        self._mask_items.clear()
        self._mask_arrays.clear()
        self._point_items = []
        self._ignore_disk_items = []
        self._box_rect_item = None
        self._box_handles = []
        self._box_preview_item = None
        self._first_corner_marker = None
        self._poly_vertices = []
        self._poly_path_item = None
        self._poly_fill_item = None
        self._poly_points = []
        self._polybox_handles = []
        self._poly_label_item = None
        self._edit_as_box = False
        self._draw_pts = []
        self._draw_path_item = None
        self._draw_dot_items = []
        self._other_outline_items = []
        self._other_outline_data = []
        self._state = PromptState()

        self._base_rgb = np.ascontiguousarray(rgb.astype(np.uint8))
        self._window = 0.0
        self._level = 0.0

        self._image_hw = (rgb.shape[0], rgb.shape[1])
        self._image_item = self._scene.addPixmap(pixmap)
        self._image_item.setZValue(Z_IMAGE)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        self._emit_zoom()
        self.windowLevelChanged.emit(self._window, self._level)

    def _safe_remove(self, item) -> None:
        """``scene.removeItem`` that swallows the ``RuntimeError`` you get
        from touching a wrapped C++ object that was already destroyed
        (typically because ``scene.clear()`` ran between when we cached
        the reference and now)."""
        if item is None:
            return
        try:
            self._scene.removeItem(item)
        except RuntimeError:
            pass

    def reset_window_level(self) -> None:
        self._window = 0.0
        self._level = 0.0
        self._refresh_image_pixmap()
        self.windowLevelChanged.emit(self._window, self._level)

    def get_window_level(self) -> tuple[float, float]:
        return (self._window, self._level)

    def _refresh_image_pixmap(self) -> None:
        if self._image_item is None or self._base_rgb is None:
            return
        try:
            rgb = apply_window_level(self._base_rgb, self._window, self._level)
            self._image_item.setPixmap(_rgb_to_pixmap(rgb))
        except Exception:
            # Cached image was a shape we can no longer render — skip
            # silently rather than crashing the W/L slider / reset hot
            # paths. The user can pick a new file from the list.
            pass

    def set_active_tissue(self, tissue_key: str) -> None:
        self._active_tissue = tissue_key
        for it in self._point_items:
            it.setBrush(QBrush(_point_color(tissue_key, it.kind)))

    def set_mode(self, mode: str) -> None:
        if mode not in (MODE_IDLE, MODE_POINT, MODE_BOX, MODE_EDIT_POLYGON, MODE_DRAW_POLYGON):
            return
        if self._mode == mode:
            return
        # Leaving polygon edit mode → clear vertices
        if self._mode == MODE_EDIT_POLYGON and mode != MODE_EDIT_POLYGON:
            self._clear_polygon_graphics()
        # Leaving box mode mid-draw → cancel first corner
        if mode != MODE_BOX and self._state.box.first_corner_only():
            self._state.box.reset()
            self._clear_first_corner_marker()
            self._clear_box_preview()
        # Leaving draw-polygon mode → wipe in-progress sketch
        if self._mode == MODE_DRAW_POLYGON and mode != MODE_DRAW_POLYGON:
            self._cancel_polygon_draw()
        self._mode = mode

    def get_mode(self) -> str:
        return self._mode

    def set_allow_right_click_ignore(self, allow: bool) -> None:
        """Enable / disable the legacy "right-click drops an ignore point"
        behavior. Manual mode wants it off — right-click should be
        reserved for the context menu, not for adding annotation
        artifacts."""
        self._allow_right_click_ignore = bool(allow)

    def set_ignore_radius(self, r: int) -> None:
        self._ignore_radius_px = int(max(1, r))
        self._refresh_ignore_disks()

    def set_mask(self, tissue_key: str, mask: Optional[np.ndarray]) -> None:
        prev = self._mask_items.pop(tissue_key, None)
        if prev is not None:
            self._scene.removeItem(prev)
        if mask is None or not mask.any():
            self._mask_arrays.pop(tissue_key, None)
            return
        self._mask_arrays[tissue_key] = mask.astype(np.uint8)
        color = TISSUE_RGB.get(tissue_key, (255, 200, 0))
        pixmap = _mask_to_pixmap(mask, color, alpha=self._mask_alpha)
        item = self._scene.addPixmap(pixmap)
        item.setZValue(Z_MASK)
        self._mask_items[tissue_key] = item

    def set_mask_alpha(self, alpha: int) -> None:
        """Change overlay opacity for every stored tissue mask AND the
        active polygon-edit fill so the slider feels uniform across both
        ``visual sources``."""
        alpha = int(max(0, min(255, alpha)))
        if alpha == self._mask_alpha:
            return
        self._mask_alpha = alpha
        for key, mask in list(self._mask_arrays.items()):
            color = TISSUE_RGB.get(key, (255, 200, 0))
            new_pixmap = _mask_to_pixmap(mask, color, alpha=self._mask_alpha)
            item = self._mask_items.get(key)
            if item is not None:
                item.setPixmap(new_pixmap)
        # Sync the active polygon-edit fill so a single slider drives all
        # tinted regions consistently.
        if self._poly_fill_item is not None:
            color = QColor(*TISSUE_RGB.get(self._active_tissue, (255, 200, 0)))
            self._poly_fill_item.setBrush(QBrush(QColor(
                color.red(), color.green(), color.blue(), self._mask_alpha,
            )))

    def set_prompt_state(self, state: PromptState) -> None:
        self._state = state
        self._rebuild_prompt_graphics()

    def get_prompt_state(self) -> PromptState:
        return self._state

    def clear_active_prompts(self) -> None:
        self._state = PromptState()
        self._rebuild_prompt_graphics()
        self.promptsMutated.emit()

    def clear_all_visuals(self) -> None:
        """Hard-reset the canvas: drop prompts, polygons, draw-in-progress,
        and every mask overlay. Image and zoom stay. Used by the 'Clear All'
        button so manual mode doesn't leave polygon vertices on screen."""
        self._state = PromptState()
        self._rebuild_prompt_graphics()
        self._cancel_polygon_draw()
        self._clear_polygon_graphics()
        self._poly_points = []
        for key in list(self._mask_items.keys()):
            item = self._mask_items.pop(key)
            self._scene.removeItem(item)
        if self._mode == MODE_EDIT_POLYGON:
            self._mode = MODE_BOX

    def undo_last_point(self, kind: str) -> None:
        if kind == "positive" and self._state.positive:
            self._state.positive.pop()
        elif kind == "ignore" and self._state.ignore:
            self._state.ignore.pop()
        elif kind == "box":
            self._state.box.reset()
        self._rebuild_prompt_graphics()
        self.promptsMutated.emit()

    def fit_to_view(self) -> None:
        if self._image_item is None:
            return
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        self._emit_zoom()

    def zoom_in(self) -> None:
        self._zoom_by(1.25)

    def zoom_out(self) -> None:
        self._zoom_by(0.8)

    # ─── Polygon edit API ──────────────────────────────────────────────

    def enter_polygon_edit(self, polygon: list[tuple[float, float]]) -> None:
        """Switch to polygon edit mode using a pre-computed polygon.

        4-vertex axis-aligned rectangles (detector boxes, manual
        box→polygon) get the BOX edit treatment: corner handles, no
        edge-click vertex insertion, drag-maintains-rectangle. Anything
        else gets the free polygon edit treatment.
        """
        self._clear_polygon_graphics()
        self._poly_points = [tuple(p) for p in polygon]
        self._edit_as_box = _is_axis_aligned_rect(self._poly_points)
        if self._edit_as_box:
            # Canonicalize the corner order to [nw, ne, se, sw] so the
            # handles' ``corner`` strings line up with their positions.
            self._poly_points = _canonicalize_rect(self._poly_points)
        self._mode = MODE_EDIT_POLYGON
        self._rebuild_polygon_graphics()

    def exit_polygon_edit(self) -> None:
        self._clear_polygon_graphics()
        self._poly_points = []
        if self._mode == MODE_EDIT_POLYGON:
            self._mode = MODE_BOX

    def is_polygon_edit_active(self) -> bool:
        return self._mode == MODE_EDIT_POLYGON and bool(self._poly_points)

    def set_other_polygons(
        self,
        polygons_with_back_idx: list[tuple[int, list[tuple[float, float]]]],
    ) -> None:
        """Register every OTHER polygon-instance of the active tissue
        for click hit-testing. No visual outline is drawn — the union
        mask already paints those areas, and a separate dashed outline
        on top doubles the box visually. The data here exists only so
        the user can click on any non-active polygon to switch the
        edit cursor onto it (``activePolygonRequested`` signal)."""
        self._clear_other_outlines()
        if not polygons_with_back_idx:
            return
        for back_idx, poly_pts in polygons_with_back_idx:
            if len(poly_pts) < 3:
                continue
            self._other_outline_data.append((back_idx, list(poly_pts)))

    def _clear_other_outlines(self) -> None:
        for it in self._other_outline_items:
            self._safe_remove(it)
        self._other_outline_items = []
        self._other_outline_data = []

    def get_polygon(self) -> list[tuple[float, float]]:
        return list(self._poly_points)

    # ─── Events ────────────────────────────────────────────────────────

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() & Qt.ControlModifier:
            factor = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
            self._zoom_by(factor)
            event.accept()
            return
        super().wheelEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if self._mode == MODE_DRAW_POLYGON and len(self._draw_pts) >= 3:
            self._finish_polygon_draw()
            event.accept()
            return
        # Double-click in polygon edit on a polygon area = rename.
        if self._mode == MODE_EDIT_POLYGON and self._image_item is not None:
            scene_pos = self.mapToScene(event.pos())
            back_idx = self._hit_other_polygon(scene_pos)
            if back_idx is None and _point_in_polygon(
                scene_pos.x(), scene_pos.y(), self._poly_points,
            ):
                back_idx = 0
            if back_idx is not None:
                self.polygonRenameRequested.emit(back_idx)
                event.accept()
                return
        # Otherwise: background → fit + reset W/L.
        item = self.itemAt(event.pos())
        if item is None or isinstance(item, QGraphicsPixmapItem):
            self.fit_to_view()
            self.reset_window_level()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_last = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        if self._image_item is None:
            super().mousePressEvent(event)
            return

        # Right-button: stage a pending W/L drag. If the user moves past a
        # threshold we run W/L; if they release within threshold the press
        # is interpreted as an ignore-point click instead.
        if event.button() == Qt.RightButton:
            self._wl_dragging = False
            self._wl_start_pos = event.pos()
            self._wl_start_window = self._window
            self._wl_start_level = self._level
            event.accept()
            return

        scene_pos = self.mapToScene(event.pos())
        item = self.itemAt(event.pos())

        # Manual polygon draw — left click adds vertex; right release pops
        # last (handled in mouseReleaseEvent / _dispatch_right_click).
        if self._mode == MODE_DRAW_POLYGON:
            x, y = scene_pos.x(), scene_pos.y()
            H, W = self._image_hw
            if not (0 <= x < W and 0 <= y < H):
                super().mousePressEvent(event)
                return
            if event.button() == Qt.LeftButton:
                self._draw_pts.append((float(x), float(y)))
                self._refresh_polygon_draw_overlay()
                event.accept()
                return
            super().mousePressEvent(event)
            return

        # Polygon edit mode has its own input semantics.
        if self._mode == MODE_EDIT_POLYGON:
            if isinstance(item, PolygonVertex):
                if event.button() == Qt.LeftButton and (event.modifiers() & Qt.AltModifier):
                    self._delete_polygon_vertex(item.idx)
                    event.accept()
                    return
                super().mousePressEvent(event)
                return
            if isinstance(item, PolyBoxHandle):
                super().mousePressEvent(event)
                return
            if event.button() == Qt.LeftButton:
                target_back_idx = self._hit_other_polygon(scene_pos)
                if target_back_idx is not None:
                    self.activePolygonRequested.emit(target_back_idx)
                    event.accept()
                    return
                # Click on edge of the active polygon → insert vertex
                # (only allowed for free polygon edit, not box edit).
                inserted = self._maybe_insert_polygon_vertex(scene_pos)
                if inserted:
                    event.accept()
                    return
                # Background click while editing — give MainWindow a chance
                # to do something useful (e.g. exit edit + start a new box
                # in Manual Box mode).
                x, y = scene_pos.x(), scene_pos.y()
                H, W = self._image_hw
                if 0 <= x < W and 0 <= y < H:
                    self.backgroundClickedInEdit.emit(float(x), float(y))
                    event.accept()
                    return
            super().mousePressEvent(event)
            return

        # Standard prompt modes
        if isinstance(item, DraggablePoint):
            if event.button() == Qt.LeftButton and (event.modifiers() & Qt.AltModifier):
                self._delete_point(item)
                event.accept()
                return
            super().mousePressEvent(event)
            return
        if isinstance(item, BoxHandle):
            self._box_drag_in_progress = True
            super().mousePressEvent(event)
            return

        x, y = scene_pos.x(), scene_pos.y()
        H, W = self._image_hw
        if not (0 <= x < W and 0 <= y < H):
            super().mousePressEvent(event)
            return

        if self._mode == MODE_BOX and event.button() == Qt.LeftButton:
            self._apply_box_click(x, y)
            event.accept()
            return

        if event.button() == Qt.LeftButton and self._mode == MODE_POINT:
            self._add_positive(x, y)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._panning:
            delta = event.pos() - self._pan_last
            self._pan_last = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return

        # Right-button held: window/level drag once past threshold.
        if event.buttons() & Qt.RightButton and self._base_rgb is not None:
            dx = event.pos().x() - self._wl_start_pos.x()
            dy = event.pos().y() - self._wl_start_pos.y()
            if (
                not self._wl_dragging
                and (abs(dx) + abs(dy)) >= self._wl_drag_threshold_px
            ):
                self._wl_dragging = True
                self.setCursor(Qt.SizeAllCursor)
            if self._wl_dragging:
                # Horizontal = window/contrast, vertical = level/brightness
                # (down increases brightness — matches OsiriX / most viewers).
                self._window = float(self._wl_start_window + dx * 0.6)
                self._level = float(self._wl_start_level - dy * 0.6)
                self._window = max(-127.0, min(127.0, self._window))
                self._level = max(-127.0, min(127.0, self._level))
                self._refresh_image_pixmap()
                self.windowLevelChanged.emit(self._window, self._level)
                event.accept()
                return

        # Live rubber-band preview while drawing a box.
        if (
            self._mode == MODE_BOX
            and self._state.box.first_corner_only()
            and self._image_item is not None
        ):
            sp = self.mapToScene(event.pos())
            self._render_box_preview(self._state.box.x0, self._state.box.y0, sp.x(), sp.y())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MiddleButton and self._panning:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return

        if event.button() == Qt.RightButton:
            if self._wl_dragging:
                self._wl_dragging = False
                self.setCursor(Qt.ArrowCursor)
                event.accept()
                return
            # Released without dragging → it was a right-click; dispatch.
            self._dispatch_right_click(event.pos())
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def _dispatch_right_click(self, view_pos: QPoint) -> None:
        """A right-button click that wasn't a W/L drag.

        - DRAW polygon: pop last vertex.
        - EDIT polygon: if click hit a polygon area, raise the context
          menu signal (Rename / Delete) — otherwise no-op (manual mode
          has no "ignore point" concept).
        - POINT / BOX (SAM-style modes): legacy ignore-point behavior.
        """
        if self._image_item is None:
            return
        scene_pos = self.mapToScene(view_pos)
        x, y = scene_pos.x(), scene_pos.y()
        H, W = self._image_hw
        if not (0 <= x < W and 0 <= y < H):
            return
        if self._mode == MODE_DRAW_POLYGON:
            if self._draw_pts:
                self._draw_pts.pop()
                self._refresh_polygon_draw_overlay()
            return
        if self._mode == MODE_EDIT_POLYGON:
            # Did the click hit the active polygon or one of its siblings?
            back_idx = self._hit_other_polygon(scene_pos)
            if back_idx is None and _point_in_polygon(x, y, self._poly_points):
                back_idx = 0  # active is the most recent (offset 0)
            if back_idx is not None:
                global_pos = self.viewport().mapToGlobal(view_pos)
                self.polygonContextMenuRequested.emit(back_idx, global_pos)
            return
        if self._mode in (MODE_BOX, MODE_POINT):
            # SAM-style behavior — manual mode toggles
            # ``_allow_right_click_ignore`` off so right-click stays
            # reserved for the context menu there.
            if self._allow_right_click_ignore:
                self._add_ignore(x, y)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Escape:
            if self._mode == MODE_DRAW_POLYGON and self._draw_pts:
                self._cancel_polygon_draw()
                event.accept()
                return
            if self._state.box.first_corner_only():
                self._state.box.reset()
                self._clear_first_corner_marker()
                self._clear_box_preview()
                event.accept()
                return
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self._mode == MODE_DRAW_POLYGON and len(self._draw_pts) >= 3:
                self._finish_polygon_draw()
                event.accept()
                return
        super().keyPressEvent(event)

    # ─── Zoom helpers ──────────────────────────────────────────────────

    def _emit_zoom(self) -> None:
        self.zoomChanged.emit(float(self.transform().m11()))

    def _zoom_by(self, factor: float) -> None:
        if self._image_item is None:
            return
        self.scale(factor, factor)
        self._emit_zoom()

    # ─── Box flow ──────────────────────────────────────────────────────

    def _apply_box_click(self, x: float, y: float) -> None:
        b = self._state.box
        if b.finished or b.x0 is None:
            # Starting a fresh box → wipe previous
            self._clear_box_graphics()
            b.reset()
            b.x0, b.y0 = float(x), float(y)
            self._render_first_corner(x, y)
            self._render_box_preview(x, y, x, y)
            return
        # Finalizing
        b.x1, b.y1 = float(x), float(y)
        b.finished = True
        self._clear_first_corner_marker()
        self._clear_box_preview()
        self._render_finalized_box()
        self.promptsMutated.emit()

    def _render_box_preview(self, x0: float, y0: float, x1: float, y1: float) -> None:
        rect = QRectF(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
        if self._box_preview_item is None:
            item = QGraphicsRectItem(rect)
            pen = QPen(QColor(ACCENT))
            pen.setWidthF(2.0)
            pen.setStyle(Qt.DashLine)
            pen.setCosmetic(True)
            item.setPen(pen)
            item.setBrush(QBrush(QColor(4, 116, 237, 35)))
            item.setZValue(Z_BOX)
            self._scene.addItem(item)
            self._box_preview_item = item
        else:
            self._box_preview_item.setRect(rect)

    def _clear_box_preview(self) -> None:
        if self._box_preview_item is not None:
            self._scene.removeItem(self._box_preview_item)
            self._box_preview_item = None

    def _render_first_corner(self, x: float, y: float) -> None:
        self._clear_first_corner_marker()
        path = QPainterPath()
        path.moveTo(x - 14, y); path.lineTo(x + 14, y)
        path.moveTo(x, y - 14); path.lineTo(x, y + 14)
        item = QGraphicsPathItem(path)
        pen = QPen(QColor(ACCENT))
        pen.setWidthF(2.0)
        pen.setCosmetic(True)
        item.setPen(pen)
        item.setZValue(Z_BOX)
        self._scene.addItem(item)
        self._first_corner_marker = item

    def _clear_first_corner_marker(self) -> None:
        if self._first_corner_marker is not None:
            self._scene.removeItem(self._first_corner_marker)
            self._first_corner_marker = None

    def _render_finalized_box(self) -> None:
        # Clear previous graphics
        if self._box_rect_item is not None:
            self._scene.removeItem(self._box_rect_item)
            self._box_rect_item = None
        for h in self._box_handles:
            self._scene.removeItem(h)
        self._box_handles = []

        xyxy = self._state.box.to_xyxy()
        if xyxy is None:
            return
        x0, y0, x1, y1 = xyxy.tolist()
        rect = QRectF(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
        rect_item = QGraphicsRectItem(rect)
        pen = QPen(QColor(ACCENT))
        pen.setWidthF(2.0)
        pen.setStyle(Qt.DashLine)
        pen.setCosmetic(True)
        rect_item.setPen(pen)
        rect_item.setBrush(QBrush(QColor(4, 116, 237, 18)))
        rect_item.setZValue(Z_BOX)
        self._scene.addItem(rect_item)
        self._box_rect_item = rect_item

        for corner, (cx, cy) in (
            ("nw", (rect.left(), rect.top())),
            ("ne", (rect.right(), rect.top())),
            ("sw", (rect.left(), rect.bottom())),
            ("se", (rect.right(), rect.bottom())),
        ):
            h = BoxHandle(corner)
            self._scene.addItem(h)
            h.setPos(QPointF(cx, cy))
            self._box_handles.append(h)

    def _on_box_handle_dragging(self, handle: BoxHandle, new_pos: QPointF) -> QPointF:
        """Update box rect live during corner drag. Returns the (clamped)
        position the handle should snap to. Peers are repositioned with
        their re-entry guard set so we don't recurse."""
        if self._box_rect_item is None or not self._box_handles:
            return new_pos
        H, W = self._image_hw
        nx = max(0.0, min(float(W - 1), new_pos.x()))
        ny = max(0.0, min(float(H - 1), new_pos.y()))

        opp = _opposite_corner(handle.corner)
        opp_handle = next((h for h in self._box_handles if h.corner == opp), None)
        if opp_handle is None:
            return QPointF(nx, ny)
        ox, oy = opp_handle.pos().x(), opp_handle.pos().y()

        x0, x1 = sorted([nx, ox])
        y0, y1 = sorted([ny, oy])
        self._box_rect_item.setRect(QRectF(x0, y0, x1 - x0, y1 - y0))

        self._state.box.x0, self._state.box.y0 = float(x0), float(y0)
        self._state.box.x1, self._state.box.y1 = float(x1), float(y1)
        self._state.box.finished = True

        corner_pos = {
            "nw": QPointF(x0, y0),
            "ne": QPointF(x1, y0),
            "sw": QPointF(x0, y1),
            "se": QPointF(x1, y1),
        }
        for h in self._box_handles:
            if h is handle:
                continue
            target = corner_pos[h.corner]
            h._suppress_drag_callback = True
            try:
                h.setPos(target)
            finally:
                h._suppress_drag_callback = False
        return QPointF(nx, ny)

    def _on_box_handle_released(self, handle: BoxHandle) -> None:
        self._box_drag_in_progress = False
        self.promptsMutated.emit()

    def _clear_box_graphics(self) -> None:
        if self._box_rect_item is not None:
            self._scene.removeItem(self._box_rect_item)
            self._box_rect_item = None
        for h in self._box_handles:
            self._scene.removeItem(h)
        self._box_handles = []
        self._clear_box_preview()
        self._clear_first_corner_marker()

    # ─── Point flow ────────────────────────────────────────────────────

    def _add_positive(self, x: float, y: float) -> None:
        self._state.positive.append((float(x), float(y)))
        self._spawn_point("positive", x, y)
        self.promptsMutated.emit()

    def _add_ignore(self, x: float, y: float) -> None:
        self._state.ignore.append((float(x), float(y)))
        self._spawn_point("ignore", x, y)
        self._spawn_ignore_disk(x, y)
        self.promptsMutated.emit()

    def _delete_point(self, item: DraggablePoint) -> None:
        pos = item.pos()
        coll = self._state.positive if item.kind == "positive" else self._state.ignore
        idx = _nearest_idx(coll, (pos.x(), pos.y()), tol=POINT_HIT_TOL)
        if idx is not None:
            coll.pop(idx)
        self._scene.removeItem(item)
        self._point_items = [p for p in self._point_items if p is not item]
        if item.kind == "ignore":
            self._refresh_ignore_disks()
        self.promptsMutated.emit()

    def _on_point_released(self, item: DraggablePoint) -> None:
        new_pos = item.pos()
        coll = self._state.positive if item.kind == "positive" else self._state.ignore
        # Locate this item's index within its kind
        kind_idx = 0
        for it in self._point_items:
            if it is item:
                break
            if it.kind == item.kind:
                kind_idx += 1
        if 0 <= kind_idx < len(coll):
            coll[kind_idx] = (float(new_pos.x()), float(new_pos.y()))
        if item.kind == "ignore":
            self._refresh_ignore_disks()
        self.promptsMutated.emit()

    def _spawn_point(self, kind: str, x: float, y: float) -> DraggablePoint:
        color = _point_color(self._active_tissue, kind)
        item = DraggablePoint(kind, radius_px=6.0, color=color)
        self._scene.addItem(item)
        item.setPos(QPointF(x, y))
        self._point_items.append(item)
        return item

    def _spawn_ignore_disk(self, x: float, y: float) -> QGraphicsEllipseItem:
        r = self._ignore_radius_px
        disk = QGraphicsEllipseItem(QRectF(x - r, y - r, 2 * r, 2 * r))
        pen = QPen(QColor(ALERT))
        pen.setWidth(0)
        pen.setCosmetic(True)
        disk.setPen(pen)
        disk.setBrush(QBrush(QColor(228, 116, 116, 38)))
        disk.setZValue(Z_IGNORE_DISK)
        self._scene.addItem(disk)
        self._ignore_disk_items.append(disk)
        return disk

    def _refresh_ignore_disks(self) -> None:
        for d in self._ignore_disk_items:
            self._scene.removeItem(d)
        self._ignore_disk_items.clear()
        for x, y in self._state.ignore:
            self._spawn_ignore_disk(x, y)

    def _rebuild_prompt_graphics(self) -> None:
        for it in self._point_items:
            self._scene.removeItem(it)
        self._point_items.clear()
        for d in self._ignore_disk_items:
            self._scene.removeItem(d)
        self._ignore_disk_items.clear()
        self._clear_box_graphics()

        for x, y in self._state.positive:
            self._spawn_point("positive", x, y)
        for x, y in self._state.ignore:
            self._spawn_point("ignore", x, y)
            self._spawn_ignore_disk(x, y)
        if self._state.box.finished:
            self._render_finalized_box()
        elif self._state.box.first_corner_only():
            self._render_first_corner(self._state.box.x0, self._state.box.y0)

    # ─── Polygon edit flow ─────────────────────────────────────────────

    def _rebuild_polygon_graphics(self) -> None:
        # Keep the existing handles/vertices around — only the items we
        # own. Don't touch the class label here; we re-render it at the end.
        for v in self._poly_vertices:
            self._scene.removeItem(v)
        self._poly_vertices = []
        for h in self._polybox_handles:
            self._scene.removeItem(h)
        self._polybox_handles = []
        if self._poly_path_item is not None:
            self._scene.removeItem(self._poly_path_item)
            self._poly_path_item = None
        if self._poly_fill_item is not None:
            self._scene.removeItem(self._poly_fill_item)
            self._poly_fill_item = None
        if self._poly_label_item is not None:
            self._scene.removeItem(self._poly_label_item)
            self._poly_label_item = None
        if len(self._poly_points) < 3:
            return
        color = QColor(*TISSUE_RGB.get(self._active_tissue, (255, 200, 0)))

        poly = QPolygonF([QPointF(x, y) for x, y in self._poly_points])
        path = QPainterPath()
        path.addPolygon(poly)
        path.closeSubpath()

        # Slim fill at the slider's opacity. The MainWindow excludes this
        # polygon from the union mask while editing, so this is the only
        # source of color in the polygon's area (no "ghost" left behind).
        fill_item = QGraphicsPolygonItem(poly)
        fill_item.setPen(QPen(Qt.NoPen))
        fill_item.setBrush(QBrush(QColor(
            color.red(), color.green(), color.blue(), self._mask_alpha,
        )))
        fill_item.setZValue(Z_POLYGON_EDGE - 1)
        self._scene.addItem(fill_item)
        self._poly_fill_item = fill_item

        # Bright cosmetic outline.
        edge = QGraphicsPathItem(path)
        edge_pen = QPen(QColor(255, 255, 255, 235))
        edge_pen.setWidthF(2.4)
        edge_pen.setCosmetic(True)
        edge.setPen(edge_pen)
        edge.setBrush(QBrush(Qt.NoBrush))
        edge.setZValue(Z_POLYGON_EDGE)
        self._scene.addItem(edge)
        self._poly_path_item = edge

        if self._edit_as_box:
            # Place 4 corner handles in nw / ne / se / sw order. The
            # underlying _poly_points were canonicalized to this order
            # already in ``enter_polygon_edit``.
            corners = ("nw", "ne", "se", "sw")
            for i, (x, y) in enumerate(self._poly_points):
                h = PolyBoxHandle(corners[i])
                self._scene.addItem(h)
                h.setPos(QPointF(x, y))
                self._polybox_handles.append(h)
        else:
            for i, (x, y) in enumerate(self._poly_points):
                v = PolygonVertex(i, color)
                self._scene.addItem(v)
                v.setPos(QPointF(x, y))
                self._poly_vertices.append(v)

        self._render_class_label()

    def _render_class_label(self) -> None:
        """Pin a small class-name pill at the polygon's top-left corner.
        Stays a constant on-screen size regardless of zoom."""
        if not self._poly_points or self._active_tissue not in TISSUE_PRESETS:
            return
        preset = TISSUE_PRESETS[self._active_tissue]
        r, g, b = TISSUE_RGB.get(self._active_tissue, (200, 200, 200))
        tl_x = min(x for x, _ in self._poly_points)
        tl_y = min(y for _, y in self._poly_points)
        label = QGraphicsTextItem()
        label.setHtml(
            f'<div style="background-color: rgba({r},{g},{b},230); '
            f'color: white; padding: 1px 8px; border-radius: 4px; '
            f'font-size: 11px; font-weight: 700; letter-spacing: 0.4px;">'
            f'{preset.label}</div>'
        )
        label.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        label.setZValue(Z_POINT + 1)
        label.setPos(QPointF(tl_x, tl_y - 22))
        self._scene.addItem(label)
        self._poly_label_item = label

    def _update_class_label_position(self) -> None:
        if self._poly_label_item is None or not self._poly_points:
            return
        tl_x = min(x for x, _ in self._poly_points)
        tl_y = min(y for _, y in self._poly_points)
        self._poly_label_item.setPos(QPointF(tl_x, tl_y - 22))

    def _clear_polygon_graphics(self) -> None:
        for v in self._poly_vertices:
            self._safe_remove(v)
        self._poly_vertices = []
        for h in self._polybox_handles:
            self._safe_remove(h)
        self._polybox_handles = []
        self._safe_remove(self._poly_path_item)
        self._poly_path_item = None
        self._safe_remove(self._poly_fill_item)
        self._poly_fill_item = None
        self._safe_remove(self._poly_label_item)
        self._poly_label_item = None
        self._edit_as_box = False

    def _on_polygon_vertex_dragging(self, vertex: PolygonVertex, new_pos: QPointF) -> None:
        if self._suppress_polygon_signals:
            return
        if 0 <= vertex.idx < len(self._poly_points):
            self._poly_points[vertex.idx] = (float(new_pos.x()), float(new_pos.y()))
        self._refresh_polygon_outline()

    def _on_polygon_vertex_released(self, vertex: PolygonVertex) -> None:
        self.polygonMutated.emit(self._active_tissue)

    def _refresh_polygon_outline(self) -> None:
        if not self._poly_points or len(self._poly_points) < 3:
            return
        poly = QPolygonF([QPointF(x, y) for x, y in self._poly_points])
        if self._poly_fill_item is not None:
            self._poly_fill_item.setPolygon(poly)
        if self._poly_path_item is not None:
            path = QPainterPath()
            path.addPolygon(poly)
            path.closeSubpath()
            self._poly_path_item.setPath(path)
        self._update_class_label_position()

    # ── Box-style polygon edit (4-vertex rect) ────────────────────────

    def _on_polybox_corner_dragging(self, handle: PolyBoxHandle, new_pos: QPointF):
        if not self._polybox_handles or len(self._poly_points) != 4:
            return new_pos
        H, W = self._image_hw
        nx = max(0.0, min(float(W - 1), new_pos.x()))
        ny = max(0.0, min(float(H - 1), new_pos.y()))
        opp = _opposite_corner(handle.corner)
        opp_handle = next((h for h in self._polybox_handles if h.corner == opp), None)
        if opp_handle is None:
            return QPointF(nx, ny)
        ox, oy = opp_handle.pos().x(), opp_handle.pos().y()
        x0, x1 = sorted([nx, ox])
        y0, y1 = sorted([ny, oy])
        # Rebuild the polygon in canonical [nw, ne, se, sw] order.
        self._poly_points = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        # Live-update outline/fill/label position.
        self._refresh_polygon_outline()
        # Snap the OTHER two handles to the new rect.
        corner_pos = {
            "nw": QPointF(x0, y0),
            "ne": QPointF(x1, y0),
            "sw": QPointF(x0, y1),
            "se": QPointF(x1, y1),
        }
        for h in self._polybox_handles:
            if h is handle:
                continue
            h._suppress_drag_callback = True
            try:
                h.setPos(corner_pos[h.corner])
            finally:
                h._suppress_drag_callback = False
        return QPointF(nx, ny)

    def _on_polybox_corner_released(self, handle: PolyBoxHandle) -> None:
        self.polygonMutated.emit(self._active_tissue)

    def _maybe_insert_polygon_vertex(self, scene_pos: QPointF) -> bool:
        # Box-style edit keeps the polygon as a rectangle — never insert
        # a 5th vertex from an edge click.
        if self._edit_as_box:
            return False
        if len(self._poly_points) < 2:
            return False
        target = (scene_pos.x(), scene_pos.y())
        best = _nearest_edge(self._poly_points, target, tol=EDGE_HIT_TOL * 2)
        if best is None:
            return False
        seg_idx, proj_x, proj_y = best
        insert_at = seg_idx + 1
        self._poly_points.insert(insert_at, (proj_x, proj_y))
        self._rebuild_polygon_graphics()
        self.polygonMutated.emit(self._active_tissue)
        return True

    def _hit_other_polygon(self, scene_pos: QPointF) -> Optional[int]:
        """Return the ``back_idx`` of the sibling polygon containing the
        point (or None). Iterates youngest → oldest so the topmost match
        wins, matching what the user perceives visually."""
        if not self._other_outline_data:
            return None
        px, py = scene_pos.x(), scene_pos.y()
        for back_idx, poly in self._other_outline_data:
            if _point_in_polygon(px, py, poly):
                return back_idx
        return None

    def _delete_polygon_vertex(self, idx: int) -> None:
        if not (0 <= idx < len(self._poly_points)):
            return
        if len(self._poly_points) <= 3:
            return  # keep a valid polygon
        self._poly_points.pop(idx)
        self._rebuild_polygon_graphics()
        self.polygonMutated.emit(self._active_tissue)

    # ─── Manual polygon draw flow ──────────────────────────────────────

    def start_polygon_draw(self) -> None:
        """Switch into manual polygon-draw mode for the active tissue."""
        self._cancel_polygon_draw()
        if self._mode == MODE_EDIT_POLYGON:
            self._clear_polygon_graphics()
            self._poly_points = []
        self._mode = MODE_DRAW_POLYGON

    def _refresh_polygon_draw_overlay(self) -> None:
        # Clear previous
        if self._draw_path_item is not None:
            self._scene.removeItem(self._draw_path_item)
            self._draw_path_item = None
        if self._poly_fill_item is not None and self._mode == MODE_DRAW_POLYGON:
            self._scene.removeItem(self._poly_fill_item)
            self._poly_fill_item = None
        for d in self._draw_dot_items:
            self._scene.removeItem(d)
        self._draw_dot_items = []

        if not self._draw_pts:
            return

        color = QColor(*TISSUE_RGB.get(self._active_tissue, (255, 200, 0)))

        # When we have 3+ points, paint a translucent fill so the user sees
        # the enclosed area like a real segmentation in progress.
        if len(self._draw_pts) >= 3:
            poly = QPolygonF([QPointF(x, y) for x, y in self._draw_pts])
            fill = QGraphicsPolygonItem(poly)
            fill.setPen(QPen(Qt.NoPen))
            fill.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 80)))
            fill.setZValue(Z_MASK)
            self._scene.addItem(fill)
            self._poly_fill_item = fill

        # Path connecting consecutive points (not closed yet).
        path = QPainterPath()
        path.moveTo(*self._draw_pts[0])
        for x, y in self._draw_pts[1:]:
            path.lineTo(x, y)
        item = QGraphicsPathItem(path)
        pen = QPen(QColor(color))
        pen.setWidthF(2.0)
        pen.setStyle(Qt.DashLine)
        pen.setCosmetic(True)
        item.setPen(pen)
        item.setBrush(QBrush(Qt.NoBrush))
        item.setZValue(Z_POLYGON_EDGE)
        self._scene.addItem(item)
        self._draw_path_item = item

        # Vertex dots
        for x, y in self._draw_pts:
            dot = QGraphicsEllipseItem(QRectF(-4, -4, 8, 8))
            dot.setBrush(QBrush(QColor(TEXT_PRIMARY)))
            pen2 = QPen(color)
            pen2.setWidthF(2.0)
            pen2.setCosmetic(True)
            dot.setPen(pen2)
            dot.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
            dot.setPos(QPointF(x, y))
            dot.setZValue(Z_POINT)
            self._scene.addItem(dot)
            self._draw_dot_items.append(dot)

    def _cancel_polygon_draw(self) -> None:
        self._draw_pts = []
        self._safe_remove(self._draw_path_item)
        self._draw_path_item = None
        if self._mode == MODE_DRAW_POLYGON:
            self._safe_remove(self._poly_fill_item)
            self._poly_fill_item = None
        for d in self._draw_dot_items:
            self._safe_remove(d)
        self._draw_dot_items = []

    def undo_last_draw_vertex(self) -> None:
        """Pop the last vertex from the in-progress polygon draw, or trim
        the last vertex from a polygon being edited."""
        if self._mode == MODE_DRAW_POLYGON and self._draw_pts:
            self._draw_pts.pop()
            self._refresh_polygon_draw_overlay()
            return
        if self._mode == MODE_EDIT_POLYGON and len(self._poly_points) > 3:
            self._poly_points.pop()
            self._rebuild_polygon_graphics()
            self.polygonMutated.emit(self._active_tissue)

    def _finish_polygon_draw(self) -> None:
        if len(self._draw_pts) < 3:
            return
        closed = list(self._draw_pts)
        self._cancel_polygon_draw()
        # Hand the closed polygon to the editor — user can immediately drag
        # vertices to refine.
        self.enter_polygon_edit(closed)
        # Tell the controller a draw was just completed so it can rasterize
        # the polygon to a mask + persist it.
        self.polygonDrawCompleted.emit(self._active_tissue)
        # Also fire polygonMutated so main_window does its standard storage.
        self.polygonMutated.emit(self._active_tissue)


# ─── helpers ─────────────────────────────────────────────────────────────


def _rgb_to_pixmap(rgb: np.ndarray) -> QPixmap:
    """Defensive RGB-to-QPixmap. Squeezes extra dims (e.g. multi-frame
    arrays that slipped past the loader), promotes grayscale to 3
    channels, drops alpha — anything to avoid bringing the UI down."""
    arr = np.asarray(rgb)
    while arr.ndim > 3:
        arr = arr[0]
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3:
        c = arr.shape[2]
        if c == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif c == 4:
            arr = arr[:, :, :3]
        elif c != 3:
            arr = arr[:, :, :3] if c > 3 else np.stack([arr[..., 0]] * 3, axis=-1)
    else:
        raise ValueError(f"unsupported array shape for pixmap: {rgb.shape}")
    arr = np.ascontiguousarray(arr.astype(np.uint8))
    h, w, _ = arr.shape
    img = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(img)


def _mask_to_pixmap(mask: np.ndarray, color_rgb: tuple[int, int, int], alpha: int = 110) -> QPixmap:
    h, w = mask.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    m = mask > 0
    rgba[m, 0] = color_rgb[0]
    rgba[m, 1] = color_rgb[1]
    rgba[m, 2] = color_rgb[2]
    rgba[m, 3] = alpha
    rgba = np.ascontiguousarray(rgba)
    img = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
    return QPixmap.fromImage(img)


def _point_color(tissue_key: str, kind: str) -> QColor:
    if kind == "ignore":
        return QColor(ALERT)
    r, g, b = TISSUE_RGB.get(tissue_key, (200, 200, 200))
    return QColor(r, g, b)


def _nearest_idx(coll: list[tuple[float, float]], target: tuple[float, float], tol: float) -> Optional[int]:
    best_idx = None
    best_d = tol * tol
    tx, ty = target
    for i, (x, y) in enumerate(coll):
        d = (x - tx) ** 2 + (y - ty) ** 2
        if d < best_d:
            best_d = d
            best_idx = i
    return best_idx


def _nearest_edge(
    poly: list[tuple[float, float]],
    target: tuple[float, float],
    tol: float,
) -> Optional[tuple[int, float, float]]:
    """Return (segment_index, projected_x, projected_y) of the closest edge
    within ``tol`` pixels of ``target``, or None."""
    n = len(poly)
    if n < 2:
        return None
    best = None
    best_d = tol * tol
    tx, ty = target
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        dx, dy = bx - ax, by - ay
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq <= 1e-9:
            continue
        t = ((tx - ax) * dx + (ty - ay) * dy) / seg_len_sq
        t = max(0.0, min(1.0, t))
        px, py = ax + t * dx, ay + t * dy
        d = (px - tx) ** 2 + (py - ty) ** 2
        if d < best_d:
            best_d = d
            best = (i, px, py)
    return best


def _is_axis_aligned_rect(poly: list[tuple[float, float]], tol: float = 0.5) -> bool:
    """A 4-vertex polygon counts as a rectangle when its corners only
    use two distinct x-values and two distinct y-values (within ``tol``).
    Detector boxes and the Manual box→polygon conversion both satisfy
    this; SAM-mask-derived polygons and free-drawn polygons usually do
    not (more vertices, or odd angles)."""
    if len(poly) != 4:
        return False
    xs = sorted(round(x / tol) for x, _ in poly)
    ys = sorted(round(y / tol) for _, y in poly)
    return len(set(xs)) == 2 and len(set(ys)) == 2


def _canonicalize_rect(poly: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Reorder a 4-vertex rectangle's vertices to ``[nw, ne, se, sw]`` so
    each ``PolyBoxHandle.corner`` string matches its actual position."""
    xs = sorted({round(x, 1) for x, _ in poly})
    ys = sorted({round(y, 1) for _, y in poly})
    if len(xs) != 2 or len(ys) != 2:
        return list(poly)
    x0, x1 = xs
    y0, y1 = ys
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _point_in_polygon(px: float, py: float, poly: list[tuple[float, float]]) -> bool:
    """Standard ray-cast point-in-polygon. Reuses the same heuristic Qt
    uses internally for ``QPolygonF.containsPoint`` — but we avoid the
    cost of building a fresh ``QPolygonF`` for every hit-test by working
    on the raw coordinates we already cache."""
    n = len(poly)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and (
            px < (xj - xi) * (py - yi) / ((yj - yi) or 1e-9) + xi
        ):
            inside = not inside
        j = i
    return inside


def _opposite_corner(corner: str) -> str:
    return {"nw": "se", "ne": "sw", "sw": "ne", "se": "nw"}[corner]


def _handle_cursor(corner: str) -> Qt.CursorShape:
    return {
        "nw": Qt.SizeFDiagCursor,
        "se": Qt.SizeFDiagCursor,
        "ne": Qt.SizeBDiagCursor,
        "sw": Qt.SizeBDiagCursor,
    }.get(corner, Qt.SizeAllCursor)


# ─── Mask <-> polygon conversion (used by main_window) ──────────────────


def mask_to_polygon(mask: np.ndarray, epsilon: float = 2.0) -> list[tuple[float, float]]:
    """Largest-contour polygon for the given binary mask. Empty list if no
    valid contour. Coordinates are in (x, y) pixel space."""
    if mask is None or not mask.any():
        return []
    m = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    biggest = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(biggest, epsilon=max(epsilon, 0.5), closed=True).reshape(-1, 2)
    if approx.shape[0] < 3:
        return []
    return [(float(p[0]), float(p[1])) for p in approx]


def polygon_to_mask(polygon: list[tuple[float, float]], image_hw: tuple[int, int]) -> np.ndarray:
    """Rasterize a polygon to a binary mask."""
    h, w = image_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(polygon) < 3:
        return mask
    pts = np.array([[int(round(x)), int(round(y))] for x, y in polygon], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask

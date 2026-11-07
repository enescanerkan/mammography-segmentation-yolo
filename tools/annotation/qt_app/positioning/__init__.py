"""Breast Positioning tool (MLO / CC landmark-based positioning assessment).

YOLO-pose models predict the nipple (both views) and two pectoral-line points (MLO);
the tool compares the MLO nipple-to-pectoral distance with the CC nipple-to-chest-wall
distance against a millimetre threshold. Launched in-process by
``qt_app.positioning_launcher``.
"""

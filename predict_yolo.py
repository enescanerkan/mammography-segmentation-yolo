"""
YOLO11 Segmentation Predict Scripti
Test görüntüleri üzerinde tahmin yapar ve sonuçları kaydeder.
"""
from ultralytics import YOLO

WEIGHTS   = r"C:\Users\Monster\Desktop\segment_breast\runs\breast_seg_yolo11\weights\best.pt"
TEST_DIR  = r"C:\Users\Monster\Desktop\segment_breast\images\test"
SAVE_DIR  = r"C:\Users\Monster\Desktop\segment_breast\predictions"

def main():
    model = YOLO(WEIGHTS)

    results = model.predict(
        source=TEST_DIR,
        imgsz=640,
        conf=0.25,           # Minimum güven eşiği
        iou=0.5,
        device=0,
        save=True,           # Görüntüleri kaydet
        save_txt=True,       # Label txt kaydet
        save_conf=True,      # Confidence bilgisi
        project=r"C:\Users\Monster\Desktop\segment_breast",
        name="predictions",
        exist_ok=True,
        retina_masks=True,   # Yüksek kaliteli maskeler
        show_boxes=False,    # Bounding box gizle
        show_labels=True,
        show_conf=True,
        line_width=2,
    )

    print(f"\nTahmin tamamlandi!")
    print(f"Sonuclar: C:\\Users\\Monster\\Desktop\\segment_breast\\predictions")

if __name__ == "__main__":
    main()

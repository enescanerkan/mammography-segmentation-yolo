# Mammography Segmentation & Positioning — YOLO Pipeline

YOLO segmentasyon ve YOLO pose (rule-based) modellerini kullanarak mamografi görüntülerinde **PNL** (Posterior Nipple Line) ve **CC Depth** mesafelerini hesaplar, **10mm kuralına** göre karşılaştırır ve klinik metrikler üretir.

## Proje Yapısı

```
mammography-segmentation-yolo/
├── start.py                  # Ana giriş noktası (CLI)
├── run_train.py              # YOLO segmentasyon eğitimi
├── build_dataset.py          # Veri seti birleştirme
├── setup_dataset.py          # Train/val/test split
├── .gitignore
├── README.md
│
├── pipeline/                 # OOP karşılaştırma hattı
│   ├── orchestrator.py       # Tüm modülleri birleştiren ana orkestratör
│   ├── models.py             # Model yükleme, Google Drive indirme, Factory
│   ├── dataset.py            # CSV okuma, MLO+CC test çifti eşleştirme
│   ├── evaluator.py          # Inference çalıştırma, metrik hesaplama
│   ├── geometry.py           # DICOM↔640 koordinat dönüşümü, PNL/Depth
│   ├── dicom_utils.py        # DICOM yükleme, VOI LUT, BGR dönüşümü
│   └── visualizer.py         # DICOM grid çizimi, maske overlay
│
├── breast_seg/               # Segmentasyon analiz kütüphanesi
│   ├── analyzer.py           # MLOAnalyzer — maske çıkarma, geometri
│   ├── config.py             # Seg model ayarları
│   ├── geometry.py           # Pectoral line fitting, nipple centroid
│   ├── model.py              # YOLO seg wrapper
│   └── visualizer.py         # Analiz görselleştirme
│
├── compare-dataset/          # (gitignore) Test verileri
│   ├── labels/               # cc_labels.csv, mlo_labels.csv
│   ├── CC/                   # CC PNG'ler + metadata.csv
│   ├── MLO/                  # MLO PNG'ler + metadata.csv
│   └── test_dicom/           # DICOM dosyaları ({study_uid}/{sop}.dicom)
│
├── pose_weights/             # (gitignore) Rule-based pose modelleri
├── runs/                     # (gitignore) Eğitim çıktıları (best.pt)
└── compare_output/           # (gitignore) Karşılaştırma çıktıları
    ├── metrics_clinical.csv
    ├── classification_metrics.csv
    ├── confusion_matrix_pose.png
    ├── confusion_matrix_seg.png
    └── viz_dicom/            # DICOM görselleştirme PNG'leri
```

## Kurulum

```bash
pip install ultralytics opencv-python pandas pydicom scikit-learn seaborn matplotlib gdown tqdm
```

## Kullanım

### Karşılaştırma Pipeline'ını Çalıştırma

```bash
# Tüm test setini çalıştır
python start.py compare

# İlk N çiftle hızlı test
python start.py compare --limit 5

# Görselleştirme olmadan sadece metrikler
python start.py compare --no-dicom-viz
```

### Segmentasyon Eğitimi

```bash
python start.py train
```

## Klinik Metrikler

| Metrik | Açıklama |
|--------|----------|
| **PNL** | MLO görüntüsünde nipple'dan pectoral çizgiye dik mesafe (mm) |
| **Depth** | CC görüntüsünde nipple'dan medial kenara (göğüs duvarı) mesafe (mm) |
| **10mm Kuralı** | \|PNL − Depth\| ≤ 10mm ise **Good**, değilse **Bad** |

## Çıktılar

- `metrics_clinical.csv` — Tüm vakaların PNL, Depth, Error değerleri
- `classification_metrics.csv` — Accuracy, Precision, Recall, F1 tablosu
- `confusion_matrix_pose.png` / `confusion_matrix_seg.png` — Karşılaştırma matrisleri
- `viz_dicom/*.png` — GT / Rule-Based / Seg-Model karşılaştırma görselleri

## Pose Model İndirme

Pose modelleri `pose_weights/` dizinine konulmalı:
- **MLO**: [Google Drive](https://drive.google.com/drive/folders/1V9j-Hm4j64lh2doTpoj4u07-F-vKNrUJ)
- **CC**: [Google Drive](https://drive.google.com/drive/folders/11p_uYnbdJmnIjHbNgKMgkEe7mtsdyVQE)

> `gdown` kuruluysa otomatik indirilir. Aksi halde modelleri indirip `pose_weights/` altına koyun.

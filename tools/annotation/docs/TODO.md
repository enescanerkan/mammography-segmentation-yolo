# MedSAM2 — İyileştirme TODO listesi

Bu liste, fine-tune + interactive pipeline'ı iyileştirmek için önerilen
işlerin **önceliklendirilmiş, doğrulanabilir** kaydıdır. Her madde için:

- **Doğrulama durumu**: iddianın kanıtlandığı mı, varsayım mı?
- **Etki**: gerçekleşirse beklenen kazanç (rakam veya niteliksel).
- **Risk**: yan etki / regresyon ihtimali.
- **Süre**: tahmini iş + (varsa) retrain süresi.
- **Kabul kriteri**: "iyileşti" değil; **ölçülebilir** bitiş koşulu.

> Sıralama prensibi: **önce doğrulama, sonra kod, sonra retrain**. Kanıtlanmamış
> varsayım üzerinden 6-12 saatlik retrain'e gitmek kaçınılması gereken hata.

İşaretleme:
- ⬜ yapılmadı
- 🟡 sürüyor / kısmen yapıldı
- ✅ tamamlandı
- ⏸ ertelendi (gerekçesi notta)

---

## FAZ A — Doğrulama ve hızlı kazançlar (kod sıfır risk)

### A1 ⬜ Polygon overlap doğrulama scripti

- **Doğrulama durumu**: 🔴 **TEMELİ DOĞRULAMAK ŞART**. Önerinin tüm
  argümanı "radyolog pectoral ve breast'i overlap çiziyor" varsayımına
  dayanıyor. Bu yapılmadan retrain'e gitme.
- **Yapılacak iş**: `medsam2_experiments/tools/verify_polygon_overlap.py`
  yaz. Her `seg-dataset/labels/{split}/*.txt` için:
  - YOLO polygon'larını parse et (`export_seg_dataset_to_raw_mammo.parse_yolo_seg_lines` mevcut)
  - Her sınıf için ayrı boş maskede polygon'ları çiz (`fillPoly` her sınıfa)
  - `overlap_pectoral_breast = pectoral_mask AND breast_mask`
  - `iou_overlap = |overlap| / |pectoral_union_breast|`
  - View (CC/MLO) bilgisi varsa `view_map.csv`'den oku
- **Çıktı**: CSV with columns `case_id, view, pectoral_px, breast_px, overlap_px, overlap_ratio_to_pectoral`.
- **Süre**: 15 dk
- **Kabul kriteri**:
  - Script çalışır, tüm split'leri tarar.
  - Özet basar: "MLO içinde overlap > %5 olan case oranı = %N".
  - N > 30% ise A4 (per-class binary mask) onaylanır.
  - N < 5% ise A4 reddedilir (overlap mythical, mevcut export doğru).

---

### A2 ⬜ MedSAM2_latest.pt uyumluluk testi

- **Doğrulama durumu**: README'de "may not load into sam2_hiera_t.yaml on
  every checkout" uyarısı var. Bilinmiyor.
- **Yapılacak iş**:
  1. `checkpoints/MedSAM2_latest.pt` var mı kontrol et (yoksa README §2 linkinden indir).
  2. Çalıştır:
     ```powershell
     $env:MEDSAM2_ZERO_SHOT_CKPT = ".\checkpoints\MedSAM2_latest.pt"
     python zero_shot_test.py --limit 10 --save-vis
     ```
  3. `build_sam2` patlamazsa ortalama Dice'ı `data/zero_shot_output/summary.txt`'den oku.
- **Süre**: 5 dk + 5-10 dk zero-shot run
- **Kabul kriteri**:
  - LOAD BAŞARILI ise: Dice (MedSAM2) vs Dice (SAM2 base) tablosu. Eğer
    MedSAM2 > SAM2 base, retrain için MedSAM2'den başla (A5).
  - LOAD PATLARSA: hata logla, A5 listeden düş, SAM2 Hiera Tiny base'de kal.

---

### A3 ⬜ `id()` cache → hash cache (defensive fix)

- **Doğrulama durumu**: Bug **teorik olarak gerçek** (Python id reuse).
  Pratik tetiklenme nadir (Gradio gr.State numpy array'i tutuyor). Yine de
  küçük fix, sıfır risk.
- **Yapılacak iş**: `medsam2_experiments/interactive/model.py` →
  `MedSAM2Service.set_image`:
  ```python
  import hashlib
  ident = hashlib.sha1(rgb.tobytes()[:4096]).hexdigest()
  if ident == self._image_cached_hash:
      return
  ```
  `_image_cached_id` → `_image_cached_hash` rename.
- **Süre**: 10 dk
- **Kabul kriteri**:
  - Aynı görüntü iki kez yüklendiğinde `set_image` ikinci sefer no-op
    (mevcut davranış korunur).
  - Farklı iki görüntünün byte içeriği ilk 4KB farklıysa cache miss olur.
  - Lint + py_compile temiz.

---

## FAZ B — Eğitim verisi / hyperparam (A1 sonucuna bağlı)

### B1 ⬜ Per-class binary mask export (A1 onaylarsa)

- **Doğrulama durumu**: A1 olmadan başlama.
- **Yapılacak iş**: `export_seg_dataset_to_raw_mammo.py` revize:
  - `fill_mask_from_yolo` yerine `fill_per_class_binary_masks`.
  - Her sınıf için bağımsız binary PNG kaydet:
    `masks_cls1_pectoral/{stem}.png`, `masks_cls2_breast/{stem}.png`,
    `masks_cls3_nipple/{stem}.png` (mevcut `masks/{stem}.png` multi-class
    versiyonu **yan yana** kalsın, geriye dönük uyumluluk için).
  - `prepare_medsam2_data.py` revize: binary mask'ı doğrudan oku
    (`cv2.imread(masks_cls{N}/...).astype(np.uint8)`), `mask_full == cls_id`
    yapımına gerek kalmaz.
- **Süre**: 30 dk fix + 5 dk re-export + **6-12 sa retrain**
- **Kabul kriteri**:
  - `tools/verify_polygon_overlap.py` çıktısında pectoral piksel sayısı
    re-export sonrasında **artmalı** (overlap geri yüklendi).
  - Retrain sonrası MLO breast tissue Dice ≥ 0.05 mutlak artış (vs eski).
  - MLO pectoral Dice da düşmemeli (≥ eski seviye).
  - Karşılaştırma: `zero_shot_test.py --limit 30 --save-vis` öncesi/sonrası.

---

### B2 ⬜ `bbox_shift` default 10 → 5 (retrain'e girerken)

- **Doğrulama durumu**: Bowang-lab `finetune_sam2_img.py`'de `bbox_shift`
  256-space'te uygulanıyor (vendor kodunda doğrulanmalı, sözde böyle).
  Nipple ortalama 8-15 px → 10 px jitter saldırgan.
- **Yapılacak iş**: `train_medsam2.py:37` default'unu 5 yap.
  Class-conditional jitter upstream PR gerektirir, kapsam dışı.
- **Süre**: 1 dk (B1 ile birleştirilecek retrain içinde)
- **Kabul kriteri**:
  - Loss eğrisi (work_dir/MammoBNP_v1-*/loss.png veya csv) önceki run'a
    göre **daha az spike** içersin (epoch 20+ sonrasında > 2× artış sayısı
    azalsın).

---

### B3 ⬜ `MIN_FG_PIXELS` 50 → 100 (retrain'e girerken)

- **Doğrulama durumu**: 256-space'te 50 px = 7×7. Sınırda nipple örnekleri
  loss'a gürültü ekler.
- **Önce ölç**: A4-öncesi sayım yap:
  ```python
  # tools/count_samples.py — kaç (case, class) örneği eşik aralığında?
  ```
  50-100 arası örnek %X kayıp olur. X > %20 ise eşiği 75'e indir.
- **Süre**: 10 dk sayım + 1 dk değişiklik
- **Kabul kriteri**:
  - Eğitim sample sayısı %20'den fazla düşmemeli.
  - Retrain sonrası nipple Dice'ı düşmemeli.

---

## FAZ C — Inference / UX iyileştirmeleri (kod, retrain yok)

### C1 ⬜ Iterative refinement (mask_input pathway)

- **Doğrulama durumu**: SAM2'nin native özelliği. Fine-tune decoder
  mask_input görmediği için OOD riski var.
- **Yapılacak iş**:
  - `MedSAM2Service.predict` → `prev_mask_low_res` parametresi.
  - SAM2 256×256 low-res logit'leri kabul ediyor; mevcut binary mask'ı
    `cv2.resize` ile 256×256'ya küçült, `predictor.predict(mask_input=...)`'a
    geç.
  - `InferencePipeline.run` → önceki predict'in low-res mask'ını state'e
    sakla, sonraki çağrıda geri ver.
- **Süre**: 1-2 sa
- **Kabul kriteri**:
  - **A/B test**: aynı (image, prompts) için mask_input AÇIK vs KAPALI →
    çoklu refinement (3+ tıklama) senaryosunda mask_input AÇIK olan
    versiyonun Dice'ı daha yüksek olmalı. Toggle olarak ekle, default
    KAPALI (ölçülene kadar güvenli).

---

### C2 ⬜ Per-class threshold (adaptive)

- **Doğrulama durumu**: Olası mikro-optimizasyon. Önce ölç.
- **Yapılacak iş**:
  - `MedSAM2Service.predict` → `threshold: dict[str, float] | float` parametresi.
  - Pipeline pass-through: tissue.key'e göre eşik geç.
  - `TissuePreset`'e `mask_threshold` alanı ekle: nipple=0.35, breast=0.5,
    pectoral=0.5.
- **Süre**: 30 dk
- **Kabul kriteri**:
  - A/B test (test set'in alt kümesinde): nipple Dice ↑ ≥ 0.02, breast/pectoral
    değişmesin (±0.01).

---

### C3 ⬜ Connected component cleanup (ignore disk fragmentation için)

- **Doğrulama durumu**: Gerçek UX sorunu ama kullanıcı bilinçli iki bölge
  istediyse zarar.
- **Yapılacak iş**: `HardIgnoreMask.apply` → `keep_largest: bool` flag.
  TissuePreset'e `keep_largest_component`: breast=True, pectoral=True,
  nipple=False. UI'da göstermeye gerek yok.
- **Süre**: 20 dk
- **Kabul kriteri**:
  - Manuel test: breast ortasına ignore noktası at, parça kalmasın.
  - Manuel test: iki ayrı nipple kümesi olan görüntüde (varsa) nipple modu
    her ikisini de tutsun.

---

### C4 ⬜ CC'de pectoral mode disable (view-aware UI)

- **Doğrulama durumu**: View map mevcut (`data/raw_mammo/view_map.csv`).
  Düşük öncelikli.
- **Yapılacak iş**: `ui.py` → görüntü stem'inden view'i look up et
  (`view_map.csv` yüklü), CC ise tissue radyosunda "Pectoral" seçeneğini
  greyleştir + tooltip "CC görüntüde pectoral yoktur".
- **Süre**: 30 dk
- **Kabul kriteri**:
  - CC view tag'li görüntü yüklendiğinde Pectoral radyosu disabled.
  - View bilinmiyor ise no-op (mevcut davranış).

---

## FAZ D — Reliability / monitoring (refactor)

### D1 ⬜ Validation Dice tracking script

- **Doğrulama durumu**: Bowang-lab finetune sadece train loss takip
  ediyor olabilir. Vendor koduna bakılarak doğrulanmalı.
- **Yapılacak iş**: `medsam2_experiments/tools/eval_checkpoints.py`:
  - `work_dir/MammoBNP_v1-*/medsam_model_epoch_*.pth` (varsa) veya
    sadece `best.pth` + `latest.pth`'ı sırayla load et.
  - `data/medsam2_npy/val/` üzerinde mean Dice ölç.
  - CSV yaz: `epoch, mean_dice_pectoral, mean_dice_breast, mean_dice_nipple`.
- **Süre**: 1-2 sa
- **Kabul kriteri**:
  - CSV üretilir, ≥3 sınıf için kolon var.
  - Eğer epoch checkpoint yoksa (sadece best/latest), 2 satırlık özet
    bile yeterli.

---

### D2 ⬜ Loss eğrisi spike analizi

- **Doğrulama durumu**: Kullanıcı spike'lar olduğunu söyledi ama numerik
  doğrulama yok.
- **Yapılacak iş**: `work_dir/MammoBNP_v1-*/`'de loss CSV/PNG aç,
  epoch 20+ sonrası `Δloss > 2×median` olan epoch sayısını çıkar.
  > 5 ise B2 (LR/bbox_shift) zorunlu.
- **Süre**: 30 dk
- **Kabul kriteri**:
  - Spike epoch sayısı raporlanır.
  - Eylem kararı (B2 zorunlu mu) verilir.

---

## FAZ E — Yapma listesi (gerekçeli)

### E1 ⏸ EMA / SWA — **YAPMA**
Karmaşıklık ↔ kazanç oranı kötü. 142 case'lik küçük dataset'te EMA marjinal.
Önce veri kalitesi ve hyperparams çözülsün.

### E2 ⏸ Augmentation pipeline — **VENDOR KONTROL ETMEDEN YAPMA**
Bowang-lab `finetune_sam2_img.py` zaten internal augmentation yapıyor
olabilir; duplicate eklersek doku özelliklerini bozarız. Önce vendor
kodunu oku (`vendor/MedSAM/finetune_sam2_img.py`), augmentation listesini
çıkar, eksik olanları belirle, sonra ekle.

### E3 ⏸ Per-sample loss logging — **ERTELE**
Debugging için kıymetli ama acil değil. Spike analizi (D2) önce yapılsın;
oradan zaten hangi epoch'larda sorun var anlaşılır.

### E4 ⏸ Class-conditional bbox_shift — **UPSTREAM PR GEREKTİRİR**
Bowang-lab `finetune_sam2_img.py` CLI'ı tek bir `bbox_shift` parametresi
alıyor. Per-class jitter için kodu fork'lamak veya monkey-patch lazım.
Şu an kapsamımızın dışı; B2 (default'u düşürmek) yeterli.

---

## Önerilen başlangıç sırası

```
A1 → A2 → A3   (paralel olabilir, hepsi non-destructive)
        ↓
[A1 sonucuna göre]
        ↓
B1 + B2 + B3 birlikte retrain  (eğer A1 overlap kanıtladıysa)
        ↓
D2 (spike analizi)
        ↓
C1, C2, C3 (sıralı veya paralel — UX iyileştirmeleri)
        ↓
C4, D1 (lower priority)
```

---

## Karar günlüğü (kullanıcı kararları)

| Tarih | Madde | Karar | Gerekçe |
|-------|-------|-------|---------|
| _(boş)_ | A1 | Beklemede | Onay verilirse kısa script yazılacak |
| _(boş)_ | A2 | Beklemede | `MedSAM2_latest.pt` indirilmiş mi? |

---

## İlgili kod yerleri

| Konu | Dosya | Satırlar |
|------|-------|----------|
| Polygon stacking | `export_seg_dataset_to_raw_mammo.py` | 49-67 |
| `MIN_FG_PIXELS` | `config.py` | 47 |
| `bbox_shift` default | `train_medsam2.py` | 37 |
| Box-only forward | `wrapped_model.py` | 23-48 |
| `id()` cache | `interactive/model.py` | ~80-85 |
| Class competition | `interactive/postprocess.py`, `inference.py` | — |
| Tissue presets | `interactive/prompts.py` | `TISSUE_PRESETS` |

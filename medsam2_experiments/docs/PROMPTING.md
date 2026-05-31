# İnteraktif MedSAM2 — Prompt Davranışı, Pipeline ve İyileştirmeler

Bu döküman, `interactive_demo.py` arayüzünün **ne yaptığını**, **neden öyle yaptığını**
ve **modelin tahmin yolunu (forward path)** açıklar. Hedef kitle: bu repoda çalışan
geliştirici / araştırmacı.

> Yer:
> - Kod: `medsam2_experiments/interactive/`
> - Entry point: `medsam2_experiments/interactive_demo.py`
> - Bu döküman: `medsam2_experiments/docs/PROMPTING.md`

---

## 0) Üst düzey diyagram

```
                ┌─────────────────────────────────────────────────────────┐
   PNG (RGB) ──►│  MedSAM2Service                                         │
                │   ├── SAM2 backbone (Hiera-Tiny, frozen prompt encoder) │
                │   └── Fine-tuned mask decoder (medsam_model_best.pth)   │
                └────────────┬────────────────────────────────────────────┘
                             │  set_image() → image embeddings cache
   prompts:                  │
   ├─ box (xyxy) ────────────┤
   ├─ pos points (1)─────────┤   predict() →  raw mask  (HxW uint8)
   └─ neg points (0)─────────┤              + score
                             ▼
                ┌─────────────────────────────────────────────────────────┐
                │  InferencePipeline.run(state, tissue, settings)         │
                │   1. resolve box (explicit | implicit-from-points)      │
                │   2. service.predict(box=..., points=...)               │
                │   3. fit_to_image (resize NEAREST)                      │
                │   4. ClassCompetition (sadece breast tissue için)       │
                │   5. HardIgnoreMask (radius R disk subtract)            │
                └────────────┬────────────────────────────────────────────┘
                             ▼
                       PredictionResult (mask + score + notes)
                             │
                             ▼
                    UI: render_overlay(image, state, mask)
```

---

## 0.5) Önemli: model class-agnostic

Yaygın yanlış anlaşılma: "Doku tipini Nipple seçince model nippleı buluyor".
**Hayır.** Sebebi:

- Fine-tune edilen tek bir mask decoder var; sınıf bilgisini girdi olarak
  almıyor. Tek girdi: `(image, box, points)`.
- Doku tipi seçimi (UI'daki radyo) sadece **iki** şeyi etkiler:
  1. Maskenin overlay rengi (görsel).
  2. Class competition için "hangi maske exclude olarak kullanılacak"
     eşleştirmesi (bkz. §4 ve `postprocess.py:ClassCompetition.resolve_exclude`).
- Modele tıkladığın **konum** ve box geometrisi söyler ne istediğini.
  Nipple üzerine tıkladığında nipple, breast'in ortasına tıkladığında breast
  bulur — doku radio'sundan bağımsız.

Yani UI çoklu doku sınıfı oluşturmak için kullanıcı **işbirliğini** gerektirir:
"Pectoral'i çiziyorum şimdi" diyorsa pektoral kutusuna tıklamak, "Breast"
diyorsa breast içine tıklamak. Model bu metadata'yı bilmez, sadece konumdan
çalışır. Multi-class joint fine-tune ile bu değiştirilebilir (§8).

---

## 1) Prompt türleri ve semantikleri

### 1.1 Pozitif Nokta (Positive Point)

- **Anlamı:** "Bu pikselin üzerinde olan obje, segmente etmek istediğim obje."
- **UI:** sol tıkla mod = `Pozitif Nokta`. Yeşil halka + iç dolu nokta olarak çizilir.
- **Modele beslenme:** SAM2 prompt encoder'a `(x, y, label=1)` olarak gider.
- **Etki alanı:** SAM2 nokta promptlarını embedding olarak kodlar; tek bir piksel
  değil, o piksel etrafında "obje burada" sinyalini öğrenir. Öyleyse
  *"kaç piksel kapsıyor?"* sorusunun klasik bir cevabı **yok** — pozitif nokta
  bir piksel'dir; etkisi modelin attention'ı üzerinden tüm görüntüye yayılır.
- **Birden fazla pozitif nokta:** SAM2 hepsini "aynı objenin parçası" olarak
  yorumlar. Doğru kullanım: hepsi aynı yapının üzerinde olmalı.

> **Önemli not:** Bu repodaki fine-tune (`medsam_model_best.pth`) **yalnızca box
> prompt** ile eğitildi (bkz. `wrapped_model.py`). Saf nokta promptu, fine-tune
> dağılımının dışında kalır. Bu nedenle UI varsayılanı, en az bir pozitif nokta
> verildiğinde otomatik olarak bir **Implicit Box** türetir (bkz. §3).

### 1.2 Yoksay (Ignore) Noktası — HARD davranır

Senin "katı olsun" dediğin nokta **bu**.

- **Anlamı:** "Bu noktanın etrafında segmentasyon **olmasın**."
- **UI:** sol tıkla mod = `Yoksay Noktası`. Yarıçap R'lik kırmızı disk + ortada X.
- **İki katmanlı uygulama (belt + suspenders):**
  1. **Soft prior (modele girdi):** SAM2'ye `(x, y, label=0)` (negative point)
     olarak gönderilir. Bu modeli "bu noktayı dahil etme" yönünde nudge'lar.
     Tek başına **garanti vermez** — SAM negatif noktayı *önceleyebilir* veya
     *gözardı edebilir*.
  2. **Hard subtract (post-process, garantili):** Model maskeyi döndürdükten
     sonra, her ignore noktası etrafında bir disk **mekanik olarak çıkarılır**:
     ```python
     mask[disk(x, y, radius=R)] = 0
     ```
     Yani modelin ne dediği önemli değil — o piksel artık 0.
- **Disk kapsama (kaç piksel?):** Yarıçap R px → disk alanı
  `round(π · R²)` px. Slider değiştikçe UI bu sayıyı canlı gösterir.
  Tipik değerler:

  | R (px) | Disk pikseli (≈ π·R²) |
  | -----: | --------------------: |
  |     10 |                   314 |
  |     20 |                 1,257 |
  |     30 |                 2,827 |
  |     50 |                 7,854 |

- **Kapatma:** "Modele soft negatif besle" checkbox'ı kapatılırsa sadece hard
  subtract çalışır; bu, modelin negatif noktaya hiç temas etmemesini sağlar.

### 1.3 Bounding Box

- **Anlamı:** "Obje bu dikdörtgenin içinde."
- **UI:** sol tıkla mod = `Bounding Box`. **İki tık** ile axis-aligned (eksenle
  hizalı) dikdörtgen çizilir: ilk tık = sol-üst köşe (sarı X marker'la
  gösterilir), ikinci tık = sağ-alt köşe → otomatik tahmin.
- **Modele beslenme:** SAM2 box'ı iki köşe noktası `[(x0, y0, label=2),
  (x1, y1, label=3)]` olarak işler — bu, fine-tune'un eğitildiği formattır.
- **OBB (oriented box) yok:** Geçerli sürüm sadece axis-aligned destekler.
  SAM2 mimarisi natively OBB kabul etmediği için (rotasyon → görüntü döndürme +
  ters döndürme gerekir), bu özellik kapsam dışı bırakıldı.

### 1.4 Karma promptlar

Kombinasyonlar serbestçe verilebilir:

| Prompt kombinasyonu          | Davranış                                          |
| ---------------------------- | ------------------------------------------------- |
| Box                          | **En kararlı.** Fine-tune dağılımı.               |
| Pozitif nokta(lar)           | Implicit box türetilir, sonra box+points birlikte |
| Box + pozitif noktalar       | Box geometriyi yakalar, noktalar refine eder      |
| Box + ignore noktaları       | Box'tan tahmin → diskleri çıkar                   |
| Yalnız ignore noktası        | Pozitif yok → tahmin yok (UI yumuşak yönlendirir) |

---

## 2) Modelin forward path'i — "Tahmin nasıl üretiliyor?"

Tek bir tahmin için adım adım:

1. **Görüntü → embedding (bir defa):**
   `MedSAM2Service.set_image(rgb)` çağrısı, SAM2'nin Hiera-Tiny image
   encoder'ını çalıştırır ve embedding'leri cache'ler. Aynı görüntü üzerinde
   sonraki tıklamalar bu embedding'i tekrar kullanır → her tıklama ~ms.

2. **Prompt encoding:**
   - Box → 2 corner-point token (`labels = [2, 3]`)
   - Pozitif nokta → token (`label = 1`)
   - Negatif nokta → token (`label = 0`)
   - Sparse embeddings + dense embeddings (no_mask varsayılanı) üretilir.
   - Bu blok fine-tune sırasında **donduruldu** (`requires_grad = False`,
     bkz. `wrapped_model.py:20-22`).

3. **Mask decoder forward:**
   - Image embed + prompt embed → SAM2 mask decoder.
   - **Bu decoder fine-tune'da güncellendi** — mamografi dokuları üzerinde
     ince ayar yapıldı.
   - Çıktı: low-resolution mask logits.

4. **Mask post-processing (model içi):**
   - Sigmoid → olasılık haritası.
   - Bilinear upscale → girdi çözünürlüğüne (genelde 256→1024).
   - `> 0.5` threshold → binary mask.
   - Eğer çıktı şekli orijinal görüntüyle uyuşmuyorsa, NEAREST resize ile
     hizalanır (`MedSAM2Service.fit_to_image`).

5. **Pipeline post-process (model dışı):**
   - **Class competition** (sadece breast tissue seçiliyse + toggle açıksa):
     pektoral'i yeni bir tahminle bul ve breast maskesinden çıkar.
   - **Hard ignore disks** uygulanır.

6. **UI render:**
   - Maske doku rengiyle %45 opacity overlay + kontur çizgisi.
   - Kullanıcı promptları (noktalar, box) maskenin üstüne çizilir.

---

## 3) Implicit Box: nokta-yalnız modu neden iyileştirildi?

### Sorun
Eski `interactive_demo.py` saf nokta promptu kullanıyordu
(`box=None, point_coords=pts`). Fine-tune box-only olduğu için decoder
"box olmadan" davranışı asla görmedi → çoklu nokta eklendikçe maske
tutarsız oluyordu.

### Çözüm
`InferencePipeline`, kullanıcı box çizmediği halde pozitif nokta varsa,
şu işlemi yapar:
```python
implicit_box = bbox(positive_points) + pad   # convex bbox + 64px (default)
```
ve `predict(box=implicit_box, points=...)` çağırır.

### Neden işe yarıyor?
- Decoder'ın eğitim dağılımı `(image, box) → mask`. Ona her zaman bir box
  veriyoruz; box bazen kullanıcının çizdiği, bazen pozitif noktalardan
  türetilen.
- Pad sayesinde box, gerçek objeden bir miktar dışarı taşar — bu da SAM2'nin
  tipik davranışıyla uyumlu.

### Kapatma
"Implicit box (önerilir)" checkbox'ı kapatılırsa eski davranışa döner
(saf nokta). Karşılaştırma için kullanışlıdır.

---

## 4) Class Competition — MLO breast/pectoral karışması

### Gözlem
Fine-tune **per-class binary** yapıldı (bkz. `prepare_medsam2_data.py:104-111`):
her sınıf için ayrı `(image, box, mask)` örnekleri. Sonuç:
- CC görüntülerde pektoral küçük/yok → breast tissue tahmini temiz.
- **MLO** görüntülerde pektoral büyük → breast tissue maskesi pektoralin
  içine "sızıyor" (decoder pektoralin varlığını bilmiyor).

### Çözüm: kullanıcının çizdiği pektoral maskeyi referans al

İlk versiyon "pektorali otomatik tahmin et" yaklaşımıyla yapıldı; bu, breast
prompt'unun seed'i pektoral tahmininin kutusuna kaçtığı için **breast'i
kendisinden çıkarıyordu** (maske sıfıra düşüyordu). Yeni davranış:

1. Kullanıcı **önce** "Göğüs Kası (Pectoral)" dokusunu seçer, box veya
   noktayla pektorali çizer → mask state'e kaydedilir.
2. Doku tipini "Meme Dokusu (Breast Tissue)"a çevirir (eski maske ekranda
   kalmaya devam eder), yeni prompt'larla breast'i çizer.
3. Class competition AÇIK ise, çizdiği pektoral piksellerini breast
   tahmininden mekanik çıkarır: `breast_final = breast_pred AND NOT pectoral_user`.

### Avantajlar
- Model'e ek forward pass YOK → daha hızlı, sürprizsiz.
- Kullanıcı pektorali kendi gözleriyle doğruladığı için sübtraksiyon
  doğrulanmış olur (auto-prediction'ın hatalı pektoral riski yok).
- CC görüntülerde kullanıcı pektorali çizmezse no-op; sıfır yan etki.

### Toggle davranışı
- Class competition AÇIK + pektoral mask yok → no-op.
- Class competition AÇIK + pektoral mask var → subtract uygulanır.
- Class competition KAPALI → her zaman no-op.

### Sınırlar
- Kullanıcı pektoralı kötü çizerse subtraction da kötü olur (garbage in,
  garbage out). Box prompt en güvenilir; çizim hatalıysa "Bu doku
  maskesini sil" butonuyla pektorali sıfırlayıp yeniden çiz.

---

## 4.5) Çoklu doku maske persistance'ı

UI session state şu yapıdadır:

```
session = {
  "prompts": { positive: [...], ignore: [...], box: {...} },
  "masks":   { "nipple": np.ndarray | None,
               "pectoral": np.ndarray | None,
               "breast": np.ndarray | None }
}
```

- **Prompt state**: aktif olan tek bir doku için geçerlidir (pozitifler,
  ignore'lar, box). Doku tipi değiştirildiğinde **sıfırlanır** — yeni dokuya
  yeni promptlarla başlanır.
- **Masks dict**: her doku için en son üretilen maske burada saklanır.
  Doku tipi değişse bile bu dict aynen kalır.

**Render davranışı** (`prompts.render_overlay`):
1. Aktif olmayan doku maskeleri α=0.35 ile çizilir (silik).
2. Aktif doku maskesi α=0.50 + kontur ile çizilir (öne çıkar).
3. Üzerine box ve nokta marker'ları binilir.

**Doku tipi değiştirme akışı** (`ui.on_tissue_change`):
1. Önceki tipin prompt'ları silinir.
2. `masks` dict aynen korunur.
3. Görüntü yeniden render edilir (model çağrısı yok).

**"Bu doku maskesini sil" butonu**: yalnız aktif tipin maskesini ve
prompt'larını sıfırlar; diğer maskelere dokunmaz.

**"TÜM maskeleri temizle" butonu**: session state'i tam sıfırlar.

---

## 5) UI tasarım kararları

### Tıklama akışı
- Mode = `Pozitif Nokta` → her tık state'e bir nokta ekler, anında re-predict.
- Mode = `Yoksay Noktası` → her tık ignore listesine ekler, re-predict.
- Mode = `Bounding Box` → tek tık ilk köşeyi koyar (sarı X marker görünür);
  ikinci tık dikdörtgeni tamamlar ve re-predict tetiklenir; üçüncü tık yeni
  bir kutu başlatır.

### Görsel iyileştirmeler (eski versiyona göre)
- Pozitif nokta: dış halka + dolu iç + iç-içe stroke (`prompts.py`).
- Ignore nokta: yarıçap R'lik şeffaf disk + dış halka + X işareti.
- Box: dashed sarı çizgi + 4 köşede dolu nokta marker.
- Maske: doku rengine göre overlay + kontur çizgisi.
- Info paneli: piksel sayısı, SAM skor, hangi box kullanıldı, pektoral'den
  ne kadar piksel çıkarıldı, hard subtract'ın etkisi.

### Aksiyonlar
- `↶ Son pozitif` / `↶ Son ignore` / `□ Box reset` — selectif geri al.
- `⨯ Tümünü temizle` — tüm prompt state sıfırlanır.
- `⟳ Yeniden tahminle` — ayarları değiştirdikten sonra mevcut promptlarla
  yeniden çalıştırır (radyo / slider değişikliklerinde otomatik de tetiklenir).

---

## 6) Mimari: SOLID seçimleri

| Modül                          | Sorumluluk                                | İlke |
| ------------------------------ | ----------------------------------------- | ---- |
| `interactive/model.py`         | Model yükleme + raw predict               | SRP  |
| `interactive/prompts.py`       | Prompt state + drawing                    | SRP  |
| `interactive/postprocess.py`   | HardIgnoreMask, ClassCompetition          | SRP, Strategy |
| `interactive/inference.py`     | Composition: model + post-process         | DIP  |
| `interactive/ui.py`            | Gradio orchestration                      | SRP  |
| `interactive_demo.py`          | Composition root (paths → pipeline → UI)  | DIP  |

- **OCP:** yeni post-process eklemek (örn. CRF, morfolojik smoothing) için
  `postprocess.py`'ye yeni sınıf yaz, `InferencePipeline` constructor'ına
  inject et — UI'a dokunmadan açılır.
- **LSP:** `MedSAM2Service` arkasındaki SAM2 backbone'unu farklı bir SAM2
  varyantıyla değiştirmek mümkün (signature aynı kaldığı sürece).
- **ISP:** UI sadece `InferencePipeline`'ın `run()` metodunu görür; model
  detayları gizli.
- **DIP:** UI ve pipeline somut SAM2 sınıflarına değil, `MedSAM2Service`
  ve `InferencePipeline` soyutlamalarına bağlı.

---

## 7) Performans notları

- **Image encoding 1 kere:** her görüntü değiştiğinde `set_image()` çağrılır,
  embedding cache'lenir. Sonraki tıklamalar ~ms.
- **Class competition pahalısı:** breast tissue + toggle açıksa her tahminde
  EK bir model çağrısı yapılır (pektoral için). Embedding cache'i sayesinde
  bu da hızlı, ama point-tıklama latency'sini ~2× artırır. CC görüntülerde
  prediction skor düşük çıkarsa erken çıkış yapar.
- **CPU vs CUDA:** kod CUDA varsa otomatik kullanır; CPU'da bir tıklama 1-3
  saniye sürebilir.

---

## 8) Bilinen sınırlar / Future work

- **Box-only fine-tune dağılımı:** Implicit box bunu maskeler ama gerçek
  çözüm, fine-tune'u box+point karışık prompt'la yeniden yapmaktır.
  `prepare_medsam2_data.py`'ye nokta promptu örnekleme ekleyip
  `train_medsam2.py`'yi güncellemek gerekir.
- **Multi-class joint training:** Class competition, per-class binary
  fine-tune'un yan etkisini hafifletir. Kalıcı çözüm: tek decoder, 4-sınıf
  softmax çıktı (background + pectoral + breast + nipple). Veri seti zaten
  multi-class PNG mask formatında (`config.MASK_CLASS_LABELS`), ama eğitim
  pipeline'ı binary varsayıyor.
- **OBB yok:** Mamografide pektoral hattı eğikli; bunu yakalamak istiyorsak
  OBB veya rotasyon-augmented training gerekir.
- **Ignore disk bir tek yarıçap:** Tüm ignore noktaları aynı R kullanır.
  Per-point R için UI extend edilebilir.

---

## 9) Hızlı doğrulama (manuel kabul testi)

1. `python interactive_demo.py`
2. Bir CC görüntüsü yükle:
   - Mode = `Pozitif Nokta`, doku = `Meme Dokusu`. Meme'nin ortasına 1 nokta.
   - Beklenen: meme alanına yaslanan maske (Implicit box hits).
3. Aynı görüntüde mode = `Yoksay Noktası`, meme dışına bir nokta at.
   - Beklenen: o noktanın etrafında **disk kadar** bir delik.
   - Slider'ı 10 → 50 yap; deliğin büyüdüğünü gör.
4. Bir MLO görüntüsü yükle:
   - Class competition AÇIK + doku = `Meme Dokusu` + 1 pozitif nokta meme'de.
   - Beklenen: pektoral bölgesi maskeden çıkarılmış (info panelinde
     "Class competition: removed N px" notu).
   - Class competition KAPALI ve aynı tıklama → maskenin pektorale taştığını
     karşılaştırma için gör.
5. Mode = `Bounding Box`. CC'de meme civarına iki tık. Beklenen: en kararlı,
   en temiz maske.

Eğer (1)-(5) bekleneni vermiyorsa: `MEDSAM2_INTERACTIVE_CKPT` doğru mu, fine-tune
gerçekten bittiyse mi (best.pth var mı), terminalde MedSAM2 import hataları yok mu
kontrol et.

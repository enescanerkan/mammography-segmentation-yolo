// Interface strings. English is the default; a key missing in another language falls back to English.
// Static markup carries data-i18n (text), data-i18n-title (title) and data-i18n-aria (aria-label); everything the
// scripts build asks t() directly and is rebuilt when the language changes.

const EN = {
  app_title: 'Mammography Positioning Review',
  subtitle: 'PNL · chest-wall distance · ΔD',
  threshold_label: 'ΔD threshold',
  threshold_title: 'Studies with ΔD above this are called inadequate',
  threshold_range: 'The ΔD threshold must be between 1 and 60 mm.',
  decrease: 'Decrease', increase: 'Increase',
  prev_views: 'Previous views', next_views: 'Next views',
  layout_title: 'Grid layout', style_title: 'Overlay appearance',
  criteria_btn: 'Criteria', report_btn: 'Report', edit_labels: 'Edit labels', new_study: 'New study',
  language: 'Language',
  theme_title: 'Dark / light mode', fullscreen_title: 'Full screen (F)',
  status_connecting: 'Connecting…', status_loading: 'Loading models…', status_ready: 'Ready · {n} models',
  status_missing: 'Models missing', status_offline: 'Offline',
  models_failed: 'Models failed to load.\n{list}',
  dismiss: 'Dismiss', close: 'Close', cancel: 'Cancel', cancelled: 'Cancelled.',

  splash_sub: 'Preparing the review workstation…',
  sp_server: 'Connecting to the server', sp_pose: 'Landmark models (MLO, CC)', sp_seg: 'Segmentation model', sp_retry: 'Server not reachable — retrying…',

  pretitle: 'Positioning quality',
  headline: 'Two independent methods, one verdict per breast.',
  lede: 'Drop a four-view study. Pose landmarks and segmentation masks each produce the posterior nipple line and the chest-wall distance; where they disagree is where a reader should look.',
  point1: 'Breast tissue, pectoral muscle and nipple, segmented per view',
  point2: 'PNL and chest-wall distance in millimetres, from the DICOM pixel spacing',
  point3: 'Every contour and point editable, with the measurements recomputed on save',
  privacy: 'Uploaded DICOMs are never written to disk. Only a display preview and the label coordinates are stored.',
  retention: 'Stored studies are deleted automatically after {days} days.',
  retention_off: 'Stored studies are kept until they are deleted.',
  upload_title: 'Open a study',
  upload_hint: 'R-MLO, L-MLO, R-CC and L-CC of one study — loose DICOM files or the study folder.',
  drop_main: 'Drop DICOM files or a study folder', drop_sub: 'or click to choose files',
  drop_aria: 'Drop DICOM files here, or click to choose files',
  choose_files: 'Choose files', choose_folder: 'Choose a folder',
  choose_dataset: 'Open a dataset', dataset_hint: 'A folder of many studies, reviewed one study after another.',
  ds_title: 'Dataset', ds_prev_btn: 'Previous', ds_next_btn: 'Next', ds_finish_btn: 'Finish',
  ds_prev_title: 'Previous study (PageUp)', ds_next_title: 'Next study (PageDown)', ds_keys: 'PageDown next · PageUp previous',
  ds_finish: 'Finish the dataset', ds_close: 'Leave the dataset', ds_pick: 'Go to study',
  ds_option: '{i} / {n} · {name}', ds_scanning: 'Reading the study headers… {d} / {n} files',
  ds_analysing: 'Study {i} / {n} — analysing…', ds_working: 'opening study {i}…', ds_last: 'last study',
  ds_done: 'finished · {o} / {n} opened', ds_state_failed: 'could not be analysed', ds_state_removed: 'removed',
  ds_skipped: '{n} files skipped (not DICOM)',
  ds_empty: 'No DICOM study was found in this folder.',
  ds_failed: 'Study {i} ({name}) could not be analysed: {msg}',
  ds_finished_all: 'Dataset finished: all {n} studies were opened.',
  ds_finished_counts: 'Dataset finished: {o} of {n} studies opened, {f} failed, {u} not opened.',
  busy_wait: 'Still working on the previous step — try again in a moment.',
  tree_summary: 'Accepted folder layout',
  tree_note: 'The folder name carries the view, so the files inside can be named anything. Loose DICOM files work as well.',
  how_title: 'How it works',
  step1: 'Upload', step1_d: 'MLO and CC of both breasts; view and side are read from the DICOM header, or from the folder name.',
  step2: 'Review', step2_d: 'ΔD per breast from the landmarks and from the segmentation, side by side, with the pectoral angle and the MLO criteria.',
  step3: 'Correct', step3_d: 'Edit any contour or point; every measurement and the report follow.',
  recent: 'Recent studies', refresh: 'refresh', clear: 'clear', no_recent: 'No stored studies yet.',
  st_upload: 'Uploading', st_analyse: 'Reading DICOM, placing landmarks, segmenting', st_render: 'Preparing the views',
  files_n: '{n} files', file_1: '1 file',
  disclaimer: 'Research prototype — not a medical device; every result must be checked by a reader.',
  empty_files: 'The selected files are empty.',
  clear_confirm: 'Delete every stored study, with its previews and labels? This cannot be undone.',
  deleted_1: '1 study deleted.', deleted_n: '{n} studies deleted.',
  opening: 'Opening study…', loading_views: 'Loading the views…',

  layout_1x1: 'single view', layout_2x1: 'two side by side', layout_1x2: 'two stacked', layout_2x2: 'all four views',
  cell_aria: '{c} by {r}',
  pair: 'Pair', pair_view: 'By view', pair_view_hint: 'same projection, both breasts',
  pair_breast: 'By breast', pair_breast_hint: 'both projections of one breast',
  crop: 'Crop to tissue', sync: 'Synchronise zoom & pan',
  fill_opacity: 'Fill opacity', line_width: 'Line width', image: 'Image', brightness: 'Brightness', contrast: 'Contrast',
  invert: 'Invert grey scale', reset_window: 'Reset window', reset_appearance: 'Reset appearance',

  c_breast: 'Breast tissue', c_pectoral: 'Pectoral muscle', c_nipple: 'Nipple contour', c_landmarks: 'Landmarks', c_measure_landmark: 'Measure · landmark', c_measure_mask: 'Measure · segmentation',
  layer_points: 'Pose landmarks',
  pt_nipple: 'nipple', pt_pectoral_top: 'pec ↑', pt_pectoral_bottom: 'pec ↓',
 

  not_uploaded: 'not uploaded', preview_unavailable: 'preview unavailable', enlarge: 'Enlarge {view}',
  worst_on_view: 'Worst criterion on this view: {grade}',
  badge_view: 'view: {source}', badge_view_title: 'View/side taken from {sources} — the DICOM header did not carry it',
  src_folder: 'folder name', src_file: 'file name', src_path: 'upload path',
  badge_edited: 'edited', badge_edited_title: 'Corrected by a reviewer',
  badge_no_landmarks: 'no landmarks',
  badge_mirrored: 'mirrored', badge_mirrored_title: 'The model only detected anything on the mirrored image',
  badge_rescued: 'nipple rescued',
  badge_rescued_title: 'Segmentation found the nipple at low confidence; the pose model vouched for it',

  right_breast: 'Right breast', left_breast: 'Left breast', no_views: 'no views',
  nothing_uploaded: 'Nothing uploaded for this breast.',
  method_landmark: 'Landmark', method_mask: 'Segmentation',
  verdict_good: 'Adequate', verdict_bad: 'Inadequate', no_measurement: 'no measurement',
  pnl_mlo: 'PNL · MLO', wall_cc: 'Nipple → wall · CC', threshold: 'Threshold',
  cmp_agree: 'Both methods agree.', cmp_disagree: 'The methods disagree on the verdict — review this study.',
  cmp_wide: 'Same verdict, but the measurements differ by {gap} mm.',
  cmp_partial: 'Only one method produced a result.', cmp_none: 'Neither method produced a result.',

  criteria_title: 'Positioning criteria',
  criteria_counts: '{g} good · {a} acceptable · {p} poor',
  criteria_empty: 'No criteria for this study yet — they are computed from the segmentation masks, so a view without them has no usable contours.',
  grade_good: 'Good', grade_acceptable: 'Acceptable', grade_poor: 'Poor', grade_na: 'n/a', grade_info: 'information',
  mlo_alone_label: 'MLO alone', mlo_alone_adequate: 'adequate', mlo_alone_inadequate: 'inadequate',
  both_breasts: 'Both breasts', est: 'est', est_title: 'Estimated — shown, never graded',
  notes: 'Notes',
  notice_flagged: 'The two methods disagree — this study needs a review.',
  notice_spacing: 'Pixel spacing is missing from the header ({views}) — millimetre values assume 0.085 mm and are estimates.',
  download_labels: 'Download labels', delete_study: 'Delete study',
  delete_confirm: 'This study and its previews will be permanently deleted. Continue?',
  foot_study: 'study {id}', foot_views: '{n} views', foot_edited: 'edited {time}',

  report: 'Report', report_lang: 'Report language', copy: 'Copy', copied: 'Copied', download: 'Download', print: 'Print',
  clipboard_refused: 'The browser refused clipboard access — use Download instead.',
  stale_code: 'The page is running an older copy of its own code. Reload with Ctrl+Shift+R (Cmd+Shift+R on a Mac).',

  back: '← Back to review', unsaved: 'Unsaved changes', undo: 'Undo', revert: 'Revert view',
  save: 'Save & recompute', saving: 'Saving…', save_failed: 'Save failed — retry',
  ed_view: 'View', ed_structure: 'Structure', ed_tool: 'Tool', ed_shortcuts: 'Shortcuts',
  k_select: 'Select', k_polygon: 'Polygon', k_point: 'Point', k_finish: 'Finish polygon',
  k_cancel: 'Cancel drawing', k_delete_vertex: 'Delete a vertex', k_insert: 'Insert a vertex',
  k_undo: 'Undo',
  k_fit: 'Fit to view', k_zoom: 'Zoom', k_pan: 'Pan',
  key_alt_click: 'Alt+click', key_shift_click: 'Shift+click', key_dblclick: 'Double-click', key_scroll: 'Scroll',
  key_middle: 'Middle-drag',
  hint_select: 'Drag a handle to move it. Shift+click an edge inserts a vertex, Alt+click removes one.',
  hint_polygon: 'Click to place vertices. Enter, a double-click, or a click on the first point closes the shape and replaces the current one.',
  hint_point: 'Click to place the selected landmark. Alt+click removes it.',
  cls_breast: 'Breast tissue', cls_pectoral: 'Pectoral muscle', cls_nipple: 'Nipple contour',
  cls_nipple_point: 'Nipple point', cls_pec_top: 'Pectoral top', cls_pec_bottom: 'Pectoral bottom',
  st_pts: '{n} pts', st_none: 'none', st_set: 'set', edited_mark: 'Edited',
 
  discard_confirm: 'Discard the unsaved label changes?',
  zoom_in: 'Zoom in', zoom_out: 'Zoom out', fit: 'Fit',

  lightbox_aria: 'Enlarged view', lightbox_hint: 'Drag to pan · Scroll to zoom · Esc to close',
  views_n: '{n} views', flag_title: 'The methods disagree',

  err_timeout: 'The server did not respond in time. Is it still running?',
  err_network: 'Could not reach the server. Is the service running?',
  err_invalid: 'Invalid request: {msg}', err_http: 'Request failed (HTTP {status}).',
};

const TR = {
  app_title: 'Mamografi Pozisyonlama Değerlendirmesi',
  subtitle: 'PNL · göğüs duvarı mesafesi · ΔD',
  threshold_label: 'ΔD eşiği',
  threshold_title: 'ΔD bu değerin üzerindeyse pozisyonlama yetersiz sayılır',
  threshold_range: 'ΔD eşiği 1 ile 60 mm arasında olmalıdır.',
  decrease: 'Azalt', increase: 'Artır',
  prev_views: 'Önceki görüntüler', next_views: 'Sonraki görüntüler',
  layout_title: 'Izgara düzeni', style_title: 'Katman görünümü',
  criteria_btn: 'Kriterler', report_btn: 'Rapor', edit_labels: 'Etiketleri düzenle', new_study: 'Yeni çalışma',
  language: 'Dil',
  theme_title: 'Koyu / açık tema', fullscreen_title: 'Tam ekran (F)',
  status_connecting: 'Bağlanıyor…', status_loading: 'Modeller yükleniyor…', status_ready: 'Hazır · {n} model',
  status_missing: 'Model eksik', status_offline: 'Çevrim dışı',
  models_failed: 'Modeller yüklenemedi.\n{list}',
  dismiss: 'Kapat', close: 'Kapat', cancel: 'İptal', cancelled: 'İptal edildi.',

  splash_sub: 'Değerlendirme istasyonu hazırlanıyor…',
  sp_server: 'Sunucuya bağlanılıyor', sp_pose: 'Nokta modelleri (MLO, CC)', sp_seg: 'Segmentasyon modeli', sp_retry: 'Sunucuya ulaşılamıyor — yeniden deneniyor…',

  pretitle: 'Pozisyonlama kalitesi',
  headline: 'İki bağımsız yöntem, her meme için tek karar.',
  lede: 'Dört görüntülük bir çalışma bırakın. Nokta modeli ve segmentasyon maskeleri posterior meme ucu çizgisini (PNL) ve göğüs duvarı mesafesini ayrı ayrı ölçer; ayrıştıkları yer okuyucunun bakması gereken yerdir.',
  point1: 'Her görüntüde meme dokusu, pektoral kas ve meme ucu segmentasyonu',
  point2: 'DICOM piksel aralığından milimetre cinsinden PNL ve göğüs duvarı mesafesi',
  point3: 'Her kontur ve nokta düzenlenebilir; kaydedince ölçümler yeniden hesaplanır',
  privacy: 'Yüklenen DICOM dosyaları diske yazılmaz. Yalnız bir görüntüleme önizlemesi ve etiket koordinatları saklanır.',
  retention: 'Saklanan çalışmalar {days} gün sonra kendiliğinden silinir.',
  retention_off: 'Saklanan çalışmalar silinene kadar tutulur.',
  upload_title: 'Bir çalışma açın',
  upload_hint: 'Bir çalışmanın R-MLO, L-MLO, R-CC ve L-CC görüntüleri — tek tek DICOM dosyaları ya da çalışma klasörü.',
  drop_main: 'DICOM dosyalarını ya da çalışma klasörünü bırakın', drop_sub: 'ya da dosya seçmek için tıklayın',
  drop_aria: 'DICOM dosyalarını buraya bırakın ya da seçmek için tıklayın',
  choose_files: 'Dosya seç', choose_folder: 'Klasör seç',
  choose_dataset: 'Veri seti aç', dataset_hint: 'Birçok çalışma içeren bir klasör; çalışmalar sırayla tek tek açılır.',
  ds_title: 'Veri seti', ds_prev_btn: 'Önceki', ds_next_btn: 'Sonraki', ds_finish_btn: 'Bitir',
  ds_prev_title: 'Önceki çalışma (PageUp)', ds_next_title: 'Sonraki çalışma (PageDown)', ds_keys: 'PageDown sonraki · PageUp önceki',
  ds_finish: 'Veri setini bitir', ds_close: 'Veri setinden çık', ds_pick: 'Çalışmaya git',
  ds_option: '{i} / {n} · {name}', ds_scanning: 'Çalışma başlıkları okunuyor… {d} / {n} dosya',
  ds_analysing: 'Çalışma {i} / {n} — analiz ediliyor…', ds_working: '{i}. çalışma açılıyor…', ds_last: 'son çalışma',
  ds_done: 'bitti · {o} / {n} açıldı', ds_state_failed: 'analiz edilemedi', ds_state_removed: 'silindi',
  ds_skipped: '{n} dosya atlandı (DICOM değil)',
  ds_empty: 'Bu klasörde DICOM çalışması bulunamadı.',
  ds_failed: 'Çalışma {i} ({name}) analiz edilemedi: {msg}',
  ds_finished_all: 'Veri seti bitti: {n} çalışmanın hepsi açıldı.',
  ds_finished_counts: 'Veri seti bitti: {n} çalışmadan {o} açıldı, {f} analiz edilemedi, {u} açılmadı.',
  busy_wait: 'Önceki işlem sürüyor — biraz sonra yeniden deneyin.',
  tree_summary: 'Kabul edilen klasör düzeni',
  tree_note: 'Görünüm klasör adından okunur; içindeki dosyalar herhangi bir adla olabilir. Tek tek DICOM dosyaları da kabul edilir.',
  how_title: 'Nasıl çalışır',
  step1: 'Yükleyin', step1_d: 'İki memenin MLO ve CC görüntüleri; görünüm ve taraf DICOM başlığından ya da klasör adından okunur.',
  step2: 'İnceleyin', step2_d: 'Her meme için nokta ve segmentasyon kaynaklı ΔD yan yana; pektoral açı ve MLO kriterleriyle birlikte.',
  step3: 'Düzeltin', step3_d: 'Herhangi bir konturu ya da noktayı düzenleyin; tüm ölçümler ve rapor güncellenir.',
  recent: 'Son çalışmalar', refresh: 'yenile', clear: 'temizle', no_recent: 'Henüz saklanan çalışma yok.',
  st_upload: 'Yükleniyor', st_analyse: 'DICOM okunuyor, noktalar yerleştiriliyor, segmentasyon yapılıyor',
  st_render: 'Görüntüler hazırlanıyor',
  files_n: '{n} dosya', file_1: '1 dosya',
  disclaimer: 'Araştırma prototipi — tıbbi cihaz değildir; her sonuç bir okuyucu tarafından kontrol edilmelidir.',
  empty_files: 'Seçilen dosyalar boş.',
  clear_confirm: 'Saklanan tüm çalışmalar önizlemeleri ve etiketleriyle silinsin mi? Bu işlem geri alınamaz.',
  deleted_1: '1 çalışma silindi.', deleted_n: '{n} çalışma silindi.',
  opening: 'Çalışma açılıyor…', loading_views: 'Görüntüler yükleniyor…',

  layout_1x1: 'tek görüntü', layout_2x1: 'yan yana iki', layout_1x2: 'alt alta iki', layout_2x2: 'dört görüntü',
  cell_aria: '{c} × {r}',
  pair: 'Eşle', pair_view: 'Görünüme göre', pair_view_hint: 'aynı projeksiyon, iki meme',
  pair_breast: 'Memeye göre', pair_breast_hint: 'bir memenin iki projeksiyonu',
  crop: 'Dokuya kırp', sync: 'Yakınlaştırma ve kaydırmayı eşle',
  fill_opacity: 'Dolgu opaklığı', line_width: 'Çizgi kalınlığı', image: 'Görüntü', brightness: 'Parlaklık',
  contrast: 'Kontrast', invert: 'Gri skalayı ters çevir', reset_window: 'Pencereyi sıfırla',
  reset_appearance: 'Görünümü sıfırla',

  c_breast: 'Meme dokusu', c_pectoral: 'Pektoral kas', c_nipple: 'Meme ucu konturu', c_landmarks: 'Noktalar', c_measure_landmark: 'Ölçüm · nokta', c_measure_mask: 'Ölçüm · segmentasyon',
  layer_points: 'Poz noktaları',
  pt_nipple: 'meme ucu', pt_pectoral_top: 'pek ↑', pt_pectoral_bottom: 'pek ↓',
 

  not_uploaded: 'yüklenmedi', preview_unavailable: 'önizleme yok', enlarge: '{view} büyüt',
  worst_on_view: 'Bu görüntüdeki en kötü kriter: {grade}',
  badge_view: 'görünüm: {source}', badge_view_title: 'Görünüm/taraf {sources} kaynağından alındı — DICOM başlığında yoktu',
  src_folder: 'klasör adı', src_file: 'dosya adı', src_path: 'yükleme yolu',
  badge_edited: 'düzenlendi', badge_edited_title: 'Bir okuyucu tarafından düzeltildi',
  badge_no_landmarks: 'nokta yok',
  badge_mirrored: 'aynalı', badge_mirrored_title: 'Model yalnız aynalanmış görüntüde bir şey buldu',
  badge_rescued: 'meme ucu kurtarıldı',
  badge_rescued_title: 'Segmentasyon meme ucunu düşük güvenle buldu; nokta modeli konumunu doğruladı',

  right_breast: 'Sağ meme', left_breast: 'Sol meme', no_views: 'görüntü yok',
  nothing_uploaded: 'Bu meme için görüntü yüklenmedi.',
  method_landmark: 'Nokta bazlı', method_mask: 'Segmentasyon',
  verdict_good: 'Yeterli', verdict_bad: 'Yetersiz', no_measurement: 'ölçüm yok',
  pnl_mlo: 'PNL · MLO', wall_cc: 'Meme ucu → duvar · CC', threshold: 'Eşik',
  cmp_agree: 'İki yöntem aynı kararı verdi.', cmp_disagree: 'Yöntemler kararda ayrışıyor — çalışmayı gözden geçirin.',
  cmp_wide: 'Karar aynı, ancak ölçümler {gap} mm ayrışıyor.',
  cmp_partial: 'Yalnız bir yöntem sonuç üretti.', cmp_none: 'İki yöntem de sonuç üretmedi.',

  criteria_title: 'Pozisyonlama kriterleri',
  criteria_counts: '{g} iyi · {a} kabul edilebilir · {p} zayıf',
  criteria_empty: 'Bu çalışma için henüz kriter yok — kriterler segmentasyon maskelerinden hesaplanır; maskesi olmayan görüntüde kullanılabilir kontur yoktur.',
  grade_good: 'İyi', grade_acceptable: 'Kabul edilebilir', grade_poor: 'Zayıf', grade_na: 'yok', grade_info: 'bilgi',
  mlo_alone_label: 'MLO tek başına', mlo_alone_adequate: 'yeterli', mlo_alone_inadequate: 'yetersiz',
  both_breasts: 'İki meme', est: 'tah', est_title: 'Tahmini — gösterilir, derecelendirilmez',
  notes: 'Notlar',
  notice_flagged: 'İki yöntem ayrışıyor — bu çalışma gözden geçirilmeli.',
  notice_spacing: 'Başlıkta piksel aralığı yok ({views}) — milimetre değerleri 0,085 mm varsayar ve tahminidir.',
  download_labels: 'Etiketleri indir', delete_study: 'Çalışmayı sil',
  delete_confirm: 'Bu çalışma ve önizlemeleri kalıcı olarak silinecek. Devam edilsin mi?',
  foot_study: 'çalışma {id}', foot_views: '{n} görüntü', foot_edited: 'düzenlendi {time}',

  report: 'Rapor', report_lang: 'Rapor dili', copy: 'Kopyala', copied: 'Kopyalandı', download: 'İndir', print: 'Yazdır',
  clipboard_refused: 'Tarayıcı panoya erişime izin vermedi — İndir seçeneğini kullanın.',
  stale_code: 'Sayfa kendi kodunun eski bir kopyasını çalıştırıyor. Ctrl+Shift+R (Mac’te Cmd+Shift+R) ile yeniden yükleyin.',

  back: '← İncelemeye dön', unsaved: 'Kaydedilmemiş değişiklikler', undo: 'Geri al', revert: 'Görüntüyü geri al',
  save: 'Kaydet ve yeniden hesapla', saving: 'Kaydediliyor…', save_failed: 'Kaydedilemedi — yeniden dene',
  ed_view: 'Görüntü', ed_structure: 'Yapı', ed_tool: 'Araç', ed_shortcuts: 'Kısayollar',
  k_select: 'Seç', k_polygon: 'Çokgen', k_point: 'Nokta', k_finish: 'Çokgeni bitir',
  k_cancel: 'Çizimi iptal et', k_delete_vertex: 'Köşe sil', k_insert: 'Köşe ekle',
  k_undo: 'Geri al',
  k_fit: 'Sığdır', k_zoom: 'Yakınlaştır', k_pan: 'Kaydır',
  key_alt_click: 'Alt+tık', key_shift_click: 'Shift+tık', key_dblclick: 'Çift tık', key_scroll: 'Tekerlek',
  key_middle: 'Orta tuşla sürükle',
  hint_select: 'Taşımak için bir tutamacı sürükleyin. Shift+tık kenara köşe ekler, Alt+tık köşeyi siler.',
  hint_polygon: 'Köşe eklemek için tıklayın. Enter, çift tık ya da ilk noktaya tık şekli kapatır ve mevcut şeklin yerine geçer.',
  hint_point: 'Seçili noktayı yerleştirmek için tıklayın. Alt+tık siler.',
  cls_breast: 'Meme dokusu', cls_pectoral: 'Pektoral kas', cls_nipple: 'Meme ucu konturu',
  cls_nipple_point: 'Meme ucu noktası', cls_pec_top: 'Pektoral üst', cls_pec_bottom: 'Pektoral alt',
  st_pts: '{n} nokta', st_none: 'yok', st_set: 'var', edited_mark: 'Düzenlendi',
 
  discard_confirm: 'Kaydedilmemiş etiket değişiklikleri silinsin mi?',
  zoom_in: 'Yakınlaştır', zoom_out: 'Uzaklaştır', fit: 'Sığdır',

  lightbox_aria: 'Büyütülmüş görüntü', lightbox_hint: 'Kaydırmak için sürükleyin · Yakınlaştırmak için tekerlek · Kapatmak için Esc',
  views_n: '{n} görüntü', flag_title: 'Yöntemler ayrışıyor',

  err_timeout: 'Sunucu zamanında yanıt vermedi. Hâlâ çalışıyor mu?',
  err_network: 'Sunucuya ulaşılamadı. Servis çalışıyor mu?',
  err_invalid: 'Geçersiz istek: {msg}', err_http: 'İstek başarısız (HTTP {status}).',
};

// Criterion names and descriptions, keyed by the criterion id the server sends. English falls back to the
// server's own text, so only the translations are listed.
const CRITERIA_TR = {
  pectoral_angle: 'Pektoral açı',
  pectoral_depth: 'Pektoral meme ucu seviyesine iniyor',
  nipple_centred: 'Meme ucu ortalanmış',
  mlo_alone: 'MLO tek başına',
  gantry: 'C-kol açısı',
};

// The server's detail texts (criteria.py), each a pattern whose numbers carry over into the translation. A detail
// with no pattern here is shown as the server wrote it.
const DETAILS_TR = [
  [/^angle of the fitted pectoral line from the vertical$/, () => 'oturtulan pektoral çizginin dikeyle açısı'],
  [/^positive = pectoral extends below the nipple$/, () => 'pozitif = pektoral meme ucunun altına uzanıyor'],
  [/^distance from the middle of the breast's height \((\d+)% of the half-height\)$/,
    (share) => `meme yüksekliğinin ortasına uzaklık (yarı yüksekliğe oranı: %${share})`],
  [/^from the DICOM header$/, () => 'DICOM başlığından'],
  [/^the posterior nipple line meets the pectoral line inside the image$/,
    () => 'PNL pektoral çizgiyle görüntünün içinde kesişiyor'],
  [/^the posterior nipple line meets the pectoral line outside the image, ([\d.]+) mm beyond the edge$/,
    (mm) => `PNL pektoral çizgiyle görüntünün dışında kesişiyor, kenarın ${mm.replace('.', ',')} mm ötesinde`],
  [/^the posterior nipple line meets the pectoral line outside the image$/,
    () => 'PNL pektoral çizgiyle görüntünün dışında kesişiyor'],
  [/^needs the pectoral line and the nipple$/, () => 'pektoral çizgi ve meme ucu gerekli'],
  [/^no image geometry stored for this study$/, () => 'bu çalışma için görüntü geometrisi kayıtlı değil'],
  [/^no pectoral line$/, () => 'pektoral çizgisi yok'],
  [/^needs the pectoral and the nipple$/, () => 'pektoral ve meme ucu gerekli'],
  [/^needs the nipple and the breast$/, () => 'meme ucu ve meme dokusu gerekli'],
  [/^no breast tissue$/, () => 'meme dokusu yok'],
];

// The measurement code explains a missing value in English; the review screen translates it.
const REASONS_TR = {
  'needs both MLO and CC': 'MLO ve CC birlikte gerekli',
  'no measurement': 'ölçüm yok',
  'nipple keypoint not detected': 'meme ucu noktası bulunamadı',
  'pectoral keypoints not detected': 'pektoral noktaları bulunamadı',
  'no nipple mask': 'meme ucu maskesi yok',
  'no breast tissue mask': 'meme dokusu maskesi yok',
  'nipple mask is empty': 'meme ucu maskesi boş',
  'chest wall edge not found': 'göğüs duvarı kenarı bulunamadı',
  'no pectoral mask': 'pektoral maskesi yok',
  'could not fit a pectoral line': 'pektoral çizgisi oturtulamadı',
};

const LANGS = { en: EN, tr: TR };
const STORAGE_KEY = 'mammo.lang.v1';
const listeners = new Set();
let current = 'en';

export const LANGUAGES = Object.keys(LANGS);

export function language() { return current; }

export function t(key, vars = {}, fallback) {
  const table = LANGS[current] || EN;
  const text = table[key] ?? EN[key] ?? fallback ?? key;
  return String(text).replace(/\{(\w+)\}/g, (_, name) => (name in vars ? vars[name] : `{${name}}`));
}

export function criterionName(item) {
  return current === 'tr' && CRITERIA_TR[item.id] ? CRITERIA_TR[item.id] : item.name;
}

export function criterionDetail(item) {
  const detail = item.detail || '';
  if (current !== 'tr') return detail;
  for (const [pattern, translate] of DETAILS_TR) {
    const match = detail.match(pattern);
    if (match) return translate(...match.slice(1));
  }
  return detail;
}

export function reason(text) {
  if (!text) return t('no_measurement');
  return current === 'tr' ? (REASONS_TR[text] || text) : text;
}

export function locale() { return current === 'tr' ? 'tr-TR' : 'en-GB'; }

export function savedLanguage(fallback = 'en') {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    return saved && LANGS[saved] ? saved : fallback;
  } catch (error) { return fallback; }
}

/** Apply a language to the static markup and tell every listener; `remember` stores it as the reader's choice. */
export function setLanguage(lang, { remember = false } = {}) {
  current = LANGS[lang] ? lang : 'en';
  document.documentElement.lang = current;
  if (remember) {
    try { localStorage.setItem(STORAGE_KEY, current); } catch (error) { /* private mode: session only */ }
  }
  for (const node of document.querySelectorAll('[data-i18n]')) node.textContent = t(node.dataset.i18n);
  for (const node of document.querySelectorAll('[data-i18n-title]')) node.title = t(node.dataset.i18nTitle);
  for (const node of document.querySelectorAll('[data-i18n-aria]')) node.setAttribute('aria-label', t(node.dataset.i18nAria));
  for (const listener of listeners) listener(current);
}

export function onLanguage(listener) { listeners.add(listener); }

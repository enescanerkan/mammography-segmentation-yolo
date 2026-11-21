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
  ds_skipped: '{n} files skipped (not DICOM images)',
  ds_left_out: 'This study has {n} files but the server takes {limit} per study; not sent: {left}. Files whose path names a view (MLO, CC) are sent first.',
  ds_left_short: '{left} of {n} files of this study not sent',
  ds_unreadable: 'The folder could not be read: {msg}',
  ds_empty: 'No DICOM study was found in this folder.',
  ds_failed: 'Study {i} ({name}) could not be analysed: {msg}',
  ds_finished_all: 'Dataset finished: all {n} studies were opened.',
  ds_finished_counts: 'Dataset finished ({n} studies): {parts}.',
  ds_part_opened: '{c} opened ✓', ds_part_analysed: '{c} analysed but not opened •',
  ds_part_failed: '{c} could not be analysed ✕', ds_part_removed: '{c} removed –', ds_part_unseen: '{c} not reached',
  ds_mark_title: '✓ opened · • on the server, not opened in this session · ✕ could not be analysed · – removed',
  ds_legend: '✓ opened · • analysed, not opened yet · ✕ could not be analysed · – removed',
  busy_wait: 'Still working on the previous step — try again in a moment.',
  tree_summary: 'Accepted folder layout',
  tree_note: 'The folder name carries the view, so the files inside can be named anything. Loose DICOM files work as well.',
  how_title: 'How it works',
  step1: 'Upload', step1_d: 'MLO and CC of both breasts; view and side are read from the DICOM header, or from the folder name.',
  step2: 'Review', step2_d: 'ΔD per breast from the landmarks and from the segmentation, side by side, with the MLO quality and the pectoral angle.',
  step3: 'Correct', step3_d: 'Edit any contour or point; every measurement and the report follow.',
  recent: 'Recent studies', refresh: 'refresh', clear: 'clear', no_recent: 'No stored studies yet.',
  st_upload: 'Uploading', st_analyse: 'Reading DICOM, placing landmarks, segmenting', st_render: 'Preparing the views',
  files_n: '{n} files', file_1: '1 file',
  disclaimer: 'Research prototype — not a medical device; every result must be checked by a reader.',
  empty_files: 'The selected files are empty (0 bytes): {files}.',
  empty_file_1: 'The selected file is empty (0 bytes): {files}.',
  empty_skipped: 'Empty files (0 bytes) were not sent: {files}.',
  only_dicom: 'This application reads DICOM (.dcm) only; PNG/JPG files have no pixel spacing or view information, so mm cannot be measured. Not used: {names}.',
  no_dicom_files: 'None of the selected files can be a DICOM image ({n} skipped: hidden files, text, pictures or archives).',
  more_files: '+{n} more',
  clear_arm: 'delete all {n}?', clear_title: 'Delete every study stored on the server',
  clear_confirm: 'All {n} stored studies will be deleted from the server, with their previews and labels. This cannot be undone. Click “delete all {n}?” again to confirm; otherwise nothing happens.',
  clear_unopened: '({k} of them were analysed ahead in a dataset and not opened yet, so they are not in the list.)',
  no_recent_opened: 'No study has been opened yet ({k} analysed ahead in a dataset, waiting to be opened).',
  clear_nothing: 'No study is stored on the server.',
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
  badge_brightened: 'nipple found after brightening',
  badge_brightened_title: 'Not found on the image as stored; found after brightening the front of the breast. Check its position.',
  badge_estimated: 'nipple estimated (contour)',
  badge_estimated_title: 'Nipple not found: the segmentation measurement is taken at the breast-tissue contour point farthest from the pectoral line (MLO) or the chest wall (CC). It enters the verdict as an estimate.',
  est_short: 'est.', est_title: 'estimated — nipple not found (contour point)',
  card_estimated: 'Estimated — nipple not found on {views}; measured at the farthest breast-tissue contour point.',
  notice_estimated_MLO: '{view}: nipple not found — the segmentation PNL is estimated at the breast-tissue contour point farthest from the pectoral line.',
  notice_estimated_CC: '{view}: nipple not found — the segmentation distance is estimated at the breast-tissue contour point farthest from the chest wall.',

  right_breast: 'Right breast', left_breast: 'Left breast', no_views: 'no views',
  nothing_uploaded: 'Nothing uploaded for this breast.',
  method_landmark: 'Landmark', method_mask: 'Segmentation',
  verdict_good: 'Adequate', verdict_bad: 'Inadequate', no_measurement: 'no measurement',
  pnl_mlo: 'PNL · MLO', wall_cc: 'Nipple → wall · CC', threshold: 'Threshold',
  cmp_agree: 'Both methods agree.', cmp_disagree: 'The methods disagree on the verdict — review this study.',
  cmp_wide: 'Same verdict, but the measurements differ by {gap} mm.',
  cmp_partial: 'Only one method produced a result.', cmp_none: 'Neither method produced a result.',

  criteria_title: 'Positioning criteria',
  criteria_counts: '{g} good · {a} acceptable · {p} poor', criteria_counts_gp: '{g} good · {p} poor',
  criteria_empty: 'No criteria for this study yet — they are computed from the segmentation masks, so a view without them has no usable contours.',
  grade_good: 'good', grade_acceptable: 'acceptable', grade_poor: 'poor', grade_na: 'n/a', grade_info: 'information',
  mlo_quality_label: 'MLO quality', mlo_quality_good: 'good', mlo_quality_poor: 'poor',
  notes: 'Notes',
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
  key_middle: 'Middle-drag', key_pan_drag: 'Right- or middle-drag', key_h: 'H',
  k_hide: 'Hide / show contours',
  ed_hide: 'Hide contours', ed_show: 'Show contours',
  ed_hide_title: 'Hide the breast, pectoral and nipple contours to see the image under them (H). Only the view changes; the labels stay as they are.',
  hint_hidden: 'Contours are hidden (H shows them again): only the landmark points can be edited meanwhile.',
  st_hidden: 'hidden',
  ed_clean: 'Smooth / clean labels',
  ed_clean_title: 'Apply the cleaning rules to this view: each structure keeps its largest region, holes are filled, stray pixels go to the nearest structure and the edges are smoothed. An unsaved edit — Undo takes it back.',
  ed_cleaning: 'Cleaning…', ed_clean_none: 'Nothing to clean on this view: its contours already follow the rules.',
  ed_clean_done: 'Cleaned: {classes}. Not saved yet — Undo takes it back.',
  ed_clean_done_view: '{view} cleaned: {classes}. Not saved yet — Undo takes it back.',
  ed_clean_none_view: 'Nothing to clean on {view}: its contours already follow the rules.',
  ed_clean_stale: '{view} was edited while it was being cleaned, so the cleaning was not applied. Click again to clean it as it is now.',
  hint_select: 'Drag a handle to move it. Shift+click an edge inserts a vertex, Alt+click removes one.',
  hint_polygon: 'Click to place vertices. Enter, a double-click, or a click on the first point closes the shape and replaces the current one.',
  hint_point: 'Click to place the selected landmark. Alt+click removes it.',
  cls_breast: 'Breast tissue', cls_pectoral: 'Pectoral muscle', cls_nipple: 'Nipple contour',
  cls_nipple_point: 'Nipple point', cls_pec_top: 'Pectoral top', cls_pec_bottom: 'Pectoral bottom',
  st_pts: '{n} pts', st_none: 'none', st_set: 'set', edited_mark: 'Edited',
 
  discard_confirm: 'Discard the unsaved label changes?',
  zoom_in: 'Zoom in', zoom_out: 'Zoom out', fit: 'Fit',

  lightbox_aria: 'Enlarged view', lightbox_hint: 'Drag to pan · Scroll to zoom · Esc to close',
  views_n: '{n} views', flag_title: 'The methods disagree on the verdict', flag_wide_title: 'Same verdict, but the measurements differ',
  flag_partial_title: 'Only one method produced a result',

  err_timeout: 'The server did not respond in time. Is it still running?',
  err_network: 'Could not reach the server. Is the service running?',
  err_file_unreadable: 'The browser could not read {file}; it may have been moved or changed.',
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
  ds_skipped: '{n} dosya atlandı (DICOM görüntüsü değil)',
  ds_left_out: 'Bu çalışmada {n} dosya var ama sunucu çalışma başına {limit} dosya alıyor; gönderilmeyen: {left}. Yolunda görünüm adı (MLO, CC) geçen dosyalar önce gönderilir.',
  ds_left_short: 'bu çalışmanın {n} dosyasından {left} tanesi gönderilmedi',
  ds_unreadable: 'Klasör okunamadı: {msg}',
  ds_empty: 'Bu klasörde DICOM çalışması bulunamadı.',
  ds_failed: 'Çalışma {i} ({name}) analiz edilemedi: {msg}',
  ds_finished_all: 'Veri seti bitti: {n} çalışmanın hepsi açıldı.',
  ds_finished_counts: 'Veri seti bitti ({n} çalışma): {parts}.',
  ds_part_opened: '{c} açıldı ✓', ds_part_analysed: '{c} analiz edildi ama açılmadı •',
  ds_part_failed: '{c} analiz edilemedi ✕', ds_part_removed: '{c} silindi –', ds_part_unseen: '{c} sırası gelmedi',
  ds_mark_title: '✓ açıldı · • sunucuda kayıtlı, bu oturumda açılmadı · ✕ analiz edilemedi · – silindi',
  ds_legend: '✓ açıldı · • analiz edildi, henüz açılmadı · ✕ analiz edilemedi · – silindi',
  busy_wait: 'Önceki işlem sürüyor — biraz sonra yeniden deneyin.',
  tree_summary: 'Kabul edilen klasör düzeni',
  tree_note: 'Görünüm klasör adından okunur; içindeki dosyalar herhangi bir adla olabilir. Tek tek DICOM dosyaları da kabul edilir.',
  how_title: 'Nasıl çalışır',
  step1: 'Yükleyin', step1_d: 'İki memenin MLO ve CC görüntüleri; görünüm ve taraf DICOM başlığından ya da klasör adından okunur.',
  step2: 'İnceleyin', step2_d: 'Her meme için nokta ve segmentasyon kaynaklı ΔD yan yana; MLO kalitesi ve pektoral açıyla birlikte.',
  step3: 'Düzeltin', step3_d: 'Herhangi bir konturu ya da noktayı düzenleyin; tüm ölçümler ve rapor güncellenir.',
  recent: 'Son çalışmalar', refresh: 'yenile', clear: 'temizle', no_recent: 'Henüz saklanan çalışma yok.',
  st_upload: 'Yükleniyor', st_analyse: 'DICOM okunuyor, noktalar yerleştiriliyor, segmentasyon yapılıyor',
  st_render: 'Görüntüler hazırlanıyor',
  files_n: '{n} dosya', file_1: '1 dosya',
  disclaimer: 'Araştırma prototipi — tıbbi cihaz değildir; her sonuç bir okuyucu tarafından kontrol edilmelidir.',
  empty_files: 'Seçilen dosyalar boş (0 bayt): {files}.',
  empty_file_1: 'Seçilen dosya boş (0 bayt): {files}.',
  empty_skipped: 'Boş dosyalar (0 bayt) gönderilmedi: {files}.',
  only_dicom: 'Bu uygulama yalnız DICOM (.dcm) okur; PNG/JPG’de piksel aralığı ve görünüm bilgisi olmadığı için mm ölçülemez. Kullanılmayan: {names}.',
  no_dicom_files: 'Seçilen dosyaların hiçbiri DICOM görüntüsü olamaz ({n} dosya atlandı: gizli dosya, metin, resim ya da arşiv).',
  more_files: '+{n} dosya daha',
  clear_arm: '{n} çalışma silinsin mi?', clear_title: 'Sunucuda saklanan bütün çalışmaları sil',
  clear_confirm: '{n} çalışmanın tümü sunucudan silinecek (önizlemeleri ve etiketleriyle). Bu işlem geri alınamaz. Onaylamak için “{n} çalışma silinsin mi?” düğmesine bir kez daha tıklayın; tıklamazsanız hiçbir şey silinmez.',
  clear_unopened: '(Bunların {k} tanesi bir veri setinde önden analiz edildi ve henüz açılmadı; bu yüzden listede görünmüyor.)',
  no_recent_opened: 'Henüz açılan çalışma yok ({k} çalışma bir veri setinde önden analiz edildi, açılmayı bekliyor).',
  clear_nothing: 'Sunucuda saklanan çalışma yok.',
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
  badge_brightened: 'meme ucu parlatmayla bulundu',
  badge_brightened_title: 'Görüntünün kendisinde bulunamadı; memenin ön ucu parlatılınca bulundu. Konumunu kontrol edin.',
  badge_estimated: 'meme ucu tahmini (kontur)',
  badge_estimated_title: 'Meme ucu bulunamadı: segmentasyon ölçümü meme dokusu konturunun pektoral çizgiye (MLO) ya da göğüs duvarına (CC) en uzak noktasından alındı. Karara tahmini değer olarak girer.',
  est_short: 'tahmini', est_title: 'tahmini — meme ucu bulunamadı (kontur noktası)',
  card_estimated: 'Tahmini — {views} görüntüsünde meme ucu bulunamadı; meme dokusu konturunun en uzak noktasından ölçüldü.',
  notice_estimated_MLO: '{view}: meme ucu bulunamadı — segmentasyon PNL’si meme dokusu konturunun pektoral çizgiye en uzak noktasından tahmin edildi.',
  notice_estimated_CC: '{view}: meme ucu bulunamadı — segmentasyon mesafesi meme dokusu konturunun göğüs duvarına en uzak noktasından tahmin edildi.',

  right_breast: 'Sağ meme', left_breast: 'Sol meme', no_views: 'görüntü yok',
  nothing_uploaded: 'Bu meme için görüntü yüklenmedi.',
  method_landmark: 'Nokta bazlı', method_mask: 'Segmentasyon',
  verdict_good: 'Yeterli', verdict_bad: 'Yetersiz', no_measurement: 'ölçüm yok',
  pnl_mlo: 'PNL · MLO', wall_cc: 'Meme ucu → duvar · CC', threshold: 'Eşik',
  cmp_agree: 'İki yöntem aynı kararı verdi.', cmp_disagree: 'Yöntemler kararda ayrışıyor — çalışmayı gözden geçirin.',
  cmp_wide: 'Karar aynı, ancak ölçümler {gap} mm ayrışıyor.',
  cmp_partial: 'Yalnız bir yöntem sonuç üretti.', cmp_none: 'İki yöntem de sonuç üretmedi.',

  criteria_title: 'Pozisyonlama kriterleri',
  criteria_counts: '{g} iyi · {a} kabul edilebilir · {p} kötü', criteria_counts_gp: '{g} iyi · {p} kötü',
  criteria_empty: 'Bu çalışma için henüz kriter yok — kriterler segmentasyon maskelerinden hesaplanır; maskesi olmayan görüntüde kullanılabilir kontur yoktur.',
  grade_good: 'iyi', grade_acceptable: 'kabul edilebilir', grade_poor: 'kötü', grade_na: 'ölçülemedi', grade_info: 'bilgi',
  mlo_quality_label: 'MLO kalitesi', mlo_quality_good: 'iyi', mlo_quality_poor: 'kötü',
  notes: 'Notlar',
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
  key_middle: 'Orta tuşla sürükle', key_pan_drag: 'Sağ ya da orta tuşla sürükle', key_h: 'H',
  k_hide: 'Konturları gizle / göster',
  ed_hide: 'Konturları gizle', ed_show: 'Konturları göster',
  ed_hide_title: 'Altındaki görüntüyü görmek için meme, pektoral ve meme ucu konturlarını gizleyin (H). Yalnız görünüm değişir; etiketler olduğu gibi kalır.',
  hint_hidden: 'Konturlar gizli (H yeniden gösterir): bu sırada yalnız noktalar düzenlenebilir.',
  st_hidden: 'gizli',
  ed_clean: 'Etiketleri yumuşat / temizle',
  ed_clean_title: 'Temizleme kurallarını bu görüntüye uygular: her yapının en büyük bölgesi kalır, delikler doldurulur, kopuk pikseller en yakın yapıya geçer ve kenarlar yumuşatılır. Kaydedilmemiş bir düzenlemedir; Geri al ile geri alınır.',
  ed_cleaning: 'Temizleniyor…', ed_clean_none: 'Bu görüntüde temizlenecek bir şey yok: konturlar kurallara zaten uyuyor.',
  ed_clean_done: 'Temizlendi: {classes}. Henüz kaydedilmedi; Geri al ile geri alınır.',
  ed_clean_done_view: '{view} temizlendi: {classes}. Henüz kaydedilmedi; Geri al ile geri alınır.',
  ed_clean_none_view: '{view} görüntüsünde temizlenecek bir şey yok: konturlar kurallara zaten uyuyor.',
  ed_clean_stale: '{view} temizlenirken düzenlendi; bu yüzden temizleme uygulanmadı. Şimdiki haliyle temizlemek için yeniden tıklayın.',
  hint_select: 'Taşımak için bir tutamacı sürükleyin. Shift+tık kenara köşe ekler, Alt+tık köşeyi siler.',
  hint_polygon: 'Köşe eklemek için tıklayın. Enter, çift tık ya da ilk noktaya tık şekli kapatır ve mevcut şeklin yerine geçer.',
  hint_point: 'Seçili noktayı yerleştirmek için tıklayın. Alt+tık siler.',
  cls_breast: 'Meme dokusu', cls_pectoral: 'Pektoral kas', cls_nipple: 'Meme ucu konturu',
  cls_nipple_point: 'Meme ucu noktası', cls_pec_top: 'Pektoral üst', cls_pec_bottom: 'Pektoral alt',
  st_pts: '{n} nokta', st_none: 'yok', st_set: 'var', edited_mark: 'Düzenlendi',
 
  discard_confirm: 'Kaydedilmemiş etiket değişiklikleri silinsin mi?',
  zoom_in: 'Yakınlaştır', zoom_out: 'Uzaklaştır', fit: 'Sığdır',

  lightbox_aria: 'Büyütülmüş görüntü', lightbox_hint: 'Kaydırmak için sürükleyin · Yakınlaştırmak için tekerlek · Kapatmak için Esc',
  views_n: '{n} görüntü', flag_title: 'Yöntemler kararda ayrışıyor', flag_wide_title: 'Karar aynı, ancak ölçümler ayrışıyor',
  flag_partial_title: 'Yalnız bir yöntem sonuç üretti',

  err_timeout: 'Sunucu zamanında yanıt vermedi. Hâlâ çalışıyor mu?',
  err_network: 'Sunucuya ulaşılamadı. Servis çalışıyor mu?',
  err_file_unreadable: 'Tarayıcı şu dosyayı okuyamadı: {file}. Dosya taşınmış ya da değişmiş olabilir.',
  err_invalid: 'Geçersiz istek: {msg}', err_http: 'İstek başarısız (HTTP {status}).',
};

// Criterion names and descriptions, keyed by the criterion id the server sends. English falls back to the
// server's own text, so only the translations are listed.
const CRITERIA_TR = {
  pectoral_angle: 'Pektoral açı',
  nipple_centred: 'Meme ucu ortalanmış',
  mlo_quality: 'MLO kalitesi',
  gantry: 'C-kol açısı',
};

// The server's explanations of a criterion, by the code it sends with each one (criteria.py DETAILS); the numbers come
// as parameters. English is the server's own sentence. tests/test_positioning.py checks that every code is here.
const DETAILS_TR = {
  pectoral_angle: 'oturtulan pektoral çizginin dikeyle açısı',
  no_pectoral_line: 'pektoral çizgisi yok',
  mlo_inside: 'PNL pektoral çizgiyle görüntünün içinde kesişiyor',
  mlo_outside: 'PNL pektoral çizgiyle görüntünün dışında kesişiyor',
  mlo_outside_by: 'PNL pektoral çizgiyle görüntünün dışında kesişiyor, kenarın {mm} mm ötesinde',
  needs_line_and_nipple: 'pektoral çizgi ve meme ucu gerekli',
  no_geometry: 'bu çalışma için görüntü geometrisi kayıtlı değil',
  nipple_centred: 'meme yüksekliğinin ortasına uzaklık (yarı yüksekliğe oranı: %{share})',
  needs_nipple_and_breast: 'meme ucu ve meme dokusu gerekli',
  no_breast_tissue: 'meme dokusu yok',
  from_header: 'DICOM başlığından',
};

// The server's refusals, by the code it sends with each one (service.py ERRORS); numbers and names come as
// parameters. English is the server's own sentence. tests/test_positioning_app.py checks that every code is here.
const ERRORS_TR = {
  study_not_found: 'Çalışma bulunamadı.',
  image_not_found: 'Görüntü bulunamadı.',
  models_loading: 'Modeller hâlâ yükleniyor — biraz sonra yeniden deneyin.',
  models_missing: 'Modeller yüklü değil — uygulamanın sorumlusuna haber verin.',
  no_files: 'Dosya gönderilmedi.',
  too_many_files: 'Bir seferde en çok {limit} dosya yüklenebilir.',
  nothing_processed: 'Hiçbir görüntü işlenemedi.',
  only_dicom: 'Bu uygulama yalnız DICOM (.dcm) okur; PNG/JPG’de piksel aralığı ve görünüm bilgisi olmadığı için mm ölçülemez. Kullanılmayan: {names}.',
  clear_changed: 'Şu anda {stored} çalışma saklanıyor, {expected} değil; hiçbir şey silinmedi. Listeyi yenileyip yeniden deneyin.',
};

// Why one file of an upload was not used (service.py PROBLEMS); listed under "nothing_processed", and kept with a
// stored study as its upload notes (`problem_codes`, see uploadNote).
const PROBLEMS_TR = {
  not_an_image: '{file}: atlandı — .{suffix} dosyası, görüntü değil',
  raster_image: '{file}: atlandı — bu uygulama yalnız DICOM (.dcm) okur; PNG/JPG’de piksel aralığı ve görünüm bilgisi olmadığı için mm ölçülemez.',
  not_dicom: '{file}: DICOM olarak okunamadı ({why})',
  undecodable: '{file}: piksel verisi çözülemedi — bir sıkıştırma çözücüsü eksik olabilir ({why})',
  not_single_frame: '{file}: tek kareli, gri tonlu bir görüntü bekleniyordu',
  view_unknown: '{file}: CC mi MLO mu anlaşılamadı. Başlıkta görünümü bildiren ViewPosition, ViewCodeSequence ya da seri açıklaması yok — dosyanın adına görünümü yazın, ör. L-MLO.dcm ya da R_CC.dcm.',
  side_unknown: '{file}: sol mu sağ mı anlaşılamadı. Başlıkta ImageLaterality ya da Laterality yok — dosyanın adına tarafı yazın, ör. L-MLO.dcm.',
  view_unsupported: '{file}: ViewPosition “{view}” — yalnız MLO ve CC desteklenir',
  file_too_large: '{file}: dosya çok büyük',
  duplicate_view: '{file}: {view} zaten {first} olarak yüklendi — atlandı',
  analysis_refused: '{file}: analiz yapılamadı ({why})',
  inference_failed: '{file}: analiz başarısız oldu ({why})',
};

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
// Every start is in the server's default language (English unless configured). The EN / TR switch holds for this
// browser session (this tab or window, through a reload, until it is closed), like the other apps of the project;
// an older version remembered it for good, and that key is removed.
const SESSION_KEY = 'mammo.lang.session.v1';
const OLD_STORAGE_KEY = 'mammo.lang.v1';
const listeners = new Set();
let current = 'en';
try { localStorage.removeItem(OLD_STORAGE_KEY); } catch (error) { /* blocked storage: nothing was kept either */ }

export const LANGUAGES = Object.keys(LANGS);

export function language() { return current; }

export function t(key, vars = {}, fallback) {
  const table = LANGS[current] || EN;
  const text = table[key] ?? EN[key] ?? fallback ?? key;
  return String(text).replace(/\{(\w+)\}/g, (_, name) => (name in vars ? vars[name] : `{${name}}`));
}

/**
 * A number as the interface language writes it: 75,6 and −47,2 in Turkish, 75.6 and -47.2 in English.
 * ``digits === null`` keeps the number as it is (a threshold of 10 stays 10).
 */
export function formatNumber(value, digits = 1) {
  const number = Number(value);
  const text = digits === null ? String(number) : number.toFixed(digits);
  return current === 'tr' ? text.replace('.', ',').replace('-', '\u2212') : text;
}

/** A value with its own unit: a space before the unit (75,6 mm, 98,0 N), none before the degree sign (18,7°). */
export function formatValue(value, unit = '', digits = 1) {
  const number = formatNumber(value, digits);
  return !unit || unit === '°' ? `${number}${unit}` : `${number} ${unit}`;
}

/** Millimetres, with a space before the unit. */
export function formatMm(value, digits = 1) {
  return formatValue(value, 'mm', digits);
}

export function criterionName(item) {
  return current === 'tr' && CRITERIA_TR[item.id] ? CRITERIA_TR[item.id] : item.name;
}

export function criterionDetail(item) {
  const detail = item.detail || '';
  const template = current === 'tr' ? DETAILS_TR[item.code] : undefined;
  if (!template) return detail; // English, or a record stored before the codes existed
  const params = item.params || {};
  return template.replace(/\{(\w+)\}/g, (_, name) => (
    name in params ? formatNumber(params[name], name === 'mm' ? 1 : 0) : `{${name}}`));
}

const fill = (template, params) => template.replace(/\{(\w+)\}/g, (_, name) => (
  name in params ? String(params[name]) : `{${name}}`));

/**
 * A refusal of the server as the reader's language says it: `payload` is the error answer ({ detail, code?, params? }).
 * English, an answer without a code and a code this page does not know all give the server's own sentence.
 */
export function serverError(payload) {
  const detail = payload && typeof payload.detail === 'string' ? payload.detail : '';
  const template = current === 'tr' && payload ? ERRORS_TR[payload.code] : undefined;
  if (!template) return detail;
  const params = payload.params || {};
  const lines = [fill(template, params)];
  for (const file of Array.isArray(params.files) ? params.files : []) { // one line per file that was not used
    const reasonOf = file && PROBLEMS_TR[file.code];
    lines.push(reasonOf ? fill(reasonOf, file.params || {}) : String((file && file.text) || ''));
  }
  return lines.join('\n');
}

/**
 * One upload note of a stored study in the reader's language: `note` is the stored English sentence, `entry` the
 * code and parameters stored with it ({ code, params }). A note stored without a code, English, and a code this page
 * does not know give the sentence as it is.
 */
export function uploadNote(note, entry) {
  const template = current === 'tr' && entry ? PROBLEMS_TR[entry.code] : undefined;
  return template ? fill(template, entry.params || {}) : String(note || '');
}

export function reason(text) {
  if (!text) return t('no_measurement');
  return current === 'tr' ? (REASONS_TR[text] || text) : text;
}

export function locale() { return current === 'tr' ? 'tr-TR' : 'en-GB'; }

/** The language picked in this browser session, else `fallback` (the server's default). */
export function savedLanguage(fallback = 'en') {
  try {
    const picked = sessionStorage.getItem(SESSION_KEY);
    return picked && LANGS[picked] ? picked : fallback;
  } catch (error) { return fallback; }
}

/**
 * Apply a language to the static markup and tell every listener; `remember` makes it the reader's choice for this
 * browser session (the next start opens in the default language again).
 */
export function setLanguage(lang, { remember = false } = {}) {
  current = LANGS[lang] ? lang : 'en';
  document.documentElement.lang = current;
  if (remember) {
    try { sessionStorage.setItem(SESSION_KEY, current); } catch (error) { /* blocked storage: this page only */ }
  }
  for (const node of document.querySelectorAll('[data-i18n]')) node.textContent = t(node.dataset.i18n);
  for (const node of document.querySelectorAll('[data-i18n-title]')) node.title = t(node.dataset.i18nTitle);
  for (const node of document.querySelectorAll('[data-i18n-aria]')) node.setAttribute('aria-label', t(node.dataset.i18nAria));
  for (const listener of listeners) listener(current);
}

export function onLanguage(listener) { listeners.add(listener); }

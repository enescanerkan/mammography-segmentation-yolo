"""Pure functions of the review screen, run as JavaScript in headless Chromium (they import browser modules).

* ``labelSignature`` (editor.js): an edit undone or a vertex dragged back is no change, so the editor does not ask to
  discard and does not send the view to be marked as edited.
* ``formatNumber`` / ``formatMm`` / ``formatValue`` (i18n.js): a decimal comma and a minus sign in Turkish, a space
  before a unit, and the criterion's own unit rather than millimetres for everything.
* ``criterionDetail`` (i18n.js): translated by the code and parameters the server sends, not by its English wording.
* ``buildDataset`` (dataset.js): a DICOMDIR, a DICOM file with no study UID and one with no pixel data are skipped
  rather than opened as one-file studies; two studies in one folder get different names.
* ``studyKey`` / ``sha256`` (dataset.js): the key the page makes is the HMAC-SHA-256 Python makes from the same salt.
* ``studyUpload`` (dataset.js): past the server's limit the files whose path names a view go first, and the caller
  learns how many stayed behind.
* ``drawOverlay`` (overlay.js): when the segmentation route measured nothing, the line and the perpendicular the MLO
  quality was judged from are still drawn.
* ``formatNumber`` against ``numbers.fixed``: the interface and the report round every number to the same digits.
* ``serverError`` (i18n.js): a refusal of the server is translated by its code and parameters; English, an answer
  without a code and an unknown code show the server's sentence.
* ``setLanguage`` / ``savedLanguage`` (i18n.js): the EN / TR choice holds for the browser session and is never stored
  for good; a choice an older version stored is dropped. ``uploadNote``: a stored study's upload notes are
  translated by their codes.
* ``drawOverlay`` (overlay.js): a distance measured without a nipple (the contour estimate) is dashed and says "est.".

Skipped without a headless Chromium (``$BREAST_SEG_CHROMIUM`` or a Playwright download).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json

import pytest

from .test_dataset_js import _chromium, _dataset, _dicomdir, _file, run_in_chromium

BODY = """
const { labelSignature } = await import(ORIGIN + '/js/editor.js');
const i18n = await import(ORIGIN + '/js/i18n.js');
const loaded = { view_key: 'L-MLO', points: { nipple: [300, 320] },
  polygons: { nipple: { points: [[1, 2], [3, 4], [5, 6]], confidence: 0.4, rescued: false } } };
const copy = () => JSON.parse(JSON.stringify(loaded));
const moved = copy(); moved.polygons.nipple.points[1] = [9, 9]; moved.polygons.nipple.edited = true;
const back = copy(); back.polygons.nipple.edited = true;           // dragged away and back: only the flag differs
const reordered = { view_key: 'L-MLO', polygons: { nipple: { rescued: false, confidence: 0.4,
  points: [[1, 2], [3, 4], [5, 6]] } }, points: { nipple: [300, 320] } };
const pointMoved = copy(); pointMoved.points.nipple = [301, 320];
const out = {
  same: labelSignature(copy()) === labelSignature(loaded),
  moved: labelSignature(moved) === labelSignature(loaded),
  back: labelSignature(back) === labelSignature(loaded),
  reordered: labelSignature(reordered) === labelSignature(loaded),
  pointMoved: labelSignature(pointMoved) === labelSignature(loaded),
};
i18n.setLanguage('en');
out.en = [i18n.formatNumber(75.6), i18n.formatNumber(-47.2), i18n.formatMm(6.78), i18n.formatMm(10, null)];
i18n.setLanguage('tr');
out.tr = [i18n.formatNumber(75.6), i18n.formatNumber(-47.2), i18n.formatMm(6.78), i18n.formatMm(12.5, null)];
out.units = [i18n.formatValue(98, 'N'), i18n.formatValue(12.34, 'kPa'), i18n.formatValue(18.7, '°'),
  i18n.formatValue(1.5, 'mGy'), i18n.formatValue(40, '%'), i18n.formatValue(3, '')];
const outside = { code: 'mlo_outside_by', params: { mm: 3 }, detail: 'ENGLISH' };
const centred = { code: 'nipple_centred', params: { share: 25 }, detail: 'ENGLISH' };
out.details = [i18n.criterionDetail(outside), i18n.criterionDetail(centred),
  i18n.criterionDetail({ code: 'not_a_code', detail: 'as the server wrote it' }),
  i18n.criterionDetail({ detail: 'a record stored before the codes' })];
i18n.setLanguage('en');
out.english = i18n.criterionDetail(outside);

// buildDataset on files built from real DICOM bytes (pydicom), with folder paths as a folder pick gives them
const { buildDataset } = await import(ORIGIN + '/js/dataset.js');
const file = (path, b64) => {
  const raw = atob(b64);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i += 1) bytes[i] = raw.charCodeAt(i);
  const made = new File([bytes], path.split('/').pop());
  Object.defineProperty(made, 'webkitRelativePath', { value: path });
  return made;
};
const built = await buildDataset(FILES.map(([path, b64]) => file(path, b64)));
out.dataset = { name: built.name, skipped: built.skipped,
  studies: built.studies.map((study) => [study.name, study.files.length]) };
out.uids = built.studies.map((study) => study.uid);
return out;
"""


def _dataset_files():
    """A picked folder: two studies in one patient folder, one in its own, and three files that are not views."""
    def image(study):
        ds = _dataset(nested=False, pixels=True)
        ds.StudyInstanceUID = study
        return _file(ds)

    files = {
        "set/DICOMDIR": _dicomdir(),
        "set/patientA/IM1.dcm": image("1.2.3.1"), "set/patientA/IM2.dcm": image("1.2.3.1"),
        "set/patientA/IM3.dcm": image("1.2.3.2"), "set/patientA/IM4.dcm": image("1.2.3.2"),
        "set/patientB/exam/IM1.dcm": image("1.2.3.3"),
        "set/patientB/exam/SR1.dcm": _file(_dataset(nested=False)),                    # a report: no pixel data
        "set/patientB/exam/XX1.dcm": _file(_dataset(nested=False, study=False, pixels=True)),   # no study UID
        "set/notes.txt": b"hello",
    }
    return [[path, base64.b64encode(data).decode()] for path, data in files.items()]


def test_editor_dirtiness_and_number_formats(tmp_path):
    chrome = _chromium()
    if not chrome:
        pytest.skip("needs a headless Chromium to run the browser modules")
    out = run_in_chromium(chrome, f"const FILES = {json.dumps(_dataset_files())};\n{BODY}", tmp_path)
    assert out["same"] and out["back"] and out["reordered"]           # no change
    assert not out["moved"] and not out["pointMoved"]                   # a real change
    assert out["en"] == ["75.6", "-47.2", "6.8 mm", "10 mm"]
    assert out["tr"] == ["75,6", "\u221247,2", "6,8 mm", "12,5 mm"]
    # the criterion's own unit, never millimetres for everything
    assert out["units"] == ["98,0 N", "12,3 kPa", "18,7°", "1,5 mGy", "40,0 %", "3,0"]
    # translated by code and parameters; an unknown code or an old record falls back to the server's sentence
    assert out["details"] == ["PNL pektoral çizgiyle görüntünün dışında kesişiyor, kenarın 3,0 mm ötesinde",
                              "meme yüksekliğinin ortasına uzaklık (yarı yüksekliğe oranı: %25)",
                              "as the server wrote it", "a record stored before the codes"]
    assert out["english"] == "ENGLISH"
    # the DICOMDIR, the report, the file with no study UID and the text file are skipped, not opened as studies;
    # the two studies that share patientA are told apart by their first file
    assert out["dataset"] == {"name": "set", "skipped": 4,
                              "studies": [["patientA · IM1.dcm", 2], ["patientA · IM3.dcm", 2], ["patientB/exam", 1]]}
    assert out["uids"] == ["1.2.3.1", "1.2.3.2", "1.2.3.3"]            # kept per study, for the study key only


KEY_BODY = """
const { sha256, studyKey, studyUpload } = await import(ORIGIN + '/js/dataset.js');
const hex = (bytes) => [...bytes].map((byte) => byte.toString(16).padStart(2, '0')).join('');
const sized = (size) => ({ size });
const named = (path) => ({ name: path.split('/').pop(), webkitRelativePath: path, size: 1 });
const out = {
  digests: LENGTHS.map((length) => hex(sha256(new Uint8Array(length).map((_, i) => (i * 7 + 3) % 256)))),
  keys: CASES.map(([uid, sizes]) => studyKey(SALT, uid, sizes.map(sized))),
  none: [studyKey('', '1.2.3'), studyKey('abc', '1.2.3'), studyKey(SALT, ''), studyKey(SALT.toUpperCase(), '1.2.3')],
};
const paths = ['s/a/IM1.dcm', 's/b/IM2.dcm', 's/RMLO/IM3.dcm', 's/c/IM4.dcm', 's/L_CC/IM5.dcm'];
const pick = (limit) => { const sent = studyUpload(paths.map(named), limit);
  return [sent.files.map((file) => file.webkitRelativePath), sent.left]; };
out.uploads = [pick(5), pick(3), pick(1)];
return out;
"""


def test_the_study_key_is_the_hmac_python_makes_and_the_upload_prefers_views(tmp_path):
    chrome = _chromium()
    if not chrome:
        pytest.skip("needs a headless Chromium to run the browser modules")
    salt = "0f" * 16 + "a1" * 16
    lengths = [0, 1, 55, 56, 63, 64, 65, 119, 120, 1000]              # around the padding and block boundaries
    cases = [["1.2.3.4.5.6.7.8.9.10", [100, 250]], ["1.2.3", []], ["9" * 64, [7] * 41], ["1.2.3", [350]]]
    body = (f"const SALT = {json.dumps(salt)}; const LENGTHS = {json.dumps(lengths)};"
            f" const CASES = {json.dumps(cases)};\n{KEY_BODY}")
    out = run_in_chromium(chrome, body, tmp_path)
    assert out["digests"] == [hashlib.sha256(bytes((i * 7 + 3) % 256 for i in range(n))).hexdigest() for n in lengths]

    def key(uid, sizes):
        message = f"{uid}\n{len(sizes)}\n{sum(sizes)}".encode()
        return hmac.new(bytes.fromhex(salt), message, hashlib.sha256).hexdigest()

    assert out["keys"] == [key(uid, sizes) for uid, sizes in cases]
    assert out["keys"][1] != out["keys"][3]                           # the same study with other files: another key
    assert out["none"] == ["", "", "", ""]                            # no salt, a bad salt or no UID: no key
    paths = ["s/a/IM1.dcm", "s/b/IM2.dcm", "s/RMLO/IM3.dcm", "s/c/IM4.dcm", "s/L_CC/IM5.dcm"]
    assert out["uploads"][0] == [paths, 0]                            # within the limit: everything, in order
    assert out["uploads"][1] == [["s/RMLO/IM3.dcm", "s/L_CC/IM5.dcm", "s/a/IM1.dcm"], 2]   # views first
    assert out["uploads"][2] == [["s/RMLO/IM3.dcm"], 4]


OVERLAY_BODY = """
const { drawOverlay } = await import(ORIGIN + '/js/overlay.js');
const drawn = (image) => {
  const calls = [];
  const ctx = new Proxy({}, {
    get: (target, name) => (name in target ? target[name] : (...args) => { calls.push([name, ...args]); }),
    set: (target, name, value) => { target[name] = value; return true; },
  });
  ctx.measureText = () => ({ width: 10 });
  drawOverlay(ctx, image, new Set(['measure_mask']), 1);
  const segments = [];
  calls.forEach((call, index) => {
    if (call[0] === 'lineTo' && calls[index - 1][0] === 'moveTo') segments.push([...calls[index - 1].slice(1), ...call.slice(1)]);
  });
  return { segments, texts: calls.filter((call) => call[0] === 'fillText').length };
};
const line = [[100, 0], [300, 400]];
const basis = { line, nipple: [400, 300], foot: [240, 280] };
const quality = [{ id: 'mlo_quality', grade: 'good', basis }];
const missing = { ok: false, reason: 'no nipple mask', line };
const measured = { ok: true, mm: 80, line, from: [400, 300], to: [240, 280] };
return {
  graded_without_measure: drawn({ measures: { mask: missing }, criteria: quality }),
  line_only: drawn({ measures: { mask: missing }, criteria: [{ id: 'mlo_quality', grade: 'na' }] }),
  nothing: drawn({ measures: { mask: { ok: false } }, criteria: [] }),
  measured: drawn({ measures: { mask: measured }, criteria: quality }),
};
"""


def test_the_overlay_draws_what_the_mlo_quality_was_judged_from(tmp_path):
    chrome = _chromium()
    if not chrome:
        pytest.skip("needs a headless Chromium to run the browser modules")
    out = run_in_chromium(chrome, OVERLAY_BODY, tmp_path)
    line, perpendicular = [100, 0, 300, 400], [400, 300, 240, 280]
    # no distance was measured, the quality was graded: its line and its perpendicular are drawn, with no mm label
    assert out["graded_without_measure"] == {"segments": [line, perpendicular], "texts": 0}
    assert out["line_only"] == {"segments": [line], "texts": 0}       # only the angle came from the line
    assert out["nothing"] == {"segments": [], "texts": 0}
    # a measured distance is the basis itself (the quality uses the nipple and the line it was measured from):
    # drawn once, with its millimetres
    assert out["measured"] == {"segments": [line, perpendicular], "texts": 1}


ROUNDING_BODY = """
const i18n = await import(ORIGIN + '/js/i18n.js');
i18n.setLanguage('en');
return VALUES.map(([value, digits]) => i18n.formatNumber(value, digits));
"""


def test_the_interface_and_the_report_round_a_number_to_the_same_digits(tmp_path):
    """One measurement is written by the page (tile, card, overlay, drawer) and by the server (report, notes): both
    round the exact value half away from zero, so 47.949 mm is 47.9 mm in both and a tie like 47.25 goes the same way."""
    import random

    from breast_seg.positioning.numbers import fixed

    chrome = _chromium()
    if not chrome:
        pytest.skip("needs a headless Chromium to run the browser modules")
    rng = random.Random(7)
    values = [(47.949377, 1), (47.95, 1), (47.25, 1), (47.75, 1), (-47.25, 1), (0.05, 1), (2.5, 0), (24.5, 0),
              (1.005, 2), (0.125, 2), (-0.04, 1), (63.54400961980653, 1), (10, 1), (0, 1)]
    values += [(rng.uniform(-200, 200), rng.choice((0, 1, 2))) for _ in range(300)]
    values += [(rng.randrange(-4000, 4000) / 20, 1) for _ in range(300)]        # on and next to the x.x5 ties
    values += [(rng.randrange(-4000, 4000) / 8, rng.choice((0, 1, 2))) for _ in range(300)]   # exact binary ties
    out = run_in_chromium(chrome, f"const VALUES = {json.dumps(values)};\n{ROUNDING_BODY}", tmp_path)
    assert out[:3] == ["47.9", "48.0", "47.3"]
    assert out == [fixed(value, digits) for value, digits in values]


REFUSAL_BODY = """
const i18n = await import(ORIGIN + '/js/i18n.js');
const said = (lang) => { i18n.setLanguage(lang); return ANSWERS.map((answer) => i18n.serverError(answer)); };
return { tr: said('tr'), en: said('en') };
"""


def test_a_refusal_is_translated_by_its_code_and_falls_back_to_the_servers_sentence(tmp_path):
    chrome = _chromium()
    if not chrome:
        pytest.skip("needs a headless Chromium to run the browser modules")
    answers = [
        {"detail": "Study not found.", "code": "study_not_found", "params": {}},
        {"detail": "3 studies are stored now, not 2; nothing was deleted. Refresh the list and try again.",
         "code": "clear_changed", "params": {"stored": 3, "expected": 2}},
        {"detail": "At most 40 files can be uploaded at once.", "code": "too_many_files", "params": {"limit": 40}},
        {"detail": "No image could be processed.\nnotes.txt: skipped — a .txt file, not an image\nnew.dcm: a new reason"
                   "\nold.dcm: a line without a code",
         "code": "nothing_processed", "params": {"files": [
             {"text": "notes.txt: skipped — a .txt file, not an image", "code": "not_an_image",
              "params": {"file": "notes.txt", "suffix": "txt"}},
             {"text": "new.dcm: a new reason", "code": "a_code_this_page_does_not_know", "params": {"file": "new.dcm"}},
             {"text": "old.dcm: a line without a code"}]}},
        {"detail": "A refusal of a newer server.", "code": "a_code_this_page_does_not_know", "params": {}},
        {"detail": "The study key must be 64 hexadecimal characters."},
    ]
    out = run_in_chromium(chrome, f"const ANSWERS = {json.dumps(answers)};\n{REFUSAL_BODY}", tmp_path)
    assert out["en"] == [answer["detail"] for answer in answers]       # English is the server's own sentence
    assert out["tr"] == [
        "Çalışma bulunamadı.",
        "Şu anda 3 çalışma saklanıyor, 2 değil; hiçbir şey silinmedi. Listeyi yenileyip yeniden deneyin.",
        "Bir seferde en çok 40 dosya yüklenebilir.",
        "Hiçbir görüntü işlenemedi.\nnotes.txt: atlandı — .txt dosyası, görüntü değil\nnew.dcm: a new reason"
        "\nold.dcm: a line without a code",
        "A refusal of a newer server.",
        "The study key must be 64 hexadecimal characters.",
    ]



SESSION_BODY = """
const i18n = await import(ORIGIN + '/js/i18n.js');
const { drawOverlay } = await import(ORIGIN + '/js/overlay.js');
const out = {};
// the EN / TR switch holds for the session: nothing is written for the next start, and an older stored choice goes
localStorage.setItem('mammo.lang.v1', 'tr');
const fresh = await import(ORIGIN + '/js/i18n.js?fresh');
out.fresh = [fresh.savedLanguage('en'), localStorage.getItem('mammo.lang.v1')];
fresh.setLanguage('tr', { remember: true });
out.picked = [fresh.savedLanguage('en'), localStorage.getItem('mammo.lang.v1'), Object.keys(localStorage).length,
  Object.keys(sessionStorage).length];
// upload notes by code
i18n.setLanguage('tr');
out.notes_tr = [i18n.uploadNote('x.png: skipped …', { code: 'raster_image', params: { file: 'x.png' } }),
  i18n.uploadNote('y.dcm: a newer reason', { code: 'a_code_this_page_does_not_know', params: {} }),
  i18n.uploadNote('z.txt: stored before the codes', undefined)];
i18n.setLanguage('en');
out.notes_en = [i18n.uploadNote('x.png: skipped …', { code: 'raster_image', params: { file: 'x.png' } })];
// an estimated distance is dashed, starts at an open ring and says "est."; the MLO quality's own basis is drawn too
const drawn = (image) => {
  const calls = [];
  const ctx = new Proxy({}, {
    get: (target, name) => (name in target ? target[name] : (...args) => { calls.push([name, ...args]); }),
    set: (target, name, value) => { target[name] = value; return true; },
  });
  ctx.measureText = () => ({ width: 10 });
  drawOverlay(ctx, image, new Set(['measure_mask']), 1);
  return {
    dashed: calls.filter((call) => call[0] === 'setLineDash' && call[1].length).length,
    texts: calls.filter((call) => call[0] === 'fillText').map((call) => call[1]),
    strokes: calls.filter((call) => call[0] === 'stroke').length,
  };
};
const line = [[100, 0], [300, 400]];
const basis = { line, nipple: [400, 300], foot: [240, 280] };
const quality = [{ id: 'mlo_quality', grade: 'good', basis }];
out.estimated = drawn({ measures: { mask: { ok: true, mm: 80, line, from: [420, 350], to: [250, 300], estimated: true } },
  criteria: quality });
out.measured = drawn({ measures: { mask: { ok: true, mm: 80, line, from: [400, 300], to: [240, 280] } }, criteria: quality });
return out;
"""


def test_the_language_holds_for_one_page_notes_translate_and_estimates_are_drawn_as_such(tmp_path):
    chrome = _chromium()
    if not chrome:
        pytest.skip("needs a headless Chromium to run the browser modules")
    out = run_in_chromium(chrome, SESSION_BODY, tmp_path)
    assert out["fresh"] == ["en", None]                               # a choice stored by an older version is dropped
    assert out["picked"] == ["tr", None, 0, 1]                        # ... and a new one only for the session
    assert out["notes_tr"] == ["x.png: atlandı — bu uygulama yalnız DICOM (.dcm) okur; PNG/JPG’de piksel aralığı ve "
                               "görünüm bilgisi olmadığı için mm ölçülemez.",
                               "y.dcm: a newer reason", "z.txt: stored before the codes"]
    assert out["notes_en"] == ["x.png: skipped …"]                    # English is the stored sentence
    estimated, measured = out["estimated"], out["measured"]
    assert estimated["texts"] == ["80.0 mm est."] and estimated["dashed"] >= 3   # pectoral line, basis, estimate
    assert measured["texts"] == ["80.0 mm"] and measured["dashed"] == 1          # only the pectoral line

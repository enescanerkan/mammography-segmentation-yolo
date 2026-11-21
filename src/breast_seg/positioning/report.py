"""A written positioning report, generated from the numbers rather than from a language model.

Everything a report would say is already a structured measurement: a millimetre, a grade, a verdict. Nothing has
to be retrieved and nothing has to be inferred, so a template turns the record into sentences, and every sentence
can be traced back to the field it came from. A model asked to phrase this would sometimes round 27.6 to "about
28", sometimes to "nearly 30", and occasionally produce a number that is in no field at all; in a document that
grades a view, a wrong number is worse than a plain sentence. The report states the quality; it does not advise
repeating a view.

Two languages, one set of facts: the strings differ, the numbers do not.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .criteria import INFO
from .numbers import fixed
from .study import view_rank

LANGUAGES = ("en", "tr")
DEFAULT_LANGUAGE = "en"

WORDS: Dict[str, Dict[str, Any]] = {
    "en": {
        "title": "Mammography Positioning Report",
        "study": "Study",
        "date": "Date",
        "views": "Views",
        "device": "Device",
        "threshold": "ΔD threshold",
        "verdicts": "Verdict",
        "left": "Left breast",
        "right": "Right breast",
        "landmark": "Landmark",
        "mask": "Segmentation",
        "pnl": "PNL (MLO)",
        "ccd": "Nipple to chest wall (CC)",
        "delta": "ΔD",
        "good": "adequate",
        "bad": "inadequate",
        "agree": "Both methods agree.",
        "disagree": "The methods disagree on the verdict ({gap} mm apart) — review this study.",
        "wide": "Same verdict, but the measurements differ by {gap} mm.",
        "no_pair": "No measurement: {reason}.",
        "criteria": "Positioning criteria",
        "conclusion": "Conclusion",
        "all_good": "All criteria fall within acceptable limits.",
        "no_delta": "ΔD could not be computed for either breast — no measurable MLO–CC pair.",
        "review": "The findings below should be reviewed.",
        "grades": {"good": "good", "acceptable": "acceptable", "poor": "poor", "na": "n/a", "info": "information"},
        "mlo_quality": {"good": "good", "poor": "poor"},
        "edited": "Labels were corrected by a reviewer.",
        "brightened": "{view}: nipple found after brightening the front of the breast — check its position.",
        "estimated": {
            "MLO": "{view}: nipple not found — the segmentation PNL is estimated at the breast-tissue contour point "
                   "farthest from the pectoral line.",
            "CC": "{view}: nipple not found — the segmentation distance is estimated at the breast-tissue contour "
                  "point farthest from the chest wall.",
        },
        "estimated_mark": "estimated",
        "footer": "For research use; not for diagnostic purposes. Values marked information are shown ungraded.",
        "problems": "Upload notes",
        "decimal": ".",
        "minus": "-",
    },
    "tr": {
        "title": "Mamografi Pozisyonlama Raporu",
        "study": "Çalışma",
        "date": "Tarih",
        "views": "Görüntüler",
        "device": "Cihaz",
        "threshold": "ΔD eşiği",
        "verdicts": "Karar",
        "left": "Sol meme",
        "right": "Sağ meme",
        "landmark": "Nokta bazlı",
        "mask": "Segmentasyon",
        "pnl": "PNL (MLO)",
        "ccd": "Meme ucu – göğüs duvarı (CC)",
        "delta": "ΔD",
        "good": "yeterli",
        "bad": "yetersiz",
        "agree": "İki yöntem aynı kararı verdi.",
        "disagree": "Yöntemler kararda ayrışıyor ({gap} mm fark) — çalışmayı gözden geçirin.",
        "wide": "Karar aynı, ancak ölçümler {gap} mm ayrışıyor.",
        "no_pair": "Ölçüm yapılamadı: {reason}.",
        "criteria": "Pozisyonlama kriterleri",
        "conclusion": "Sonuç",
        "all_good": "Tüm kriterler yeterli sınırlar içinde.",
        "no_delta": "ΔD hiçbir meme için hesaplanamadı — ölçülebilen bir MLO–CC çifti yok.",
        "review": "Aşağıdaki bulgular gözden geçirilmelidir.",
        "grades": {"good": "iyi", "acceptable": "kabul edilebilir", "poor": "kötü", "na": "ölçülemedi",
                   "info": "bilgi"},
        "mlo_quality": {"good": "iyi", "poor": "kötü"},
        "edited": "Etiketler bir kullanıcı tarafından düzeltilmiştir.",
        "brightened": "{view}: meme ucu memenin ön ucu parlatılınca bulundu — konumunu kontrol edin.",
        "estimated": {
            "MLO": "{view}: meme ucu bulunamadı — segmentasyon PNL’si meme dokusu konturunun pektoral çizgiye en "
                   "uzak noktasından tahmin edildi.",
            "CC": "{view}: meme ucu bulunamadı — segmentasyon mesafesi meme dokusu konturunun göğüs duvarına en "
                  "uzak noktasından tahmin edildi.",
        },
        "estimated_mark": "tahmini",
        "footer": "Araştırma amaçlıdır; tanısal kullanım için değildir. Bilgi olarak işaretlenen değerler "
                  "derecelendirilmez.",
        "problems": "Yükleme notları",
        "decimal": ",",
        "minus": "\u2212",
    },
}



def _num(value: Any, words: Mapping[str, Any], digits: Optional[int] = 1) -> str:
    """A number as the report's language writes it: 75,6 and \u221247,2 in Turkish. ``digits=None`` drops trailing zeros.
    Rounded by the one rule every shown number follows (numbers.fixed), from the unrounded value the record holds."""
    text = f"{float(value):g}" if digits is None else fixed(value, digits)
    return text.replace(".", words["decimal"]).replace("-", words["minus"])


def _with_unit(number: str, unit: str) -> str:
    """A space before a unit (75,6 mm), none before the degree sign (18,7°)."""
    return number + (unit if unit in ("", "°") else " " + unit)


# Criterion names, kept here rather than in criteria.py so the measurement code never has to know which languages
# exist.
NAMES: Dict[str, Dict[str, str]] = {
    "en": {
        "pectoral_angle": "Pectoral angle",
        "nipple_centred": "Nipple centred",
        "mlo_quality": "MLO quality",
        "gantry": "Gantry angle",
    },
    "tr": {
        "pectoral_angle": "Pektoral açı",
        "nipple_centred": "Meme ucu ortalanmış",
        "mlo_quality": "MLO kalitesi",
        "gantry": "C-kol açısı",
    },
}

# Upload notes (why a file was not used), by the code the service stores with each one (``problem_codes``; the
# codes of service.PROBLEMS). English is the stored sentence itself; a note without a code, or with a code missing
# here, is written as stored. A test checks that every code has its Turkish line.
PROBLEMS: Dict[str, Dict[str, str]] = {
    "en": {},
    "tr": {
        "not_an_image": "{file}: atlandı — .{suffix} dosyası, görüntü değil",
        "raster_image": "{file}: atlandı — bu uygulama yalnız DICOM (.dcm) okur; PNG/JPG’de piksel aralığı ve "
                        "görünüm bilgisi olmadığı için mm ölçülemez.",
        "not_dicom": "{file}: DICOM olarak okunamadı ({why})",
        "undecodable": "{file}: piksel verisi çözülemedi — bir sıkıştırma çözücüsü eksik olabilir ({why})",
        "not_single_frame": "{file}: tek kareli, gri tonlu bir görüntü bekleniyordu",
        "view_unknown": "{file}: CC mi MLO mu anlaşılamadı. Başlıkta görünümü bildiren ViewPosition, "
                        "ViewCodeSequence ya da seri açıklaması yok — dosyanın adına görünümü yazın, ör. L-MLO.dcm "
                        "ya da R_CC.dcm.",
        "side_unknown": "{file}: sol mu sağ mı anlaşılamadı. Başlıkta ImageLaterality ya da Laterality yok — "
                        "dosyanın adına tarafı yazın, ör. L-MLO.dcm.",
        "view_unsupported": "{file}: ViewPosition “{view}” — yalnız MLO ve CC desteklenir",
        "file_too_large": "{file}: dosya çok büyük",
        "duplicate_view": "{file}: {view} zaten {first} olarak yüklendi — atlandı",
        "analysis_refused": "{file}: analiz yapılamadı ({why})",
        "inference_failed": "{file}: analiz başarısız oldu ({why})",
    },
}


def upload_notes(record: Mapping[str, Any], lang: str = DEFAULT_LANGUAGE) -> List[str]:
    """The record's upload notes in ``lang``: translated by their stored codes where the record has them (one per
    note, in the same order), the stored English sentence otherwise."""
    notes = [str(note) for note in record.get("problems") or []]
    coded = record.get("problem_codes")
    if not isinstance(coded, list) or len(coded) != len(notes):
        return notes  # stored before the codes were kept, or not by this service
    table = PROBLEMS.get(lang, {})
    out = []
    for note, entry in zip(notes, coded):
        template = table.get(entry.get("code")) if isinstance(entry, dict) else None
        params = entry.get("params") if isinstance(entry, dict) and isinstance(entry.get("params"), dict) else {}
        try:
            out.append(template.format(**params) if template else note)
        except (KeyError, IndexError, ValueError):
            out.append(note)
    return out


# The reasons the measurement code gives for a missing value are English; without this the Turkish report carries
# an English clause in the middle of a sentence.
REASONS: Dict[str, Dict[str, str]] = {
    "en": {},
    "tr": {
        "needs both MLO and CC": "MLO ve CC birlikte gerekli",
        "no measurement": "ölçüm yok",
        "nipple keypoint not detected": "meme ucu noktası bulunamadı",
        "pectoral keypoints not detected": "pektoral noktaları bulunamadı",
        "no nipple mask": "meme ucu maskesi yok",
        "no breast tissue mask": "meme dokusu maskesi yok",
        "nipple mask is empty": "meme ucu maskesi boş",
        "chest wall edge not found": "göğüs duvarı kenarı bulunamadı",
        "no pectoral mask": "pektoral maskesi yok",
        "could not fit a pectoral line": "pektoral çizgisi oturtulamadı",
    },
}


def build_report(record: Mapping[str, Any], lang: str = DEFAULT_LANGUAGE) -> Dict[str, Any]:
    """Study record -> ``{"lang", "title", "sections": [{"heading", "rows"}], "footer", "text"}``.

    A row is ``{"label", "value", "grade"?, "criterion"?, "strong"?}`` (``criterion`` is the id of a criterion row)
    or a free sentence ``{"text"}``; ``text`` is the whole report as plain text.
    """
    lang = lang if lang in LANGUAGES else DEFAULT_LANGUAGE
    words = WORDS[lang]
    names = NAMES[lang]
    sections: List[Dict[str, Any]] = [{"heading": words["title"], "rows": _header_rows(record, words)}]

    verdict_rows: List[Dict[str, Any]] = []
    sides = sorted((record.get("summary") or {}).get("sides", []), key=lambda side: side["laterality"] != "R")
    for side in sides:  # right breast first, as on the screen
        verdict_rows.extend(_side_rows(side, words, REASONS[lang]))
    sections.append({"heading": words["verdicts"], "rows": verdict_rows})

    sections.append({"heading": words["criteria"], "rows": _criteria_rows(record, words, names)})

    conclusion = _conclusion(record, words, names)
    sections.append({"heading": words["conclusion"], "rows": [{"text": line} for line in conclusion]})

    if record.get("problems"):
        sections.append({"heading": words["problems"], "rows": [{"text": p} for p in upload_notes(record, lang)]})

    return {
        "lang": lang,
        "title": words["title"],
        "sections": sections,
        "footer": words["footer"],
        "text": _as_text(sections, words, lang),
    }


def _ordered_images(record: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    return sorted(record.get("images") or [], key=lambda image: view_rank(image.get("view_key", "")))


def _header_rows(record: Mapping[str, Any], words: Mapping[str, Any]) -> List[Dict[str, Any]]:
    images = _ordered_images(record)
    created = record.get("created")
    device = next((i.get("manufacturer") for i in images if i.get("manufacturer")), "—")
    rows: List[Dict[str, Any]] = [
        {"label": words["study"], "value": record.get("id", "—")},
        {"label": words["date"],
         "value": time.strftime("%d.%m.%Y %H:%M", time.localtime(created)) if created else "—"},
        {"label": words["views"], "value": ", ".join(i["view_key"] for i in images) or "—"},
        {"label": words["device"], "value": device or "—"},
        {"label": words["threshold"],
         "value": _with_unit(_num((record.get("rules") or {}).get("threshold_mm", 10), words, None), "mm")},
    ]
    if record.get("edited"):
        rows.append({"text": words["edited"]})
    for image in images:
        if ((image.get("polygons") or {}).get("nipple") or {}).get("brightened"):
            rows.append({"text": words["brightened"].format(view=image["view_key"])})
        if (((image.get("measures") or {}).get("mask") or {}).get("estimated")
                and image.get("view") in words["estimated"]):
            rows.append({"text": words["estimated"][image["view"]].format(view=image["view_key"])})
    return rows


def _side_rows(side: Mapping[str, Any], words: Mapping[str, Any], reasons: Mapping[str, str]) -> List[Dict[str, Any]]:
    title = words["left"] if side["laterality"] == "L" else words["right"]
    rows: List[Dict[str, Any]] = [{"label": title, "value": " + ".join(side["views"]), "strong": True}]

    for route in ("landmark", "mask"):
        result = side.get(route) or {}
        if not result.get("ok"):
            reason = result.get("reason") or "—"
            rows.append({"label": words[route], "value": words["no_pair"].format(reason=reasons.get(reason, reason))})
            continue
        verdict = words["good"] if result["verdict"] == "good" else words["bad"]
        estimated = result.get("estimated") or []

        def mm(value: Any, view: str = "", estimated: Any = estimated) -> str:
            text = _with_unit(_num(value, words), "mm")
            return f"{text} ({words['estimated_mark']})" if view in estimated else text

        rows.append({
            "label": words[route],
            "value": "{} {} · {} {} · {} {} → {}".format(
                words["pnl"], mm(result["pnl_mm"], "MLO"), words["ccd"], mm(result["cc_mm"], "CC"),
                words["delta"], mm(result["delta_mm"]), verdict),
            "grade": result["verdict"],
        })

    comparison = side.get("comparison") or {}
    gap = _num(comparison.get("gap_mm") or 0.0, words)
    if comparison.get("state") == "disagree":
        rows.append({"text": words["disagree"].format(gap=gap)})
    elif comparison.get("wide_gap"):
        rows.append({"text": words["wide"].format(gap=gap)})
    elif comparison.get("state") == "agree":
        rows.append({"text": words["agree"]})
    return rows


def _value_word(entry: Mapping[str, Any], words: Mapping[str, Any]) -> str:
    """The value as written: a number with its unit, or a word in the report's language."""
    if entry["id"] == "mlo_quality" and entry["value"] in words["mlo_quality"]:
        return words["mlo_quality"][entry["value"]]
    if entry["value"] is None:
        return ""
    if isinstance(entry["value"], (int, float)) and not isinstance(entry["value"], bool):
        return _with_unit(_num(entry["value"], words), entry["unit"])
    return _with_unit(str(entry["value"]), entry["unit"])


def _criterion_value(entry: Mapping[str, Any], words: Mapping[str, Any]) -> str:
    value = _value_word(entry, words)
    if entry["grade"] == INFO or entry["id"] == "mlo_quality":
        return value  # information carries no grade word; the MLO quality's value already is its grade
    return "{} · {}".format(value, words["grades"][entry["grade"]]).strip(" ·")


def _criteria_rows(record: Mapping[str, Any], words: Mapping[str, Any],
                   names: Mapping[str, str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for image in _ordered_images(record):
        items = [entry for entry in image.get("criteria", []) if entry["grade"] != "na"]
        if not items:
            continue
        rows.append({"label": image["view_key"], "value": "", "strong": True})
        for entry in items:
            rows.append({"label": "   " + names.get(entry["id"], entry["name"]),
                         "value": _criterion_value(entry, words),
                         "grade": None if entry["grade"] == INFO else entry["grade"], "criterion": entry["id"]})
    return rows


def _conclusion(record: Mapping[str, Any], words: Mapping[str, Any], names: Mapping[str, str]) -> List[str]:
    """One sentence per failing thing, each traceable to a measured field."""
    lines: List[str] = []

    # Name the route whose number actually failed. Quoting one method's millimetres under the other's verdict is
    # how a report ends up saying "7.2 mm > 10 mm".
    measured = False
    for side in sorted((record.get("summary") or {}).get("sides", []), key=lambda side: side["laterality"] != "R"):
        title = words["left"] if side["laterality"] == "L" else words["right"]
        for route in ("landmark", "mask"):
            result = side.get(route) or {}
            measured = measured or bool(result.get("ok"))
            if result.get("ok") and result.get("verdict") == "bad":
                lines.append("{} ({}): ΔD {} > {} — {}.".format(
                    title, words[route], _with_unit(_num(result["delta_mm"], words), "mm"),
                    _with_unit(_num(result["threshold_mm"], words, None), "mm"), words["bad"]))
    if not measured:
        lines.append(words["no_delta"])

    # Poor graded items are listed for review, never an information value. The report states the quality and names
    # no repeat.
    poor: List[Tuple[str, Mapping[str, Any]]] = []
    for image in _ordered_images(record):
        for entry in image.get("criteria", []):
            if entry["grade"] == "poor":
                poor.append((image["view_key"], entry))

    if poor:
        lines.append(words["review"])
        for view_key, entry in poor:
            value = _value_word(entry, words)
            lines.append("· {} — {}{}".format(view_key, names.get(entry["id"], entry["name"]),
                                              f": {value}" if value else ""))

    if not lines:
        lines.append(words["all_good"])
    return lines


def _upper(text: str, lang: str) -> str:
    """Upper case with the Turkish dotted capital (i -> İ) where the language needs it."""
    return (text.replace("i", "İ") if lang == "tr" else text).upper()


def _as_text(sections: List[Mapping[str, Any]], words: Mapping[str, Any], lang: str = DEFAULT_LANGUAGE) -> str:
    out = []
    for section in sections:
        out.append(_upper(section["heading"], lang))
        out.append("─" * max(len(section["heading"]), 8))
        for row in section["rows"]:
            if "text" in row:
                out.append(row["text"])
            elif row.get("value"):
                out.append("{:<34}{}".format(row["label"] + ":", row["value"]))
            else:
                out.append(row["label"])
        out.append("")
    out.append(words["footer"])
    return "\n".join(out)

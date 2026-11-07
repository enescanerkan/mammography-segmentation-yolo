"""A written positioning report, generated from the numbers rather than from a language model.

Everything a report would say is already a structured measurement: a millimetre, a grade, a verdict. Nothing has
to be retrieved and nothing has to be inferred, so a template turns the record into sentences, and every sentence
can be traced back to the field it came from. A model asked to phrase this would sometimes round 27.6 to "about
28", sometimes to "nearly 30", and occasionally produce a number that is in no field at all; in a document that
says a view should be repeated, a wrong number is worse than a plain sentence.

Two languages, one set of facts: the strings differ, the numbers do not.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Tuple

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
        "agree": "Both methods reached the same verdict.",
        "disagree": "The methods disagree ({gap} mm apart) — this study needs review.",
        "wide": "Same verdict, but the measurements differ by {gap} mm.",
        "no_pair": "No measurement: {reason}.",
        "criteria": "Positioning criteria",
        "quality_score": "Criterion score",
        "conclusion": "Conclusion",
        "all_good": "All criteria fall within acceptable limits.",
        "no_delta": "ΔD could not be computed for either breast — no measurable MLO–CC pair.",
        "repeat": "A repeat {view} is advised.",
        "review": "The findings below should be reviewed.",
        "grades": {"good": "adequate", "acceptable": "borderline", "poor": "inadequate", "na": "not measurable"},
        "edited": "Labels were corrected by a reviewer.",
        "footer": "For research use; not for diagnostic purposes. Values marked estimated are excluded from the score.",
        "problems": "Upload notes",
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
        "disagree": "Yöntemler farklı karar verdi ({gap} mm ayrışma) — gözden geçirilmeli.",
        "wide": "Kararlar aynı ama ölçümler {gap} mm ayrışıyor.",
        "no_pair": "Ölçüm yapılamadı: {reason}.",
        "criteria": "Pozisyonlama kriterleri",
        "quality_score": "Kriter skoru",
        "conclusion": "Sonuç",
        "all_good": "Tüm kriterler yeterli sınırlar içinde.",
        "no_delta": "ΔD hiçbir meme için hesaplanamadı — ölçülebilen bir MLO–CC çifti yok.",
        "repeat": "{view} tekrarı önerilir.",
        "review": "Aşağıdaki bulgular gözden geçirilmelidir.",
        "grades": {"good": "yeterli", "acceptable": "sınırda", "poor": "yetersiz", "na": "ölçülemedi"},
        "edited": "Etiketler bir kullanıcı tarafından düzeltilmiştir.",
        "footer": "Araştırma amaçlıdır; tanısal kullanım için değildir. Tahmini olarak işaretlenen değerler skora "
                  "dahil edilmez.",
        "problems": "Yükleme notları",
    },
}

# Criterion names, kept here rather than in criteria.py so the measurement code never has to know which languages
# exist.
NAMES: Dict[str, Dict[str, str]] = {
    "en": {
        "pectoral_angle": "Pectoral angle",
        "pectoral_depth": "Pectoral reaches nipple level",
        "imf": "Inframammary fold open",
        "nipple_profile": "Nipple in profile",
        "nipple_centred": "Nipple centred",
        "cc_pectoral": "Pectoral visible on CC",
        "sharpness": "Focus",
        "exposure": "Exposure",
        "symmetry_mlo": "PNL symmetry",
        "symmetry_cc": "Chest-wall distance symmetry",
        "compression_force": "Compression force",
        "thickness": "Compressed thickness",
        "compression_pressure": "Compression pressure (estimated)",
        "gantry": "Gantry angle",
        "dose": "Entrance dose",
    },
    "tr": {
        "pectoral_angle": "Pektoral açı",
        "pectoral_depth": "Pektoral meme ucu seviyesine iniyor",
        "imf": "İnframammer katlantı açık",
        "nipple_profile": "Meme ucu profilde",
        "nipple_centred": "Meme ucu ortalanmış",
        "cc_pectoral": "CC'de pektoral görünür",
        "sharpness": "Odak",
        "exposure": "Pozlama",
        "symmetry_mlo": "PNL simetrisi",
        "symmetry_cc": "Göğüs duvarı mesafesi simetrisi",
        "compression_force": "Sıkıştırma kuvveti",
        "thickness": "Sıkıştırılmış kalınlık",
        "compression_pressure": "Sıkıştırma basıncı (tahmini)",
        "gantry": "C-kol açısı",
        "dose": "Giriş dozu",
    },
}

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

    A row is ``{"label", "value", "grade"?, "strong"?}`` or a free sentence ``{"text"}``; ``text`` is the whole
    report as plain text.
    """
    lang = lang if lang in LANGUAGES else DEFAULT_LANGUAGE
    words = WORDS[lang]
    names = NAMES[lang]
    sections: List[Dict[str, Any]] = [{"heading": words["title"], "rows": _header_rows(record, words)}]

    verdict_rows: List[Dict[str, Any]] = []
    for side in (record.get("summary") or {}).get("sides", []):
        verdict_rows.extend(_side_rows(side, words, REASONS[lang]))
    sections.append({"heading": words["verdicts"], "rows": verdict_rows})

    sections.append({"heading": words["criteria"], "rows": _criteria_rows(record, words, names)})

    conclusion = _conclusion(record, words, names)
    sections.append({"heading": words["conclusion"], "rows": [{"text": line} for line in conclusion]})

    if record.get("problems"):
        sections.append({"heading": words["problems"], "rows": [{"text": p} for p in record["problems"]]})

    return {
        "lang": lang,
        "title": words["title"],
        "sections": sections,
        "footer": words["footer"],
        "text": _as_text(sections, words, lang),
    }


def _header_rows(record: Mapping[str, Any], words: Mapping[str, Any]) -> List[Dict[str, Any]]:
    images = record.get("images") or []
    created = record.get("created")
    device = next((i.get("manufacturer") for i in images if i.get("manufacturer")), "—")
    rows: List[Dict[str, Any]] = [
        {"label": words["study"], "value": record.get("id", "—")},
        {"label": words["date"],
         "value": time.strftime("%d.%m.%Y %H:%M", time.localtime(created)) if created else "—"},
        {"label": words["views"], "value": ", ".join(i["view_key"] for i in images) or "—"},
        {"label": words["device"], "value": device or "—"},
        {"label": words["threshold"],
         "value": "{:g} mm".format((record.get("rules") or {}).get("threshold_mm", 10))},
    ]
    if record.get("edited"):
        rows.append({"text": words["edited"]})
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
        rows.append({
            "label": words[route],
            "value": "{} {:.1f} mm · {} {:.1f} mm · {} {:.1f} mm → {}".format(
                words["pnl"], result["pnl_mm"], words["ccd"], result["cc_mm"],
                words["delta"], result["delta_mm"], verdict),
            "grade": result["verdict"],
        })

    comparison = side.get("comparison") or {}
    if comparison.get("state") == "disagree":
        rows.append({"text": words["disagree"].format(gap=comparison.get("gap_mm"))})
    elif comparison.get("wide_gap"):
        rows.append({"text": words["wide"].format(gap=comparison.get("gap_mm"))})
    elif comparison.get("state") == "agree":
        rows.append({"text": words["agree"]})
    return rows


def _criterion_value(entry: Mapping[str, Any], words: Mapping[str, Any]) -> str:
    value = "" if entry["value"] is None else "{}{}".format(entry["value"], entry["unit"])
    return "{} · {}".format(value, words["grades"][entry["grade"]]).strip(" ·")


def _criteria_rows(record: Mapping[str, Any], words: Mapping[str, Any],
                   names: Mapping[str, str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    summary = (record.get("criteria") or {}).get("summary") or {}
    if summary.get("score") is not None:
        rows.append({"label": words["quality_score"], "value": "{:g} / 100".format(summary["score"]),
                     "grade": summary.get("worst"), "strong": True})

    for image in record.get("images") or []:
        items = [entry for entry in image.get("criteria", []) if entry["grade"] != "na"]
        if not items:
            continue
        rows.append({"label": image["view_key"], "value": "", "strong": True})
        for entry in items:
            rows.append({"label": "   " + names.get(entry["id"], entry["name"]),
                         "value": _criterion_value(entry, words), "grade": entry["grade"]})

    for entry in (record.get("criteria") or {}).get("study", []):
        if entry["grade"] == "na":
            continue
        rows.append({"label": names.get(entry["id"], entry["name"]),
                     "value": _criterion_value(entry, words), "grade": entry["grade"]})
    return rows


def _conclusion(record: Mapping[str, Any], words: Mapping[str, Any], names: Mapping[str, str]) -> List[str]:
    """One sentence per failing thing, each traceable to a measured field."""
    lines: List[str] = []

    # Name the route whose number actually failed. Quoting one method's millimetres under the other's verdict is
    # how a report ends up saying "7.2 mm > 10 mm".
    measured = False
    for side in (record.get("summary") or {}).get("sides", []):
        title = words["left"] if side["laterality"] == "L" else words["right"]
        for route in ("landmark", "mask"):
            result = side.get(route) or {}
            measured = measured or bool(result.get("ok"))
            if result.get("ok") and result.get("verdict") == "bad":
                lines.append("{} ({}): ΔD {:.1f} mm > {:g} mm — {}.".format(
                    title, words[route], result["delta_mm"], result["threshold_mm"], words["bad"]))
    if not measured:
        lines.append(words["no_delta"])

    # Estimates are kept out of the score, so they cannot ask for a repeat either.
    poor: List[Tuple[str, Mapping[str, Any]]] = []
    for image in record.get("images") or []:
        for entry in image.get("criteria", []):
            if entry["grade"] == "poor" and not entry.get("estimated"):
                poor.append((image["view_key"], entry))
    # Left-versus-right items belong to no single view: listed, but no repeat named.
    poor_study = [entry for entry in (record.get("criteria") or {}).get("study", [])
                  if entry["grade"] == "poor" and not entry.get("estimated")]

    if poor or poor_study:
        lines.append(words["review"])
        for view_key, entry in poor + [("", entry) for entry in poor_study]:
            value = "" if entry["value"] is None else " ({}{})".format(entry["value"], entry["unit"])
            prefix = f"{view_key} — " if view_key else ""
            lines.append("· {}{}{}".format(prefix, names.get(entry["id"], entry["name"]), value))
        for view_key in sorted({view for view, _ in poor}):
            lines.append(words["repeat"].format(view=view_key))

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

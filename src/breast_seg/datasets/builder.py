"""Config-driven dataset builder: sources -> study-level split -> cleaned YOLO polygons -> audit -> extra formats.

    builder = DatasetBuilder(load_dataset_config("configs/datasets/seg_v5.yaml"), PathsConfig.load())
    report = builder.build()

The builder never modifies its sources. It writes ``<output_dir>/{images,labels}/<split>``, ``index.csv`` and,
only if the audit passes, ``data.yaml`` (plus the configured semantic / nnU-Net exports).
"""
from __future__ import annotations

import csv
import hashlib
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple

import cv2
import numpy as np
import pandas as pd

from breast_seg.config import PathsConfig
from breast_seg.datasets.config import DatasetConfig, ExcludeConfig, SourceConfig
from breast_seg.datasets.corrections import CORRECTIONS, CorrectionParams
from breast_seg.datasets.formats import NnUNetExporter, SemanticExporter, semantic_paths, write_yolo_data_yaml
from breast_seg.datasets.labels import check_label_text, masks_to_lines, polygons_to_masks, read_polygons, write_lines
from breast_seg.datasets.splits import SPLITS, find_leaks, split_groups


class DatasetAuditError(RuntimeError):
    """The built dataset failed its audit (``problems`` lists every finding)."""

    def __init__(self, problems: List[str]):
        super().__init__(f"dataset audit failed ({len(problems)} problem(s)): " + "; ".join(problems[:5]))
        self.problems = problems


@dataclass
class Item:
    """One image of the dataset being built."""

    stem: str
    key: str
    source: str
    view: str
    image: Path
    label: Path
    split: str = ""
    groups: List[str] = field(default_factory=list)
    bucket: str = ""
    corrections: str = "none"
    strict: bool = False
    replica: int = 0
    origin: str = ""          # split inherited from a yolo_layout source
    included: bool = True     # manifest rows filtered out by ``include`` still inform group exclusions


@dataclass
class BuildReport:
    """Outcome of :meth:`DatasetBuilder.build`."""

    output_dir: Path
    counts: Dict[str, Counter]
    skipped: Counter
    problems: List[str]
    data_yaml: Optional[Path] = None

    @property
    def ok(self) -> bool:
        return not self.problems


def _read_table(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    cols = list(dict.fromkeys(columns))
    return pd.read_csv(path, usecols=cols, dtype=str, keep_default_na=False)


class DatasetBuilder:
    """Builds one dataset generation from a :class:`DatasetConfig`."""

    def __init__(self, config: DatasetConfig, paths: Optional[PathsConfig] = None):
        self.config = config
        self.paths = paths or PathsConfig.load()
        self.output_dir = self.paths.resolve(config.output_dir)
        c = config.correction
        self._params = CorrectionParams(pectoral_close=c.pectoral_close, gap_kernel=c.gap_kernel)

    # ── collection ──────────────────────────────────────────────────────────────────────────────────────────

    def _group_maps(self) -> Dict[str, List[str]]:
        """item key -> ["namespace:group", ...] from every configured group table."""
        out: Dict[str, List[str]] = defaultdict(list)
        for gm in self.config.groups:
            for p in gm.csv:
                df = _read_table(self.paths.resolve(p), (gm.key, gm.group))
                for k, g in zip(df[gm.key], df[gm.group]):
                    tag = f"{gm.namespace}:{g}"
                    if g and tag not in out[k]:
                        out[k].append(tag)
        return out

    def _split_map(self) -> Dict[str, str]:
        sm = self.config.split_map
        if sm is None:
            return {}
        out: Dict[str, str] = {}
        for p in sm.csv:
            df = _read_table(self.paths.resolve(p), (sm.key, sm.column))
            for k, v in zip(df[sm.key], df[sm.column]):
                if v in sm.values:
                    out[k] = sm.values[v]
        return out

    def _exclusion_keys(self, rule: ExcludeConfig) -> Set[str]:
        keys: Set[str] = set()
        if rule.csv:
            keys |= set(_read_table(self.paths.resolve(rule.csv), (rule.column,))[rule.column])
        if rule.list_file:
            text = self.paths.resolve(rule.list_file).read_text()
            keys |= {ln.strip() for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")}
        return keys

    def _apply_excludes(self, src: SourceConfig, items: List[Item], rows: Optional[List[Dict[str, str]]],
                        skipped: Counter) -> List[Item]:
        for rule in src.exclude:
            keys = self._exclusion_keys(rule)

            def applies(i: int, rule: ExcludeConfig = rule, rows: Optional[List[Dict[str, str]]] = rows) -> bool:
                if not rule.where or rows is None:
                    return True
                return all(rows[i].get(col, "") in vals for col, vals in rule.where.items())

            hit = {i for i, it in enumerate(items) if applies(i) and it.key in keys}
            if rule.drop_same_group:
                bad = {g for i in hit for g in items[i].groups}
                hit |= {i for i, it in enumerate(items) if applies(i) and set(it.groups) & bad}
            if rule.require_group:
                hit |= {i for i, it in enumerate(items) if applies(i) and not it.groups}
            skipped[f"{src.name}: excluded"] += sum(items[i].included for i in hit)
            items = [it for i, it in enumerate(items) if i not in hit]
            rows = [r for i, r in enumerate(rows) if i not in hit] if rows is not None else None
        return items

    def _correction_for(self, src: SourceConfig, view: str) -> str:
        if src.corrections == "auto":
            return "mlo" if view == "MLO" else "cc"
        return src.corrections

    def _collect_yolo_dir(self, src: SourceConfig, groups: Mapping[str, List[str]], skipped: Counter) -> List[Item]:
        images, labels = self.paths.resolve(src.images), self.paths.resolve(src.labels)
        out = []
        for lbl in sorted(labels.glob("*.txt")):
            key = lbl.stem
            img = images / f"{key}{src.image_ext}"
            if not img.is_file():
                skipped[f"{src.name}: label without image"] += 1
                continue
            out.append(Item(stem=f"{src.prefix}{key}", key=key, source=src.name, view=src.view, image=img, label=lbl,
                            groups=list(groups.get(key, [])), corrections=self._correction_for(src, src.view),
                            strict=src.strict_classes))
        return out

    def _collect_yolo_layout(self, src: SourceConfig, groups: Mapping[str, List[str]]) -> List[Item]:
        root = self.paths.resolve(src.root)
        out = []
        for split in SPLITS:
            for img in sorted((root / "images" / split).glob(f"*{src.image_ext}")):
                stem, key, view = img.stem, img.stem, src.view
                for prefix, v in src.view_prefixes.items():
                    if stem.startswith(prefix):
                        key, view = stem[len(prefix):], v
                        break
                out.append(Item(stem=stem, key=key, source=src.name, view=view, image=img,
                                label=root / "labels" / split / f"{stem}.txt", groups=list(groups.get(key, [])),
                                corrections=self._correction_for(src, view), strict=src.strict_classes, origin=split))
        return out

    def _collect_manifest(self, src: SourceConfig, groups: Mapping[str, List[str]],
                          skipped: Counter) -> Tuple[List[Item], List[Dict[str, str]]]:
        manifest = self.paths.resolve(src.manifest)
        root = manifest.parent
        with open(manifest, newline="") as fh:
            rows = list(csv.DictReader(fh))
        c = src.columns
        items, kept = [], []
        for r in rows:
            included = all(r.get(col, "") in vals for col, vals in src.include.items())
            key, source, view = r[c.key], r.get(c.source, ""), r.get(c.view, src.view) or src.view
            fields = {**r, "root": str(root), "key": key, "source": source, "bucket": r.get(c.bucket, ""), "view": view}
            base = src.stem_by_source.get(source, src.stem_template).format(**fields)
            stem = f"{src.view_prefix.get(view, '')}{base}"
            ns = src.group_namespace_by_source.get(source, src.group_namespace)
            own = [f"{ns}:{r[c.group]}"] if r.get(c.group) else []
            items.append(Item(stem=stem, key=key, source=source or src.name, view=view,
                              image=Path(src.image_template.format(**fields)), label=Path(src.label_template.format(**fields)),
                              groups=own + [g for g in groups.get(key, []) if g not in own], bucket=fields["bucket"],
                              corrections=self._correction_for(src, view), strict=src.strict_classes, included=included))
            kept.append(r)
        return items, kept

    def collect(self) -> Tuple[List[Item], Counter]:
        """Gather every source, apply exclusions and assign splits. Nothing is written."""
        groups = self._group_maps()
        split_map = self._split_map()
        skipped: Counter = Counter()
        items: List[Item] = []
        ratio_items: List[Item] = []
        for src in self.config.sources:
            rows = None
            if src.type == "yolo_dir":
                got = self._collect_yolo_dir(src, groups, skipped)
            elif src.type == "yolo_layout":
                got = self._collect_yolo_layout(src, groups)
            else:
                got, rows = self._collect_manifest(src, groups, skipped)
            got = self._apply_excludes(src, got, rows, skipped)
            skipped[f"{src.name}: not included"] += sum(not it.included for it in got)
            for it in (it for it in got if it.included):
                if src.split == "map":
                    it.split = split_map.get(it.key, "")
                    if not it.split:
                        skipped[f"{src.name}: not in split map"] += 1
                        continue
                elif src.split == "inherit":
                    it.split = it.origin
                elif src.split == "ratio":
                    ratio_items.append(it)
                else:
                    it.split = src.split
                items.append(it)
        if ratio_items:
            # one joint split over the groups of every ratio source: a study never straddles two splits
            gkey = {id(it): (it.groups[0] if it.groups else f"stem:{it.stem}") for it in ratio_items}
            assign = split_groups(gkey.values(), self.config.ratios, self.config.ratio_seed)
            for it in ratio_items:
                it.split = assign[gkey[id(it)]]
        items = self._dedupe(items, skipped)
        return self._oversample(items), skipped

    def _dedupe(self, items: List[Item], skipped: Counter) -> List[Item]:
        seen: Dict[Tuple[str, str], Item] = {}
        out = []
        for it in items:
            k = (it.split, it.stem)
            if k in seen:
                if self.config.on_duplicate == "error":
                    raise ValueError(f"the same name occurs twice in {it.split} (sources {seen[k].source} and {it.source})")
                skipped[f"{it.source}: duplicate name"] += 1
                continue
            seen[k] = it
            out.append(it)
        return out

    def _oversample(self, items: List[Item]) -> List[Item]:
        factor = {s.name: s.oversample for s in self.config.sources}
        out = []
        for it in items:
            out.append(it)
            for r in range(1, factor.get(it.source, 1) if it.split == "train" else 1):
                out.append(Item(**{**it.__dict__, "stem": f"{it.stem}_r{r}", "replica": r, "groups": list(it.groups)}))
        return out

    # ── writing ─────────────────────────────────────────────────────────────────────────────────────────────

    def _write_label(self, it: Item, dst: Path) -> None:
        if it.corrections == "none":
            shutil.copyfile(it.label, dst)
            return
        img = cv2.imread(str(it.image), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"unreadable image of {it.source}/{it.stem}")
        h, w = img.shape
        masks = polygons_to_masks(read_polygons(it.label), h, w)
        fixed = CORRECTIONS[it.corrections](masks, self._params)
        c = self.config.correction
        write_lines(dst, masks_to_lines(fixed, c.min_contour_area, c.approx_epsilon))

    def write(self, items: List[Item]) -> None:
        out = self.output_dir
        if out.exists():
            shutil.rmtree(out)
        for split in SPLITS:
            (out / "images" / split).mkdir(parents=True)
            (out / "labels" / split).mkdir(parents=True)
        for it in items:
            if not it.image.is_file() or not it.label.is_file():
                raise FileNotFoundError(f"missing image or label for {it.source}/{it.stem}")
            shutil.copyfile(it.image, out / "images" / it.split / f"{it.stem}.png")
            self._write_label(it, out / "labels" / it.split / f"{it.stem}.txt")
        with open(out / "index.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["stem", "split", "source", "bucket", "view", "groups", "replica"])
            for it in items:
                w.writerow([it.stem, it.split, it.source, it.bucket, it.view, ";".join(it.groups), it.replica])

    # ── audit ───────────────────────────────────────────────────────────────────────────────────────────────

    def audit(self, items: List[Item]) -> List[str]:
        """Every problem of the written dataset (empty list = clean)."""
        a, out = self.config.audit, self.output_dir
        problems: List[str] = []
        seen_hash: Dict[str, str] = {}
        name_to_cls = {n: i for i, n in enumerate(self.config.class_names)}
        for it in items:
            where = f"{it.split}/{it.stem}"
            img = cv2.imread(str(out / "images" / it.split / f"{it.stem}.png"), cv2.IMREAD_UNCHANGED)
            if img is None:
                problems.append(f"{where}: unreadable image")
                continue
            if a.image_shape and (img.shape != tuple(a.image_shape) or img.dtype != np.uint8):
                problems.append(f"{where}: image {img.shape} {img.dtype}, expected {tuple(a.image_shape)} uint8")
            if a.unique_pixels and it.replica == 0:
                digest = hashlib.md5(img.tobytes()).hexdigest()
                if digest in seen_hash:
                    problems.append(f"identical pixels: {seen_hash[digest]} == {where}")
                seen_hash[digest] = where
            counts, errors = check_label_text((out / "labels" / it.split / f"{it.stem}.txt").read_text(),
                                              range(len(self.config.class_names)))
            problems += [f"{where}: {e}" for e in errors]
            breast = name_to_cls.get("breast-tissue", 1)
            if a.require_breast and breast not in counts:
                problems.append(f"{where}: no breast tissue")
            if it.strict:
                need = set(a.required_classes.get(it.view, []))
                if need - set(counts):
                    problems.append(f"{where}: missing class(es) {sorted(need - set(counts))}")
            if a.forbid_cc_pectoral and it.view == "CC" and name_to_cls.get("pectoral", 0) in counts:
                problems.append(f"{where}: pectoral polygon on a CC view")
        for split in SPLITS:
            imgs = {p.stem for p in (out / "images" / split).glob("*.png")}
            lbls = {p.stem for p in (out / "labels" / split).glob("*.txt")}
            want = {it.stem for it in items if it.split == split}
            if imgs != want or lbls != want:
                problems.append(f"{split}: files and index disagree "
                                f"({len(imgs)} images, {len(lbls)} labels, {len(want)} indexed)")
        if a.leakage:
            ids = {f"{it.split}/{it.stem}": it for it in items}
            leaks = find_leaks({k: it.split for k, it in ids.items()}, {k: it.groups for k, it in ids.items()})
            problems += [f"group shared by {sorted(s)}: {g.split(':', 1)[0]} key" for g, s in list(leaks.items())[:50]]
            if len(leaks) > 50:
                problems.append(f"... {len(leaks) - 50} more shared groups")
        return problems

    # ── exports ─────────────────────────────────────────────────────────────────────────────────────────────

    def export(self, items: List[Item]) -> None:
        ex = self.config.exports
        if not ex.semantic.enabled:
            return
        sem_dir = self.paths.resolve(ex.semantic.output_dir)
        records = [{"stem": it.stem, "split": it.split, "view": it.view, "source": it.source,
                    "group": it.groups[0] if it.groups else "",
                    "image": self.output_dir / "images" / it.split / f"{it.stem}.png",
                    "label": self.output_dir / "labels" / it.split / f"{it.stem}.txt"} for it in items]
        index = SemanticExporter(sem_dir).export(records)
        if ex.nnunet.enabled:
            n = ex.nnunet
            groups = {it.stem: (it.groups[0] if it.groups else it.stem) for it in items}
            NnUNetExporter(self.paths.resolve(n.raw_dir), self.paths.resolve(n.preprocessed_dir), n.dataset_id,
                           n.dataset_name, n.description).export(semantic_paths(sem_dir, index), n.cv_folds, n.cv_seed, groups)

    def _check_output(self) -> None:
        """The output folder is wiped before writing: refuse one that is, or contains, a source folder."""
        out = self.output_dir.resolve()
        for src in self.config.sources:
            for value in (src.images, src.labels, src.root, src.manifest):
                if value:
                    p = self.paths.resolve(value).resolve()
                    if p == out or out in p.parents:
                        raise ValueError(f"output_dir of {self.config.name} would overwrite source {src.name}")

    def build(self, strict: bool = True) -> BuildReport:
        """Collect, write, audit and (if clean) write ``data.yaml`` and the exports.

        With ``strict`` an audit failure raises :class:`DatasetAuditError`; otherwise it is reported.
        """
        self._check_output()
        items, skipped = self.collect()
        self.write(items)
        problems = self.audit(items)
        counts: Dict[str, Counter] = {sp: Counter() for sp in SPLITS}
        for it in items:
            counts[it.split][f"{it.source}/{it.view}"] += 1
        report = BuildReport(self.output_dir, counts, skipped, problems)
        if problems:
            if strict:
                raise DatasetAuditError(problems)
            return report
        report.data_yaml = write_yolo_data_yaml(self.output_dir / "data.yaml", self.output_dir, self.config.class_names)
        self.export(items)
        return report


def build_dataset(config: DatasetConfig, paths: Optional[PathsConfig] = None, strict: bool = True) -> BuildReport:
    """Convenience wrapper: ``DatasetBuilder(config, paths).build(strict)``."""
    return DatasetBuilder(config, paths).build(strict)

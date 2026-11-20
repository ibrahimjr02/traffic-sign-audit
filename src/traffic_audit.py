"""Traffic Sign Visibility Audit for the GTSRB training images.

The module contains:

* reusable functions   -> load_annotations, validate_annotations,
                          extract_image_features, assign_review_priority
* an organising class  -> TrafficSignAudit (load, validate, sample, measure,
                          flag, summarise, plot, save)

Every threshold used here is a *project-specific triage setting*. None of
them is a road-safety standard, and a review flag is not a safety verdict.

The whole pipeline can be run from the repository root with::

    python src/traffic_audit.py
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle

# ----------------------------------------------------------------------------
# Configuration constants (all can be overridden when creating the audit)
# ----------------------------------------------------------------------------
DEFAULT_SEED = 42
DEFAULT_N_CLASSES = 8
DEFAULT_PER_CLASS = 500
# name -> (low percentile, high percentile); "base" is the main rule.
DEFAULT_SETTINGS = {"strict": (5, 95), "base": (10, 90), "loose": (20, 80)}

# Annotation header (lower-cased, no spaces/underscores) -> internal name.
COLUMN_MAP = {
    "width": "width",
    "height": "height",
    "roi.x1": "x1",
    "roi.y1": "y1",
    "roi.x2": "x2",
    "roi.y2": "y2",
    "classid": "class_id",
    "path": "path",
}
REQUIRED_COLUMNS = ["width", "height", "x1", "y1", "x2", "y2", "class_id", "path"]
NUMERIC_COLUMNS = ["width", "height", "x1", "y1", "x2", "y2", "class_id"]

FEATURE_COLUMNS = [
    "brightness", "contrast", "sharpness",
    "img_width", "img_height", "aspect_ratio", "min_side",
]
ROI_FEATURE_COLUMNS = ["roi_brightness", "roi_contrast", "roi_sharpness"]

# Review conditions; each becomes a boolean column "flag_<name>".
CONDITIONS = ["dark", "overexposed", "low_contrast", "blurry", "small"]
# When several conditions fire, the first one in this list is the primary reason.
REASON_PRIORITY = ["blurry", "low_contrast", "dark", "overexposed", "small"]


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def _canonical(name) -> str:
    """Lower-case a column name and drop spaces/underscores for matching."""
    return str(name).strip().lower().replace(" ", "").replace("_", "")


def _is_file(path: Path) -> bool:
    """True if ``path`` is an existing file; never raises on odd strings."""
    try:
        return path.is_file()
    except (OSError, ValueError):
        return False


def _short_label(class_id, name, width: int = 34) -> str:
    """Compact "id: name" label for plot ticks."""
    label = f"{int(class_id)}: {name}"
    return label if len(label) <= width else label[: width - 1] + "…"


def _spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman rank correlation: Pearson correlation of the average ranks.

    Written with pandas only, so no extra package (such as SciPy) is needed.
    """
    return float(a.rank().corr(b.rank()))


def _log_sharpness(values) -> np.ndarray:
    """log10(1 + Laplacian variance); keeps zeros plottable on a log-like scale."""
    return np.log10(1.0 + np.asarray(values, dtype=float))


# ----------------------------------------------------------------------------
# Reusable function 1: read annotations
# ----------------------------------------------------------------------------
def load_class_names(names_path) -> dict:
    """Read ``signnames.csv`` and return ``{class_id: sign name}``.

    Returns an empty dict (and prints a notice) if the file is absent, so the
    audit still runs with generic class labels.
    """
    names_path = Path(names_path)
    if not _is_file(names_path):
        print(f"Notice: {names_path} not found; using generic class labels.")
        return {}
    table = pd.read_csv(names_path, encoding="utf-8-sig")
    table.columns = [_canonical(c) for c in table.columns]
    if not {"classid", "signname"} <= set(table.columns):
        print(f"Notice: {names_path} lacks ClassId/SignName columns; using generic labels.")
        return {}
    return {int(k): str(v) for k, v in zip(table["classid"], table["signname"])}


def load_annotations(csv_path, names_path=None) -> pd.DataFrame:
    """Read a GTSRB annotation CSV into a tidy, consistently named table.

    Parameters
    ----------
    csv_path : path to ``Train.csv`` (comma or semicolon separated).
    names_path : optional path to ``signnames.csv`` for readable class names.

    Returns
    -------
    DataFrame with columns ``ann_row, width, height, x1, y1, x2, y2, class_id,
    path, raw_path, image_path, class_name``. ``image_path`` is *relative to the
    data folder* (the author's ``/code/`` prefix is removed), so no personal
    absolute path is ever stored.

    Raises
    ------
    FileNotFoundError, ValueError with a message that says what to fix.
    """
    csv_path = Path(csv_path)
    if not _is_file(csv_path):
        raise FileNotFoundError(
            f"Annotation file not found: {csv_path}. The GTSRB files are expected in the "
            "data/ folder (see README.md)."
        )
    try:
        raw = pd.read_csv(csv_path, encoding="utf-8-sig")
        if raw.shape[1] == 1:  # semicolon-separated variant of the benchmark
            raw = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not parse {csv_path}: {exc}") from exc

    rename = {c: COLUMN_MAP[_canonical(c)] for c in raw.columns if _canonical(c) in COLUMN_MAP}
    df = raw.rename(columns=rename)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path.name} is missing required column(s) {missing}. "
            f"Columns found: {list(raw.columns)}"
        )
    df = df[REQUIRED_COLUMNS].copy()
    df.insert(0, "ann_row", np.arange(len(df)))
    for col in NUMERIC_COLUMNS:  # malformed values become NaN and are flagged later
        df[col] = pd.to_numeric(df[col], errors="coerce")

    raw_path = df["path"].astype(str).str.strip().str.replace("\\", "/", regex=False)
    df["raw_path"] = raw_path
    df["image_path"] = raw_path.str.replace(r"^/?code/", "", regex=True).str.lstrip("/")

    names = load_class_names(names_path) if names_path is not None else {}
    labels = df["class_id"].map(names)
    fallback = df["class_id"].apply(
        lambda v: f"Class {int(v)}" if pd.notna(v) and float(v).is_integer() else "Unknown class"
    )
    df["class_name"] = labels.where(labels.notna(), fallback)
    return df


# ----------------------------------------------------------------------------
# Reusable function 2: validate annotations (never drops rows silently)
# ----------------------------------------------------------------------------
def validate_annotations(df: pd.DataFrame, data_dir, known_class_ids=None):
    """Add validation flags to every annotation row and summarise the checks.

    Nothing is deleted. Each row gets boolean flag columns plus an
    ``exclude_reason`` string and a ``usable`` flag. An invalid ROI does **not**
    exclude a row: the full image is still measured, only the crop is skipped.

    Returns
    -------
    (validated DataFrame, checks DataFrame with columns check / rows_affected / action)
    """
    data_dir = Path(data_dir)
    out = df.copy()

    out["bad_number"] = out[NUMERIC_COLUMNS].isna().any(axis=1)
    ok = ~out["bad_number"]

    out["bad_size"] = ok & ((out["width"] <= 0) | (out["height"] <= 0))

    not_integer = ok & (out["class_id"] != np.floor(out["class_id"]))
    negative = ok & (out["class_id"] < 0)
    unknown = (
        ok & ~out["class_id"].isin(list(known_class_ids)) if known_class_ids else ok & False
    )
    out["bad_class"] = not_integer | negative | unknown

    # ROI coordinates are inclusive pixel indices: 0 <= x1 <= x2 <= width-1.
    roi_ok = (
        (out["x1"] >= 0) & (out["y1"] >= 0)
        & (out["x1"] <= out["x2"]) & (out["y1"] <= out["y2"])
        & (out["x2"] <= out["width"] - 1) & (out["y2"] <= out["height"] - 1)
    )
    out["roi_valid"] = ok & roi_ok
    out["bad_roi"] = ok & ~roi_ok

    out["file_exists"] = [_is_file(data_dir / p) for p in out["image_path"]]

    key_cols = ["width", "height", "x1", "y1", "x2", "y2", "class_id", "image_path"]
    out["duplicate_row"] = out.duplicated(subset=key_cols, keep="first")
    out["duplicate_path"] = out.duplicated(subset=["image_path"], keep="first")

    reasons = {
        "missing_or_malformed_number": out["bad_number"],
        "non_positive_size": out["bad_size"],
        "invalid_class_id": out["bad_class"],
        "file_not_found": ~out["file_exists"],
        "duplicate_image_path": out["duplicate_path"],
    }
    names = list(reasons)
    masks = [m.to_numpy() for m in reasons.values()]
    out["exclude_reason"] = [
        ";".join(n for n, hit in zip(names, row) if hit) for row in zip(*masks)
    ]
    out["usable"] = out["exclude_reason"] == ""

    checks = pd.DataFrame(
        [
            ("Rows in annotation file", len(out), "all rows kept in the validated table"),
            ("Distinct class IDs", int(out["class_id"].nunique()), "reported"),
            ("Missing or malformed numeric fields", int(out["bad_number"].sum()), "excluded"),
            ("Width or height not positive", int(out["bad_size"].sum()), "excluded"),
            ("Invalid class ID", int(out["bad_class"].sum()), "excluded"),
            ("Image file not found", int((~out["file_exists"]).sum()), "excluded"),
            ("Duplicate annotation rows (identical)", int(out["duplicate_row"].sum()), "later copies excluded"),
            ("Duplicate image paths (later copies)", int(out["duplicate_path"].sum()), "excluded"),
            ("Invalid or absent ROI", int(out["bad_roi"].sum()), "kept; ROI measures set to NaN"),
            ("Rows usable for analysis", int(out["usable"].sum()), "-"),
            ("Rows excluded", int((~out["usable"]).sum()), "listed in exclusion_log.csv"),
        ],
        columns=["check", "rows_affected", "action"],
    )
    return out, checks


# ----------------------------------------------------------------------------
# Reusable function 3: OpenCV measurements for one image
# ----------------------------------------------------------------------------
def _measure_gray(gray: np.ndarray) -> dict:
    """Brightness, contrast and sharpness of a grayscale (uint8) array."""
    return {
        "brightness": float(gray.mean()),                           # mean intensity, 0-255
        "contrast": float(gray.std()),                              # std of intensity
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),  # variance of Laplacian
    }


def crop_roi(gray: np.ndarray, roi):
    """Crop the annotated region (inclusive x1,y1,x2,y2). Return None if unusable."""
    if roi is None:
        return None
    try:
        x1, y1, x2, y2 = (int(v) for v in roi)
    except (TypeError, ValueError):
        return None
    h, w = gray.shape
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w - 1), min(y2, h - 1)
    if x2 < x1 or y2 < y1:
        return None
    crop = gray[y1:y2 + 1, x1:x2 + 1]
    return crop if crop.shape[0] >= 3 and crop.shape[1] >= 3 else None


def extract_image_features(image_path, roi=None) -> dict:
    """Read one image with ``cv2.imread`` and measure it.

    Parameters
    ----------
    image_path : path of the image file.
    roi : optional ``(x1, y1, x2, y2)`` inclusive pixel box for a crop comparison.

    Returns
    -------
    dict with ``ok`` (bool) and ``error`` (str). When ``ok`` is True it also holds
    brightness, contrast, sharpness, img_width, img_height, aspect_ratio,
    min_side and roi_brightness/roi_contrast/roi_sharpness (NaN if no usable ROI).
    A failed read never raises; it returns ``ok=False`` with a reason.
    """
    result = {"ok": False, "error": ""}
    try:
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)  # OpenCV order: B, G, R
    except Exception as exc:  # unusual codec / path problems
        result["error"] = f"cv2.imread raised {type(exc).__name__}: {exc}"
        return result
    if bgr is None or bgr.size == 0 or bgr.ndim != 3:
        result["error"] = "cv2.imread returned nothing (missing, corrupt or unsupported file)"
        return result

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    result.update(_measure_gray(gray))
    result.update(
        img_width=int(w), img_height=int(h),
        aspect_ratio=float(w) / float(h), min_side=int(min(w, h)),
    )
    crop = crop_roi(gray, roi)
    if crop is None:
        result.update(roi_brightness=np.nan, roi_contrast=np.nan, roi_sharpness=np.nan)
    else:
        m = _measure_gray(crop)
        result.update(
            roi_brightness=m["brightness"], roi_contrast=m["contrast"],
            roi_sharpness=m["sharpness"],
        )
    result["ok"] = True
    return result


def load_rgb(image_path):
    """Read an image and return it in RGB order for Matplotlib (None on failure)."""
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ----------------------------------------------------------------------------
# Reusable function 4: the review-priority rule (vectorised NumPy)
# ----------------------------------------------------------------------------
def assign_review_priority(features: pd.DataFrame, low_pct: float = 10, high_pct: float = 90):
    """Flag images for human review using percentile cutoffs of the sample.

    Conditions (all cutoffs are project-specific triage settings):
      dark          brightness   < low_pct percentile
      overexposed   brightness   > high_pct percentile
      low_contrast  contrast     < low_pct percentile
      blurry        sharpness    < low_pct percentile
      small         min_side     < low_pct percentile

    ``review_score`` counts how many conditions an image triggers; images with
    a higher score are reviewed first. ``review_tier`` is built with np.select.

    Returns
    -------
    (copy of ``features`` with flag_* / review_* columns, dict of cutoff values)
    """
    needed = ["brightness", "contrast", "sharpness", "min_side"]
    absent = [c for c in needed if c not in features.columns]
    if absent:
        raise ValueError(f"assign_review_priority needs column(s) {absent}")
    if features.empty:
        raise ValueError("assign_review_priority received no rows")
    if not 0 < low_pct < high_pct < 100:
        raise ValueError("Percentiles must satisfy 0 < low < high < 100")

    out = features.copy()
    b, c, s, m = (out[k].to_numpy(dtype=float) for k in needed)
    cutoffs = {
        "dark": float(np.percentile(b, low_pct)),
        "overexposed": float(np.percentile(b, high_pct)),
        "low_contrast": float(np.percentile(c, low_pct)),
        "blurry": float(np.percentile(s, low_pct)),
        "small": float(np.percentile(m, low_pct)),
    }
    masks = {
        "dark": b < cutoffs["dark"],
        "overexposed": b > cutoffs["overexposed"],
        "low_contrast": c < cutoffs["low_contrast"],
        "blurry": s < cutoffs["blurry"],
        "small": m < cutoffs["small"],
    }
    for name in CONDITIONS:
        out[f"flag_{name}"] = masks[name]
    score = np.sum([masks[n].astype(int) for n in CONDITIONS], axis=0)
    out["review_score"] = score
    out["review_flag"] = score >= 1
    out["review_tier"] = np.select([score >= 2, score == 1], ["high", "medium"], default="none")
    out["primary_reason"] = np.select(
        [masks[n] for n in REASON_PRIORITY], REASON_PRIORITY, default="none"
    )
    return out, cutoffs


# ----------------------------------------------------------------------------
# The class that organises the whole workflow
# ----------------------------------------------------------------------------
class TrafficSignAudit:
    """Workflow: load -> validate -> sample -> measure -> flag -> summarise -> plot."""

    def __init__(self, data_dir, output_dir, train_csv="Train.csv", names_csv="signnames.csv",
                 n_classes=DEFAULT_N_CLASSES, per_class=DEFAULT_PER_CLASS,
                 seed=DEFAULT_SEED, settings=None, classes=None):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.fig_dir = self.output_dir / "figures"
        self.proc_dir = self.output_dir / "processed"
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.proc_dir.mkdir(parents=True, exist_ok=True)
        self.train_csv = self.data_dir / train_csv
        self.names_csv = self.data_dir / names_csv
        self.n_classes, self.per_class, self.seed = n_classes, per_class, seed
        self.settings = dict(settings or DEFAULT_SETTINGS)
        self.manual_classes = list(classes) if classes else None
        self.class_names, self.captions, self.tables = {}, {}, {}
        self.annotations = self.validated = self.checks = None
        self.sample = self.features = self.skipped = None
        self.flagged = self.cutoffs = self.sens_cutoffs = None
        self.classes, self.examples = [], None

    # ---- step 1 -----------------------------------------------------------
    def load_annotations(self) -> pd.DataFrame:
        """Read Train.csv (+ signnames.csv) into ``self.annotations``."""
        self.class_names = load_class_names(self.names_csv)
        self.annotations = load_annotations(self.train_csv, self.names_csv)
        return self.annotations

    # ---- step 2 -----------------------------------------------------------
    def validate(self):
        """Validate every row; keep all rows, record exclusions."""
        self._need(self.annotations, "load_annotations()")
        self.validated, self.checks = validate_annotations(
            self.annotations, self.data_dir, set(self.class_names) or None
        )
        return self.checks

    @property
    def exclusion_log(self) -> pd.DataFrame:
        """Every excluded annotation row with its reason."""
        self._need(self.validated, "validate()")
        cols = ["ann_row", "image_path", "class_id", "exclude_reason"]
        return self.validated.loc[~self.validated["usable"], cols].reset_index(drop=True)

    # ---- step 3 -----------------------------------------------------------
    def select_sample(self) -> pd.DataFrame:
        """Pick classes (evenly spaced by size) and a seeded stratified sample."""
        self._need(self.validated, "validate()")
        usable = self.validated[self.validated["usable"]]
        counts = usable.groupby("class_id").size().sort_values(kind="mergesort")
        if self.manual_classes:
            self.classes = sorted(int(c) for c in self.manual_classes)
        else:
            k = min(self.n_classes, len(counts))
            ranks = np.unique(np.round(np.linspace(0, len(counts) - 1, k)).astype(int))
            self.classes = sorted(int(counts.index[i]) for i in ranks)
        if len(self.classes) < 5:
            raise ValueError("The project needs at least five sign classes.")
        parts = []
        for cid in self.classes:
            group = usable[usable["class_id"] == cid]
            parts.append(group.sample(n=min(self.per_class, len(group)), random_state=self.seed))
        self.sample = pd.concat(parts).sort_values("ann_row").reset_index(drop=True)
        return self.sample

    # ---- step 4 -----------------------------------------------------------
    def extract_features(self) -> pd.DataFrame:
        """Measure every sampled image; failures go to ``self.skipped``."""
        self._need(self.sample, "select_sample()")
        records, skipped = [], []
        for row in self.sample.itertuples(index=False):
            roi = (row.x1, row.y1, row.x2, row.y2) if row.roi_valid else None
            res = extract_image_features(self.data_dir / row.image_path, roi)
            if not res["ok"]:
                skipped.append({"ann_row": row.ann_row, "image_path": row.image_path,
                                "reason": res["error"]})
                continue
            res = {k: v for k, v in res.items() if k not in ("ok", "error")}
            res["ann_row"] = row.ann_row
            records.append(res)
        self.skipped = pd.DataFrame(skipped, columns=["ann_row", "image_path", "reason"])
        if not records:
            raise RuntimeError("No image could be read; the data/ folder may be missing or incomplete (see README.md).")
        meas = pd.DataFrame(records)
        base = self.sample[["ann_row", "image_path", "class_id", "class_name", "width", "height",
                            "x1", "y1", "x2", "y2", "roi_valid"]].copy()
        track = base["image_path"].str.extract(r"(\d{5})_(\d{5})_(\d{5})\.\w+$")
        base["track_id"] = (track[0] + "_" + track[1]).where(track[0].notna())
        tidy = base.merge(meas, on="ann_row", how="inner", validate="one_to_one")
        tidy["size_matches"] = (tidy["width"] == tidy["img_width"]) & (tidy["height"] == tidy["img_height"])
        tidy["class_id"] = tidy["class_id"].astype(int)
        self.features = tidy
        return tidy

    # ---- step 5 -----------------------------------------------------------
    def flag(self) -> pd.DataFrame:
        """Apply the base rule and every sensitivity setting."""
        self._need(self.features, "extract_features()")
        low, high = self.settings["base"]
        flagged, self.cutoffs = assign_review_priority(self.features, low, high)
        self.sens_cutoffs = {}
        for name, (lo, hi) in self.settings.items():
            alt, cuts = assign_review_priority(self.features, lo, hi)
            flagged[f"review_flag_{name}"] = alt["review_flag"].to_numpy()
            self.sens_cutoffs[name] = cuts
        self.flagged = flagged
        self._pick_examples()
        return flagged

    # ---- step 6: summaries ------------------------------------------------
    def sensitivity_table(self) -> pd.DataFrame:
        """Overall flag counts and cutoff values for each setting."""
        self._need(self.flagged, "flag()")
        rows = []
        for name, (lo, hi) in self.settings.items():
            n = int(self.flagged[f"review_flag_{name}"].sum())
            rows.append({"setting": name, "low_pct": lo, "high_pct": hi,
                         **{f"cut_{k}": round(v, 2) for k, v in self.sens_cutoffs[name].items()},
                         "n_images": len(self.flagged), "n_flagged": n,
                         "pct_flagged": round(100 * n / len(self.flagged), 2)})
        return pd.DataFrame(rows)

    def q1_class_counts(self) -> pd.DataFrame:
        """Q1: full-dataset counts per class, with the sampled count alongside."""
        self._need(self.flagged, "flag()")
        t = (self.validated.dropna(subset=["class_id"])
             .groupby(["class_id", "class_name"])
             .agg(annotation_rows=("ann_row", "size"), usable_rows=("usable", "sum"))
             .reset_index())
        t["class_id"] = t["class_id"].astype(int)
        t["usable_rows"] = t["usable_rows"].astype(int)
        sampled = self.flagged.groupby("class_id").size().rename("sampled_images")
        t = t.merge(sampled, on="class_id", how="left")
        t["sampled_images"] = t["sampled_images"].fillna(0).astype(int)
        t["selected"] = t["class_id"].isin(self.classes)
        self.tables["q1"] = t
        return t

    def q2_class_profiles(self) -> pd.DataFrame:
        """Q2: per-class medians and how far each class is from the overall median.

        ``*_effect`` = (class median - overall median) / overall IQR, so values are
        comparable across measures. |effect| >= 0.5 is marked ``notable`` as a
        project-specific rule of thumb, not a statistical test.
        """
        self._need(self.flagged, "flag()")
        f = self.flagged
        measures = ["min_side", "brightness", "contrast", "sharpness"]
        prof = (f.groupby(["class_id", "class_name"])
                .agg(n=("ann_row", "size"), **{f"median_{m}": (m, "median") for m in measures})
                .reset_index())
        for m in measures:
            q1, q3 = f[m].quantile([0.25, 0.75])
            iqr = (q3 - q1) or np.nan
            prof[f"{m}_effect"] = (prof[f"median_{m}"] - f[m].median()) / iqr
        eff_cols = [f"{m}_effect" for m in measures]
        prof["notable"] = prof[eff_cols].abs().max(axis=1) >= 0.5
        self.tables["q2"] = prof
        return prof

    def q3_flag_rates(self) -> pd.DataFrame:
        """Q3: percentage flagged per class under every setting, plus ranks."""
        self._need(self.flagged, "flag()")
        f = self.flagged
        rates = pd.DataFrame({
            f"pct_flagged_{name}": f.groupby("class_id")[f"review_flag_{name}"].mean() * 100
            for name in self.settings
        })
        for name in self.settings:
            rates[f"rank_{name}"] = rates[f"pct_flagged_{name}"].rank(ascending=False, method="min")
        rates["pct_high_tier_base"] = f.groupby("class_id")["review_tier"].apply(lambda t: (t == "high").mean() * 100)
        rates.insert(0, "class_name", f.groupby("class_id")["class_name"].first())
        rates = rates.reset_index()
        self.tables["q3"] = rates
        return rates

    def rank_agreement(self) -> pd.DataFrame:
        """Spearman correlation of class rankings, each setting vs the base rule."""
        rates = self.tables.get("q3", None)
        if rates is None:
            rates = self.q3_flag_rates()
        rows = []
        for name in self.settings:
            rho = _spearman(rates["rank_base"], rates[f"rank_{name}"])
            top = rates.sort_values(f"pct_flagged_{name}", ascending=False).head(3)
            rows.append({"setting": name, "spearman_rank_vs_base": round(rho, 3),
                         "top3_classes": ", ".join(str(int(c)) for c in top["class_id"])})
        return pd.DataFrame(rows)

    def q4_associations(self):
        """Q4: Spearman correlations among measures (overall and sharpness-contrast by class)."""
        self._need(self.flagged, "flag()")
        f = self.flagged
        cols = ["sharpness", "contrast", "brightness", "min_side"]
        overall = f[cols].rank().corr().round(3)  # Spearman = Pearson on ranks
        by_class = pd.DataFrame([
            {"class_id": cid, "class_name": g["class_name"].iloc[0], "n": len(g),
             "rho_sharp_contrast": round(_spearman(g["sharpness"], g["contrast"]), 3),
             "rho_sharp_brightness": round(_spearman(g["sharpness"], g["brightness"]), 3)}
            for cid, g in f.groupby("class_id")
        ])
        self.tables["q4_overall"], self.tables["q4_by_class"] = overall, by_class
        return overall, by_class

    def _pick_examples(self) -> None:
        """Q5: three different images that illustrate the rule.

        The most extreme flagged image is taken for each of: dark; blurry among
        images that are not flagged dark (so blur is shown on its own); and
        overexposed. An image is never used twice. ``illustrates`` records why
        each image was chosen.
        """
        f = self.flagged
        picks, used = [], set()
        for reason, col, ascending in [("dark", "brightness", True),
                                       ("blurry", "sharpness", True),
                                       ("overexposed", "brightness", False)]:
            pool = f[f[f"flag_{reason}"] & ~f["ann_row"].isin(used)]
            if reason == "blurry":
                alone = pool[~pool["flag_dark"]]
                pool = alone if len(alone) else pool
            if len(pool):
                row = pool.sort_values(col, ascending=ascending).iloc[0].copy()
                row["illustrates"] = reason
                picks.append(row)
                used.add(int(row["ann_row"]))
        fill = f[~f["ann_row"].isin(used)].sort_values(["review_score", "sharpness"], ascending=[False, True])
        for _, row in fill.iterrows():
            if len(picks) >= 3:
                break
            row = row.copy()
            row["illustrates"] = "highest score"
            picks.append(row)
        self.examples = pd.DataFrame(picks).reset_index(drop=True)

    def track_summary(self) -> pd.DataFrame:
        """Do flags cluster within physical signs?

        GTSRB file names are ``<class>_<track>_<frame>``; all frames of one track show the
        same physical sign. A track is *mostly flagged* when at least half of its sampled
        images are flagged under the base rule.
        """
        self._need(self.flagged, "flag()")
        f = self.flagged.dropna(subset=["track_id"]).copy()
        f["track_rate"] = f.groupby("track_id")["review_flag"].transform("mean")
        f["is_high"] = f["review_tier"] == "high"
        f["track_has_high"] = f.groupby("track_id")["is_high"].transform("max")

        def summarise(g, label, name):
            flagged = g[g["review_flag"]]
            return {"class_id": label, "class_name": name, "images": len(g),
                    "tracks": g["track_id"].nunique(),
                    "tracks_with_a_high_tier_image": g.loc[g["track_has_high"], "track_id"].nunique(),
                    "pct_flagged_images_in_mostly_flagged_tracks":
                        round(100 * (flagged["track_rate"] >= 0.5).mean(), 1) if len(flagged) else np.nan}

        rows = [summarise(g, int(cid), g["class_name"].iloc[0]) for cid, g in f.groupby("class_id")]
        rows.append(summarise(f, "all", "All selected classes"))
        table = pd.DataFrame(rows)
        self.tables["tracks"] = table
        return table

    def roi_comparison(self) -> pd.DataFrame:
        """Compare full-image and ROI-crop measures where the ROI is usable."""
        table_in = self.flagged if self.flagged is not None else self.features
        self._need(table_in, "extract_features()")
        f = table_in.dropna(subset=ROI_FEATURE_COLUMNS)
        rows = []
        for m in ["brightness", "contrast", "sharpness"]:
            diff = f[f"roi_{m}"] - f[m]
            rows.append({"measure": m, "n_images": len(f),
                         "mean_full": round(f[m].mean(), 2),
                         "mean_roi_crop": round(f[f"roi_{m}"].mean(), 2),
                         "mean_difference": round(diff.mean(), 2),
                         "spearman_full_vs_crop": round(_spearman(f[m], f[f"roi_{m}"]), 3)})
        table = pd.DataFrame(rows)
        self.tables["roi"] = table
        return table

    def review_priority_list(self) -> pd.DataFrame:
        """Flagged images in the order a reviewer should open them."""
        self._need(self.flagged, "flag()")
        cols = ["image_path", "class_id", "class_name", "review_score", "review_tier",
                "primary_reason", "brightness", "contrast", "sharpness", "min_side"]
        out = self.flagged[self.flagged["review_flag"]]
        return out.sort_values(["review_score", "sharpness"], ascending=[False, True])[cols].reset_index(drop=True)

    # ---- saving -----------------------------------------------------------
    def save_tables(self) -> list:
        """Write all derived tables to ``outputs/processed`` and return the paths."""
        self._need(self.flagged, "flag()")
        for q in (self.q1_class_counts, self.q2_class_profiles, self.q3_flag_rates,
                  self.q4_associations, self.roi_comparison):
            q()
        overall, by_class = self.tables["q4_overall"], self.tables["q4_by_class"]
        files = {
            "validation_checks.csv": self.checks,
            "exclusion_log.csv": self.exclusion_log,
            "skipped_images.csv": self.skipped,
            "full_dataset_class_counts.csv": self.tables["q1"],
            "sample_features_tidy.csv": self.flagged.drop(columns=["x1", "y1", "x2", "y2"]),
            "review_priority_list.csv": self.review_priority_list(),
            "sensitivity_summary.csv": self.sensitivity_table(),
            "flag_rate_by_class.csv": self.tables["q3"],
            "class_profiles.csv": self.tables["q2"],
            "associations_overall.csv": overall.reset_index(names="measure"),
            "associations_by_class.csv": by_class,
            "roi_vs_full_image.csv": self.tables["roi"],
            "track_summary.csv": self.track_summary(),
            "example_images.csv": self.examples[["image_path", "class_id", "class_name", "illustrates",
                                                 "review_score", "brightness", "contrast", "sharpness"]],
        }
        paths = []
        for name, table in files.items():
            path = self.proc_dir / name
            table.to_csv(path, index=False)
            paths.append(path)
        return paths

    # ---- plotting ---------------------------------------------------------
    def _finish(self, fig, name: str, caption: str, hspace: float = None, top: float = None) -> Path:
        """Add a caption, save the figure, and remember the caption text."""
        wrapped = textwrap.fill(caption, 135)
        lines = wrapped.count("\n") + 1
        fig.tight_layout(rect=(0, 0.035 * lines + 0.02, 1, 1))
        if hspace is not None:  # extra row spacing for image grids with two-line titles
            fig.subplots_adjust(hspace=hspace)
        if top is not None:  # room above the axes for multi-line titles
            fig.subplots_adjust(top=top)
        fig.text(0.01, 0.01, wrapped, ha="left", va="bottom", fontsize=8, style="italic")
        path = self.fig_dir / f"{name}.png"
        fig.savefig(path, dpi=150)
        self.captions[name] = caption
        return path

    def _class_order(self):
        """Selected classes in ID order with their short plot labels."""
        names = self.flagged.groupby("class_id")["class_name"].first()
        return self.classes, [_short_label(c, names[c]) for c in self.classes]

    def fig_class_counts(self) -> Path:
        """Figure 1: full-dataset image count for every class (selected highlighted)."""
        t = self.q1_class_counts().sort_values("class_id", ascending=False)
        colors = ["#1f77b4" if s else "#c9c9c9" for s in t["selected"]]
        fig, ax = plt.subplots(figsize=(9, 10))
        ax.barh([_short_label(c, n, 36) for c, n in zip(t["class_id"], t["class_name"])],
                t["annotation_rows"], color=colors)
        ax.set_xlabel("Annotation rows (images) in the full training set")
        ax.set_ylabel("Sign class")
        ax.set_title("Full-dataset image count per class")
        ax.tick_params(axis="y", labelsize=7)
        ax.legend(handles=[Patch(color="#1f77b4", label="selected for sampled analysis"),
                           Patch(color="#c9c9c9", label="not selected")], loc="lower right")
        sel = t[t["selected"]]
        cap = (f"Counts use every annotation row (full dataset, {int(t['annotation_rows'].sum()):,} rows). "
               f"Among the {len(sel)} selected classes the largest has {int(sel['annotation_rows'].max()):,} "
               f"images and the smallest {int(sel['annotation_rows'].min()):,}.")
        return self._finish(fig, "fig1_class_counts", cap)

    def fig_distributions(self) -> Path:
        """Figure 2: brightness and sharpness distributions with cutoff lines."""
        f, cut = self.flagged, self.cutoffs
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        axes[0].hist(f["brightness"], bins=40, color="#7a9cc6", edgecolor="white")
        axes[0].axvline(cut["dark"], color="#c0392b", ls="--", label=f"dark cutoff = {cut['dark']:.1f}")
        axes[0].axvline(cut["overexposed"], color="#e67e22", ls="--", label=f"overexposed cutoff = {cut['overexposed']:.1f}")
        axes[0].set(xlabel="Brightness (mean grayscale intensity, 0-255)", ylabel="Number of sampled images",
                    title="Brightness distribution")
        axes[0].legend(fontsize=8)
        axes[1].hist(_log_sharpness(f["sharpness"]), bins=40, color="#7a9cc6", edgecolor="white")
        axes[1].axvline(_log_sharpness(cut["blurry"]), color="#c0392b", ls="--",
                        label=f"blurry cutoff = {cut['blurry']:.1f}")
        axes[1].set(xlabel="Sharpness: log10(1 + variance of Laplacian)", ylabel="Number of sampled images",
                    title="Sharpness distribution")
        axes[1].legend(fontsize=8)
        low, high = self.settings["base"]
        cap = (f"Cutoffs are the {low}th/{high}th percentiles of the {len(f):,} sampled images "
               "(project-specific triage settings, not safety standards). Sharpness is a proxy, not a readability score.")
        return self._finish(fig, "fig2_distributions", cap)

    def fig_boxplots(self) -> Path:
        """Figure 3: class-wise boxplots of brightness, contrast and sharpness."""
        f = self.flagged
        classes, labels = self._class_order()
        spec = [("brightness", "Brightness (0-255)", f["brightness"]),
                ("contrast", "Contrast (std of grayscale)", f["contrast"]),
                ("sharpness", "Sharpness: log10(1 + Laplacian variance)", pd.Series(_log_sharpness(f["sharpness"]), index=f.index))]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True)
        for ax, (key, xlabel, values) in zip(axes, spec):
            data = [values[f["class_id"] == c] for c in classes]
            ax.boxplot(data, vert=False, tick_labels=labels, showfliers=False)
            ax.set(xlabel=xlabel, title=key.capitalize())
            ax.tick_params(axis="y", labelsize=8)
        axes[0].set_ylabel("Sign class")
        cap = ("Boxes show the median and interquartile range per class; whiskers cover 1.5 x IQR and outliers are hidden. "
               f"Each class contributes at most {self.per_class} sampled images.")
        return self._finish(fig, "fig3_class_boxplots", cap)

    def fig_flag_rate(self) -> Path:
        """Figure 4: percentage of images flagged per class under each setting."""
        rates = self.q3_flag_rates().set_index("class_id").loc[self.classes]
        names = list(self.settings)
        y = np.arange(len(self.classes))
        height = 0.8 / len(names)
        fig, ax = plt.subplots(figsize=(9, 5.2))
        for i, name in enumerate(names):
            lo, hi = self.settings[name]
            ax.barh(y + i * height, rates[f"pct_flagged_{name}"], height, label=f"{name} ({lo}/{hi} pct)")
        ax.set_yticks(y + height * (len(names) - 1) / 2,
                      [_short_label(c, rates.loc[c, "class_name"]) for c in self.classes])
        ax.invert_yaxis()
        ax.set_xlim(0, float(rates[[f"pct_flagged_{n}" for n in names]].to_numpy().max()) * 1.3)
        ax.set(xlabel="Images flagged for review (%)", ylabel="Sign class",
               title="Review-flag rate by class under three cutoff settings")
        ax.legend(title="Setting (low/high percentile)", fontsize=8)
        agree = self.rank_agreement().set_index("setting")["spearman_rank_vs_base"].to_dict()
        txt = "; ".join(f"{k} vs base rho={v}" for k, v in agree.items() if k != "base")
        cap = f"A flag is a triage signal, not a safety verdict. Class-ranking agreement (Spearman): {txt}."
        return self._finish(fig, "fig4_flag_rate_by_class", cap)

    def fig_size_contrast(self) -> Path:
        """Figure 5: image size and contrast by class."""
        f = self.flagged
        classes, labels = self._class_order()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
        axes[0].boxplot([f.loc[f["class_id"] == c, "min_side"] for c in classes], vert=False,
                        tick_labels=labels, showfliers=False)
        axes[0].set(xlabel="Shorter image side (pixels)", ylabel="Sign class", title="Image size")
        axes[1].boxplot([f.loc[f["class_id"] == c, "contrast"] for c in classes], vert=False,
                        tick_labels=labels, showfliers=False)
        axes[1].set(xlabel="Contrast (std of grayscale intensity)", title="Contrast")
        for ax in axes:
            ax.tick_params(axis="y", labelsize=8)
        cap = ("Image size is the shorter side of the stored image in pixels; GTSRB images vary from about "
               "15 to 250 pixels. Whiskers cover 1.5 x IQR; outliers hidden.")
        return self._finish(fig, "fig5_size_and_contrast", cap)

    def fig_associations(self) -> Path:
        """Figure 6: sharpness against contrast and brightness (association only)."""
        f = self.flagged
        overall, _ = self.q4_associations()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        for ax, x, xl in [(axes[0], "contrast", "Contrast (std of grayscale intensity)"),
                          (axes[1], "brightness", "Brightness (mean grayscale intensity)")]:
            ax.scatter(f[x], _log_sharpness(f["sharpness"]), s=7, alpha=0.35, color="#34699a")
            ax.set(xlabel=xl, ylabel="Sharpness: log10(1 + Laplacian variance)",
                   title=f"Sharpness vs {x}  (Spearman rho = {overall.loc['sharpness', x]:.2f})")
        cap = ("Each dot is one sampled image. Spearman rho describes monotonic association only; "
               "it does not show that one property causes another.")
        return self._finish(fig, "fig6_sharpness_associations", cap)

    def fig_contact_sheet(self, n: int = 6) -> Path:
        """Figure 7: contact sheet of flagged (top row) and unflagged (bottom row) images."""
        f = self.flagged
        rng = np.random.default_rng(self.seed)
        rows = []
        for label, pool in [("Flagged", f[f["review_flag"]]), ("Unflagged", f[~f["review_flag"]])]:
            take = rng.choice(len(pool), size=min(n, len(pool)), replace=False) if len(pool) else []
            rows.append((label, pool.iloc[sorted(take)]))
        fig, axes = plt.subplots(2, n, figsize=(2.1 * n, 5.6))
        for r, (label, pick) in enumerate(rows):
            for c in range(n):
                ax = axes[r][c]
                ax.axis("off")
                if c >= len(pick):
                    continue
                item = pick.iloc[c]
                img = load_rgb(self.data_dir / item["image_path"])
                if img is not None:
                    ax.imshow(img)
                reason = item["primary_reason"] if item["review_flag"] else "not flagged"
                ax.set_title(f"class {int(item['class_id'])} | {reason}\nB={item['brightness']:.0f}  "
                             f"C={item['contrast']:.0f}  S={item['sharpness']:.0f}", fontsize=7)
            axes[r][0].text(-0.08, 0.5, label, rotation=90, transform=axes[r][0].transAxes,
                            ha="right", va="center", fontsize=10, fontweight="bold")
        fig.suptitle("Sample contact sheet: flagged vs unflagged images", y=0.995)
        cap = (f"Random draw (seed {self.seed}) from the sampled images. B = brightness, C = contrast, "
               "S = sharpness (variance of Laplacian). Images are shown in RGB after converting from OpenCV's BGR order.")
        return self._finish(fig, "fig7_contact_sheet", cap, hspace=0.45)

    def fig_examples(self) -> Path:
        """Figure 8: three images that best illustrate the rule."""
        ex, cut = self.examples, self.cutoffs
        fig, axes = plt.subplots(1, len(ex), figsize=(4.2 * len(ex), 4.8), squeeze=False)
        for ax, (_, item) in zip(axes[0], ex.iterrows()):
            img = load_rgb(self.data_dir / item["image_path"])
            if img is not None:
                ax.imshow(img)
                if item["roi_valid"]:
                    ax.add_patch(Rectangle((item["x1"] - 0.5, item["y1"] - 0.5),
                                           item["x2"] - item["x1"] + 1, item["y2"] - item["y1"] + 1,
                                           fill=False, edgecolor="#e74c3c", lw=1.5))
            ax.set_xticks([]); ax.set_yticks([])
            triggered = ", ".join(c for c in CONDITIONS if item[f"flag_{c}"])
            ax.set_title(f"Class {int(item['class_id'])}: {item['class_name']}\n"
                         f"illustrates: {item['illustrates']} (score {int(item['review_score'])})\n"
                         f"conditions met: {triggered}", fontsize=8)
            ax.set_xlabel(f"brightness {item['brightness']:.1f} (cutoffs {cut['dark']:.1f} / {cut['overexposed']:.1f})\n"
                          f"contrast {item['contrast']:.1f} (cutoff {cut['low_contrast']:.1f})\n"
                          f"sharpness {item['sharpness']:.1f} (cutoff {cut['blurry']:.1f})", fontsize=8)
        cap = ("Three different images: the darkest flagged image, the blurriest flagged image that is not dark, and the "
               "brightest flagged image. The red box is the annotated region of interest. Measurements are shown "
               "next to the base-rule cutoffs.")
        return self._finish(fig, "fig8_rule_examples", cap, top=0.80)

    def fig_review_groups(self, n: int = 6) -> Path:
        """Figure 9: random images from three groups used to judge false alarms and misses."""
        f, cut = self.flagged, self.cutoffs
        rng = np.random.default_rng(self.seed)
        groups = [
            ("A. Flagged only as\noverexposed", f[f["flag_overexposed"] & (f["review_score"] == 1)]),
            ("B. High priority\n(score 3 or 4)", f[f["review_score"] >= 3]),
            ("C. Not flagged, sharpness\njust above the cutoff",
             f[(~f["review_flag"]) & (f["sharpness"] <= 1.5 * cut["blurry"])]),
        ]
        fig, axes = plt.subplots(len(groups), n, figsize=(2.1 * n, 2.4 * len(groups) + 0.6))
        sizes = []
        for r, (label, pool) in enumerate(groups):
            sizes.append(len(pool))
            take = sorted(rng.choice(len(pool), size=min(n, len(pool)), replace=False)) if len(pool) else []
            pick = pool.iloc[take]
            for c in range(n):
                ax = axes[r][c]
                ax.axis("off")
                if c >= len(pick):
                    continue
                item = pick.iloc[c]
                img = load_rgb(self.data_dir / item["image_path"])
                if img is not None:
                    ax.imshow(img)
                ax.set_title(f"class {int(item['class_id'])}\nB={item['brightness']:.0f}  C={item['contrast']:.0f}  "
                             f"S={item['sharpness']:.0f}", fontsize=7)
            axes[r][0].text(-0.08, 0.5, label, rotation=90, transform=axes[r][0].transAxes,
                            ha="right", va="center", fontsize=8, fontweight="bold")
        fig.suptitle("Checking false alarms and possible misses", y=0.995)
        cap = (f"Random draws (seed {self.seed}) of up to {n} images from each group. Group sizes in the sample: "
               f"A = {sizes[0]}, B = {sizes[1]}, C = {sizes[2]} images. B = brightness, C = contrast, S = sharpness.")
        return self._finish(fig, "fig9_review_groups", cap, hspace=0.5)

    def all_figures(self) -> list:
        """Create and save every figure; return the file paths."""
        return [self.fig_class_counts(), self.fig_distributions(), self.fig_boxplots(),
                self.fig_flag_rate(), self.fig_size_contrast(), self.fig_associations(),
                self.fig_contact_sheet(), self.fig_examples(), self.fig_review_groups()]

    # ---- convenience ------------------------------------------------------
    def run_all(self):
        """Run the full pipeline from raw annotations to tables and figures."""
        self.load_annotations()
        self.validate()
        self.select_sample()
        self.extract_features()
        self.flag()
        self.save_tables()
        return self.all_figures()

    @staticmethod
    def _need(obj, step: str) -> None:
        if obj is None:
            raise RuntimeError(f"{step} has not been run yet; it must come before this step.")


# ----------------------------------------------------------------------------
# Command-line entry point: python src/traffic_audit.py
# ----------------------------------------------------------------------------
def main(argv=None) -> None:
    """Rebuild every table and figure from the raw annotations."""
    import argparse
    import matplotlib

    matplotlib.use("Agg")
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Traffic Sign Visibility Audit")
    parser.add_argument("--data-dir", default=root / "data", type=Path)
    parser.add_argument("--output-dir", default=root / "outputs", type=Path)
    parser.add_argument("--seed", default=DEFAULT_SEED, type=int)
    args = parser.parse_args(argv)
    audit = TrafficSignAudit(args.data_dir, args.output_dir, seed=args.seed)
    figures = audit.run_all()
    print(audit.checks.to_string(index=False))
    print(f"\nSampled images measured: {len(audit.features)}; skipped: {len(audit.skipped)}")
    print(f"Saved {len(figures)} figures to {audit.fig_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# =========================================================
# FULL RUNNABLE SCRIPT (FXL): Compare pseudo-absence methods
#   1) RANDOM      (random CellIDs from full network, excluding presence cells)
#   2) CONTROLLED  (quartile/Mq selection from UNLABELLED cells)
#   3) LLM3        (CONTROLLED shortlist -> LLM ranks -> band-cap -> CONTROLLED selection)
#
# Evaluation:
#   - CellID aggregation
#   - CV metrics (RF): acc, f1, auc, tss
#   - Predict all cells for each scenario
#   - Distributions + histograms + scatters + Spearman correlations
#   - Mahalanobis (neg pools vs presence profile)
#   - HSA sizes + overlaps (CellID-level)
#   - Table2 predictor-space distances (FID-level) + FID-level overlaps
#
# IMPORTANT FIX INCLUDED:
#   Merge can create is_labelled_cell_cell (if your NETWORK already has is_labelled_cell).
#   This script forces suffixes and always uses CellID-level labels.
#
# NOTE:
#   Your NETWORK needs either:
#     - has_FXL_PREZ / has_FXL_TRUEABS
#     OR
#     - FXL_PREZ / FXL_TRUEABS
# =========================================================

from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyproj import Transformer
from scipy.stats import spearmanr

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


# -----------------------
# 0) EDIT PATHS
# -----------------------
NETWORK_CSV = "/Users/kristianmiok/Desktop/Lucian/LLM/Data/After Review/NETWORK_with_CellID_FIDlevel.csv"
OUT_DIR = "/Users/kristianmiok/Desktop/Lucian/LLM/Data/After Review/New/FXL Results/out_step3_compare_random_controlled_llm3"
os.makedirs(OUT_DIR, exist_ok=True)

# Species + predictors
SPECIES = "FXL"
PREFERRED_PREDICTORS = ["RWQ", "ALT", "FFP", "BIO1"]

# If CellID missing in CSV, compute from lon/lat
CELL_KM = 5

# Quartile directions (your convention)
# RWQ: MAX->0 => ascending=False
# ALT: 0->MAX => ascending=True
# FFP: 0->MAX => ascending=True
# BIO1: MAX->0 => ascending=False
QUARTILE_DIR = {"RWQ": False, "ALT": True, "FFP": True, "BIO1": False}

NEG_MULT = 1
SEED = 42

# RF / CV
N_SPLITS = 5
RF_PARAMS = dict(
    n_estimators=1000,
    max_features="sqrt",
    random_state=SEED,
    n_jobs=-1
)

# Training set sizing
BALANCE_TO_N_POS = True
FORCE_N_NEG_TRUE = None
FORCE_N_NEG_RANDOM = None
FORCE_N_NEG_CTRL = None
FORCE_N_NEG_LLM3 = None

# Evaluation settings
HSA_THR = 0.5
RIDGE = 1e-6

# Probability columns
COL_TRUE = "prob_TRUE"
COL_RAND = "prob_RANDOM"
COL_CTRL = "prob_CONTROLLED"
COL_LLM3 = "prob_LLM3"

# -----------------------
# LLM3 settings (only)
# -----------------------
USE_LLM = True
LLM_MODEL = "gpt-5.2"
LLM_BATCH_SIZE = 80

# Band-cap to avoid extremes
LLM_BAND_LOW_Q = 0.70
LLM_BAND_HIGH_Q = 0.90
MIN_BAND_POOL_MULT = 2.0

SHORTLIST_MULT = 3  # shortlist_n = target_n * max(2, SHORTLIST_MULT)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()

# Eco narrative for FXL (edit as you want)
ECOLOGY_TEXT = (
    "Faxonius limosus (FXL), spinycheek crayfish, is an invasive crayfish in many European waters. "
    "It is generally more tolerant of warmer temperatures, lower water quality, and habitat disturbance "
    "than native crayfish. It can persist in a wider range of conditions, including impacted and slower-flowing "
    "or modified river sections."
)

try:
    from openai import OpenAI  # pip install openai
except Exception:
    OpenAI = None


# -----------------------
# 1) HELPERS
# -----------------------
def pick_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None


def assert_has_cellid(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    if "CellID" not in df.columns:
        low = {c.lower(): c for c in df.columns}
        if "cellid" in low:
            df = df.rename(columns={low["cellid"]: "CellID"})
    if "CellID" not in df.columns:
        raise ValueError(f"{name}: 'CellID' not found. Columns: {list(df.columns)}")
    return df


def normalize_cellid(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    return s


def pick_label_col(cols: list[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in cols:
            return c
    return None


def make_cellid_from_lonlat(lon: np.ndarray, lat: np.ndarray, cell_km: int) -> List[str]:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    x_m, y_m = transformer.transform(lon, lat)
    cell_size_m = int(cell_km) * 1000
    tx = np.floor(np.asarray(x_m) / cell_size_m).astype(int)
    ty = np.floor(np.asarray(y_m) / cell_size_m).astype(int)
    return [f"C_{a}_{b}" for a, b in zip(tx, ty)]


def aggregate_to_cellid(net_fid: pd.DataFrame, predictors: list[str], sp: str) -> pd.DataFrame:
    net_fid = assert_has_cellid(net_fid, "NETWORK").copy()
    net_fid["CellID"] = normalize_cellid(net_fid["CellID"])

    cols = list(net_fid.columns)
    prez_src = pick_label_col(cols, [f"has_{sp}_PREZ", f"{sp}_PREZ"])
    abs_src = pick_label_col(cols, [f"has_{sp}_TRUEABS", f"{sp}_TRUEABS"])

    if prez_src is None:
        raise ValueError(f"Could not find presence column for {sp}. Tried has_{sp}_PREZ / {sp}_PREZ.")
    if abs_src is None:
        raise ValueError(f"Could not find true-absence column for {sp}. Tried has_{sp}_TRUEABS / {sp}_TRUEABS.")

    net_fid[prez_src] = pd.to_numeric(net_fid[prez_src], errors="coerce").fillna(0).astype(int)
    net_fid[abs_src] = pd.to_numeric(net_fid[abs_src], errors="coerce").fillna(0).astype(int)

    agg: dict[str, str] = {}
    for p in predictors:
        if p in net_fid.columns:
            agg[p] = "median"
    for c in ["X_WGS84_DD", "Y_WGS84_DD"]:
        if c in net_fid.columns:
            agg[c] = "median"

    agg[prez_src] = "max"
    agg[abs_src] = "max"

    cell = net_fid.groupby("CellID", as_index=False).agg(agg)
    cell = cell.rename(columns={prez_src: f"has_{sp}_PREZ", abs_src: f"has_{sp}_TRUEABS"})
    cell[f"has_{sp}_PREZ"] = cell[f"has_{sp}_PREZ"].astype(int).astype(bool)
    cell[f"has_{sp}_TRUEABS"] = cell[f"has_{sp}_TRUEABS"].astype(int).astype(bool)
    cell["is_labelled_cell"] = (cell[f"has_{sp}_PREZ"] | cell[f"has_{sp}_TRUEABS"]).astype(bool)
    return cell


def safe_qcut_4(x: pd.Series, ascending: bool) -> Tuple[pd.Series, List[float]]:
    s = pd.to_numeric(x, errors="coerce")
    if s.notna().sum() < 10:
        return pd.Series([pd.NA] * len(s), index=s.index, dtype="Int64"), []

    v = s.copy()
    if not ascending:
        v = -v

    try:
        q = pd.qcut(v, q=4, labels=[1, 2, 3, 4], duplicates="drop").astype("Int64")
    except Exception:
        qs = v.quantile([0, 0.25, 0.5, 0.75, 1.0]).to_list()
        qs = [float(z) for z in qs]
        eps = 1e-12
        for i in range(1, len(qs)):
            if qs[i] <= qs[i - 1]:
                qs[i] = qs[i - 1] + eps
        q = pd.cut(v, bins=qs, labels=[1, 2, 3, 4], include_lowest=True).astype("Int64")

    cuts = []
    try:
        orig = s.dropna()
        cuts = orig.quantile([0, 0.25, 0.5, 0.75, 1.0]).to_list()
        cuts = [float(z) for z in cuts]
    except Exception:
        cuts = []
    return q, cuts


def mq_scores(df: pd.DataFrame, q_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    Q = out[q_cols].copy()
    denom = Q.notna().sum(axis=1).replace(0, np.nan)
    for k in [1, 2, 3, 4]:
        out[f"Mq{k}"] = (Q.eq(k).sum(axis=1) / denom).astype(float)
    return out


def select_top_per_quartile_dominant(
    df: pd.DataFrame,
    cell_col: str,
    target_n: int,
    seed: int,
    score_prefix: str = "Mq",
    tiebreak_cols: Optional[List[str]] = None,
    tiebreak_desc: Optional[List[bool]] = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = df.copy()

    if tiebreak_cols is None:
        tiebreak_cols = []
    if tiebreak_desc is None:
        tiebreak_desc = [True] * len(tiebreak_cols)

    base = target_n // 4
    rem = target_n - 4 * base
    k_targets = {1: base, 2: base, 3: base, 4: base}
    for k in [1, 2, 3, 4][:rem]:
        k_targets[k] += 1

    used_cells: set[str] = set()
    picks: List[pd.DataFrame] = []

    for k in [1, 2, 3, 4]:
        score_col = f"{score_prefix}{k}"
        sub = d.dropna(subset=[score_col]).copy()
        if sub.empty:
            continue

        sub["_Mq_max"] = sub[[f"{score_prefix}1", f"{score_prefix}2", f"{score_prefix}3", f"{score_prefix}4"]].max(axis=1)
        sub = sub[sub[score_col] >= sub["_Mq_max"] - 1e-12].drop(columns=["_Mq_max"], errors="ignore")
        if sub.empty:
            continue

        sub["_jitter"] = rng.normal(0, 1e-9, size=len(sub))

        sort_cols = [score_col] + tiebreak_cols + ["_jitter"]
        sort_asc = [False] + [not dsc for dsc in tiebreak_desc] + [False]
        sub = sub.sort_values(sort_cols, ascending=sort_asc).drop(columns=["_jitter"], errors="ignore")

        chosen_k = []
        for _, row in sub.iterrows():
            cid = str(row[cell_col])
            if cid in used_cells:
                continue
            chosen_k.append(row)
            used_cells.add(cid)
            if len(chosen_k) >= k_targets[k]:
                break

        if chosen_k:
            picks.append(pd.DataFrame(chosen_k))

    selected = pd.concat(picks, ignore_index=True) if picks else d.head(0).copy()

    if len(selected) < target_n:
        d2 = d.copy()
        d2["Mq_max"] = d2[[f"{score_prefix}1", f"{score_prefix}2", f"{score_prefix}3", f"{score_prefix}4"]].max(axis=1)
        d2 = d2.dropna(subset=["Mq_max"]).sort_values("Mq_max", ascending=False)

        need = target_n - len(selected)
        fill_rows = []
        for _, row in d2.iterrows():
            cid = str(row[cell_col])
            if cid in used_cells:
                continue
            fill_rows.append(row)
            used_cells.add(cid)
            if len(fill_rows) >= need:
                break

        if fill_rows:
            selected = pd.concat([selected, pd.DataFrame(fill_rows)], ignore_index=True)

    return selected.head(target_n).drop_duplicates(subset=[cell_col], keep="first").copy()


def sample_neg(pool: pd.DataFrame, n: int, seed=42) -> pd.DataFrame:
    if pool.empty:
        return pool
    n2 = min(int(n), len(pool))
    if n2 < n:
        print(f"[WARN] Requested {n} negatives but pool has only {len(pool)}. Capping to {n2} (no replacement).")
    return pool.sample(n=n2, replace=False, random_state=seed).copy()


def tss_at_05(y_true, y_prob, thr=0.5) -> float:
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return sens + spec - 1.0


def run_cv(df: pd.DataFrame, predictors: list[str], n_splits=5) -> dict:
    X = df[predictors].to_numpy()
    y = df["presence"].to_numpy().astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    accs, f1s, aucs, tsss = [], [], [], []
    for tr, te in skf.split(X, y):
        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X[tr], y[tr])
        prob = clf.predict_proba(X[te])[:, 1]
        pred = (prob >= 0.5).astype(int)
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred))
        try:
            aucs.append(roc_auc_score(y[te], prob))
        except ValueError:
            pass
        tsss.append(tss_at_05(y[te], prob, thr=0.5))

    def ms(a):
        a = np.asarray(a, dtype=float)
        return float(np.mean(a)), float(np.std(a))

    acc_m, acc_s = ms(accs)
    f1_m, f1_s = ms(f1s)
    tss_m, tss_s = ms(tsss)
    if len(aucs) > 0:
        auc_m, auc_s = ms(aucs)
    else:
        auc_m, auc_s = np.nan, np.nan

    return dict(
        acc_mean=acc_m, acc_sd=acc_s,
        f1_mean=f1_m, f1_sd=f1_s,
        auc_mean=auc_m, auc_sd=auc_s,
        tss_mean=tss_m, tss_sd=tss_s,
    )


def train_and_predict(train_df: pd.DataFrame, all_cells: pd.DataFrame, predictors: list[str]) -> np.ndarray:
    X_tr = train_df[predictors].to_numpy()
    y_tr = train_df["presence"].astype(int).to_numpy()
    clf = RandomForestClassifier(**RF_PARAMS)
    clf.fit(X_tr, y_tr)
    X_all = all_cells[predictors].to_numpy()
    return clf.predict_proba(X_all)[:, 1]


def mahalanobis_dists(X: np.ndarray, mu: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    D = X - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", D, inv_cov, D))


def summarize_dist(name: str, d: np.ndarray) -> dict:
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return {"set": name, "n": 0}
    return {
        "set": name,
        "n": int(len(d)),
        "mean": float(d.mean()),
        "median": float(np.median(d)),
        "p10": float(np.percentile(d, 10)),
        "p90": float(np.percentile(d, 90)),
        "p95": float(np.percentile(d, 95)),
        "max": float(np.max(d)),
    }


def overlap_pct(A: set, B: set) -> float:
    if not A and not B:
        return float("nan")
    inter = len(A.intersection(B))
    uni = len(A.union(B))
    return 100.0 * inter / uni if uni else float("nan")


# -----------------------
# 2) LLM3 helpers (only)
# -----------------------
def _json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def call_gpt_json(
    client,
    input_text: str,
    cache_path: Path,
    model: str,
    temperature: float = 0.0,
    max_retries: int = 3
) -> dict:
    if cache_path.exists():
        return _json_load(cache_path)

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            res = client.responses.create(model=model, input=input_text, temperature=temperature)
            txt = getattr(res, "output_text", None)
            if txt is None:
                txt = str(res)
            obj = json.loads(txt)
            _json_dump(obj, cache_path)
            return obj
        except Exception as e:
            last_err = e
            time.sleep(0.5 * attempt)
    raise RuntimeError(f"GPT call failed: {last_err}")


def presence_profile_stats(pos: pd.DataFrame, predictors: List[str]) -> dict:
    out = {}
    for p in predictors:
        x = pd.to_numeric(pos[p], errors="coerce")
        out[p] = {
            "median": float(np.nanmedian(x)),
            "q25": float(np.nanpercentile(x, 25)),
            "q75": float(np.nanpercentile(x, 75)),
            "min": float(np.nanmin(x)),
            "max": float(np.nanmax(x)),
        }
    return out


def band_cap(df: pd.DataFrame, score_col: str, target_n: int, low_q: float, high_q: float, min_mult: float) -> pd.DataFrame:
    if score_col not in df.columns or df.empty:
        return df
    s = pd.to_numeric(df[score_col], errors="coerce")
    if s.notna().sum() < 10:
        return df

    def _cap(lq, hq):
        lo = float(s.quantile(lq))
        hi = float(s.quantile(hq))
        keep = (s >= lo) & (s <= hi)
        return df.loc[keep].copy()

    out = _cap(low_q, high_q)
    need_min = int(np.ceil(min_mult * target_n))
    if len(out) >= need_min:
        return out

    for (lq, hq) in [(0.60, 0.92), (0.50, 0.95), (0.40, 0.97), (0.0, 1.0)]:
        out2 = _cap(lq, hq)
        if len(out2) >= need_min:
            return out2
    return df


def llm3_rank_shortlist(
    client,
    cache_dir: Path,
    tag: str,
    predictors: List[str],
    profile: dict,
    shortlist: pd.DataFrame,
    batch_size: int = 80
) -> pd.Series:
    scores = pd.Series(np.nan, index=shortlist.index, dtype=float, name="llm_score")

    extra_cols = [c for c in ["X_WGS84_DD", "Y_WGS84_DD", "THS", "TYS"] if c in shortlist.columns]
    items = shortlist[["CellID"] + predictors + extra_cols].copy()

    def _fmt_profile(profile_dict: dict, preds: List[str]) -> str:
        lines = []
        for p in preds:
            if p not in profile_dict:
                continue
            d = profile_dict[p]
            med, q25, q75, mn, mx = d.get("median"), d.get("q25"), d.get("q75"), d.get("min"), d.get("max")
            if any(v is None for v in [med, q25, q75, mn, mx]):
                continue
            lines.append(
                f"- {p}: median={med:.3f}, IQR=[{q25:.3f}, {q75:.3f}], range=[{mn:.3f}, {mx:.3f}]"
            )
        return "\n".join(lines) if lines else "- (profile unavailable)"

    def _narrative_row(row: pd.Series) -> str:
        cid = str(row.get("CellID", "NA"))

        coord_part = ""
        if "Y_WGS84_DD" in row.index and "X_WGS84_DD" in row.index:
            try:
                lat = row.get("Y_WGS84_DD", None)
                lon = row.get("X_WGS84_DD", None)
                if pd.notna(lat) and pd.notna(lon):
                    coord_part = f" (lat={float(lat):.5f}, lon={float(lon):.5f})"
            except Exception:
                coord_part = ""

        pred_bits = []
        for p in predictors:
            v = row.get(p, None)
            try:
                if pd.notna(v):
                    pred_bits.append(f"{p}≈{float(v):.3f}")
            except Exception:
                pass
        preds_part = ", ".join(pred_bits) if pred_bits else "predictors unavailable"

        local_bits = []
        for c in ["THS", "TYS"]:
            if c in row.index:
                v = row.get(c, None)
                if pd.notna(v):
                    local_bits.append(f"{c}='{v}'")
        local_part = f" | Local: {', '.join(local_bits)}" if local_bits else ""

        return f"- CellID {cid}{coord_part}: {preds_part}{local_part}"

    presence_profile_text = _fmt_profile(profile, predictors)

    for bi, start in enumerate(range(0, len(items), batch_size)):
        batch = items.iloc[start:start + batch_size].copy()
        cache_path = cache_dir / f"llm3_{tag}_batch{bi:03d}.json"

        if cache_path.exists():
            out = _json_load(cache_path)
        else:
            narratives = "\n".join(_narrative_row(r) for _, r in batch.iterrows())

            prompt = f"""
Task: Rank candidate river-network cells as pseudo-absences for species {SPECIES} by UNSUITABILITY.

Ecological narrative (expert prior):
{ECOLOGY_TEXT}

Observed presence profile (summary statistics from known presence cells):
{presence_profile_text}

What to do:
- For each candidate CellID, assign a score from 0..100 where:
  - 100 = very confident UNSUITABLE / strongly incompatible with the species ecology (excellent pseudo-absence)
  - 0   = presence-like / likely suitable habitat (poor pseudo-absence)
- Use ecological reasoning: penalize candidates that resemble presence conditions across multiple predictors.
- Prefer candidates with multi-predictor patterns atypical vs the presence profile (e.g., outside IQR/range, or conflicting combination).
- Do NOT add explanations in the output.

Return STRICT JSON ONLY in exactly this format:
{{
  "scores": [{{"CellID":"...", "score": 0-100}}, ...]
}}

Candidates (eco-narrative lines):
{narratives}
""".strip()

            out = call_gpt_json(client, prompt, cache_path, model=LLM_MODEL)

        m = {
            str(x.get("CellID")): float(x.get("score"))
            for x in out.get("scores", [])
            if x.get("CellID") is not None
        }

        for idx, row in batch.iterrows():
            cid = str(row["CellID"])
            if cid in m:
                scores.loc[idx] = m[cid]

    return scores.fillna(0.0)


# -----------------------
# 3) LOAD NETWORK + AGGREGATE (CellID-level)
# -----------------------
net_fid = pd.read_csv(NETWORK_CSV)
net_fid.columns = [c.strip() for c in net_fid.columns]

# If CellID missing, create it
if "CellID" not in net_fid.columns:
    cols = list(net_fid.columns)
    x_col = pick_col(cols, ["X_WGS84_DD", "Longitude", "LONGITUDE", "lon", "Lon"])
    y_col = pick_col(cols, ["Y_WGS84_DD", "Latitude", "LATITUDE", "lat", "Lat"])
    if x_col is None or y_col is None:
        raise ValueError("CellID missing and could not find lon/lat columns to create it.")
    net_fid["CellID"] = make_cellid_from_lonlat(
        lon=pd.to_numeric(net_fid[x_col], errors="coerce").to_numpy(),
        lat=pd.to_numeric(net_fid[y_col], errors="coerce").to_numpy(),
        cell_km=CELL_KM,
    )

net_fid = assert_has_cellid(net_fid, "NETWORK")
net_fid["CellID"] = normalize_cellid(net_fid["CellID"])

predictors = [p for p in PREFERRED_PREDICTORS if p in net_fid.columns]
if not predictors:
    raise ValueError(f"No predictors found from {PREFERRED_PREDICTORS} in NETWORK. Columns: {list(net_fid.columns)}")

fid_col = pick_col(list(net_fid.columns), ["FID", "fid", "Fid", "ID", "Id"])
if fid_col is None:
    fid_col = "__ROWID__"
    net_fid[fid_col] = np.arange(len(net_fid), dtype=int)

cell = aggregate_to_cellid(net_fid, predictors=predictors, sp=SPECIES)
cell.to_csv(os.path.join(OUT_DIR, "cell_level_aggregated.csv"), index=False)

prez_col = f"has_{SPECIES}_PREZ"
abs_col = f"has_{SPECIES}_TRUEABS"

pres = cell[cell[prez_col]].copy()
pres["presence"] = 1

trueabs = cell[cell[abs_col]].copy()
trueabs["presence"] = 0

unlab_pool = cell[~cell["is_labelled_cell"]].copy()
unlab_pool["presence"] = 0

n_pos = int(len(pres))
if n_pos == 0:
    raise ValueError(f"No presence cells found for {SPECIES} (column {prez_col}).")

target_n = int(round(NEG_MULT * n_pos))
print(f"\n=== Presence cells: {n_pos} | target_n (NEG_MULT={NEG_MULT}): {target_n} ===")


# -----------------------
# 4) MERGE CellID-level labels into FID-level (ROBUST)
# -----------------------
cell_labels = cell[["CellID", prez_col, abs_col, "is_labelled_cell"]].copy()
cell_labels["CellID"] = normalize_cellid(cell_labels["CellID"])

tmp = net_fid.merge(
    cell_labels,
    on="CellID",
    how="left",
    validate="many_to_one",
    suffixes=("", "_cell"),
)

lab_col = "is_labelled_cell_cell" if "is_labelled_cell_cell" in tmp.columns else "is_labelled_cell"
prez_tmp = f"{prez_col}_cell" if f"{prez_col}_cell" in tmp.columns else prez_col
abs_tmp = f"{abs_col}_cell" if f"{abs_col}_cell" in tmp.columns else abs_col

tmp[lab_col] = tmp[lab_col].fillna(False).astype(bool)
tmp[prez_tmp] = tmp[prez_tmp].fillna(False).astype(bool)
tmp[abs_tmp] = tmp[abs_tmp].fillna(False).astype(bool)


# -----------------------
# 5) BUILD PSEUDO-ABSENCE POOLS (RANDOM, CONTROLLED, LLM3)
# -----------------------
# ----- RANDOM -----
presence_cells = set(cell_labels.loc[cell_labels[prez_col].astype(bool), "CellID"].astype(str))
all_cellids = tmp["CellID"].dropna().astype(str).drop_duplicates()
all_cellids = all_cellids[~all_cellids.isin(presence_cells)]

if len(all_cellids) < target_n:
    print(f"[WARN] Full network has only {len(all_cellids)} unique non-presence CellIDs; cannot sample {target_n}. Using all.")
    sampled_cellids = all_cellids
else:
    sampled_cellids = all_cellids.sample(n=target_n, replace=False, random_state=SEED)

rep_full = tmp.sort_values(["CellID", fid_col], ascending=[True, True]).drop_duplicates(subset=["CellID"], keep="first")
pa_random = rep_full[rep_full["CellID"].astype(str).isin(set(sampled_cellids))].copy()
pa_random = pa_random.drop_duplicates(subset=["CellID"], keep="first")
pa_random_path = os.path.join(OUT_DIR, "pa_random.csv")
pa_random.to_csv(pa_random_path, index=False)

# ----- CONTROLLED -----
candidate_fids = tmp[~tmp[lab_col]].copy()

max_unlabelled_cells = int(candidate_fids["CellID"].nunique())
if max_unlabelled_cells == 0:
    raise ValueError("No unlabelled candidate cells available for CONTROLLED/LLM3.")

target_n_eff = min(target_n, max_unlabelled_cells)
if target_n_eff < target_n:
    print(f"[WARN] target_n={target_n} > unlabelled unique CellIDs={max_unlabelled_cells}. Capping to {target_n_eff}.")
target_n = target_n_eff

q_cols = []
cut_rows = []
for p in predictors:
    asc = bool(QUARTILE_DIR.get(p, True))
    q, cuts = safe_qcut_4(candidate_fids[p], ascending=asc)
    qname = f"{p}quartile"
    candidate_fids[qname] = q
    q_cols.append(qname)
    cut_rows.append({
        "predictor": p,
        "direction": "0->MAX" if asc else "MAX->0",
        "q0_min": cuts[0] if len(cuts) == 5 else np.nan,
        "q25": cuts[1] if len(cuts) == 5 else np.nan,
        "q50": cuts[2] if len(cuts) == 5 else np.nan,
        "q75": cuts[3] if len(cuts) == 5 else np.nan,
        "q100_max": cuts[4] if len(cuts) == 5 else np.nan,
        "n_nonmissing": int(pd.to_numeric(candidate_fids[p], errors="coerce").notna().sum()),
    })

candidate_fids = mq_scores(candidate_fids, q_cols=q_cols)

pa_control = select_top_per_quartile_dominant(
    df=candidate_fids,
    cell_col="CellID",
    target_n=target_n,
    seed=SEED,
    score_prefix="Mq",
)
pa_control_path = os.path.join(OUT_DIR, "pa_controlled.csv")
pa_control.to_csv(pa_control_path, index=False)

pd.DataFrame(cut_rows).to_csv(os.path.join(OUT_DIR, "quartile_cutpoints.csv"), index=False)

# ----- LLM3 -----
llm_cache_dir = Path(OUT_DIR) / "llm_cache"
llm_cache_dir.mkdir(parents=True, exist_ok=True)

client = None
if USE_LLM:
    if OpenAI is None:
        raise RuntimeError("openai SDK not installed. Run: pip install openai")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing. Export it in your terminal.")
    client = OpenAI(api_key=OPENAI_API_KEY)

profile = presence_profile_stats(pres, predictors)

shortlist_n = min(len(candidate_fids), int(target_n * max(2, int(SHORTLIST_MULT))))
shortlist = select_top_per_quartile_dominant(
    df=candidate_fids,
    cell_col="CellID",
    target_n=shortlist_n,
    seed=SEED + 11,
    score_prefix="Mq",
).copy()

if USE_LLM:
    shortlist["llm_score"] = llm3_rank_shortlist(
        client=client,
        cache_dir=llm_cache_dir,
        tag=f"{SPECIES}_llm3",
        predictors=predictors,
        profile=profile,
        shortlist=shortlist,
        batch_size=LLM_BATCH_SIZE,
    )
else:
    shortlist["llm_score"] = 0.0

shortlist = band_cap(shortlist, "llm_score", target_n, LLM_BAND_LOW_Q, LLM_BAND_HIGH_Q, MIN_BAND_POOL_MULT)

pa_llm3 = select_top_per_quartile_dominant(
    df=shortlist,
    cell_col="CellID",
    target_n=target_n,
    seed=SEED + 23,
    score_prefix="Mq",
    tiebreak_cols=["llm_score"],
    tiebreak_desc=[True],
)
pa_llm3_path = os.path.join(OUT_DIR, "pa_llm3.csv")
pa_llm3.to_csv(pa_llm3_path, index=False)

print("\n=== Pseudo-absence files written ===")
print("RANDOM    :", pa_random_path, "rows:", len(pa_random))
print("CONTROLLED:", pa_control_path, "rows:", len(pa_control))
print("LLM3      :", pa_llm3_path, "rows:", len(pa_llm3))


# -----------------------
# 6) MAP PA FILES -> CellID-level pools
# -----------------------
s2_ids = set(pa_random["CellID"].astype(str).tolist())
s3_ids = set(pa_control["CellID"].astype(str).tolist())
s4_ids = set(pa_llm3["CellID"].astype(str).tolist())

s2_pool_all = cell[cell["CellID"].astype(str).isin(s2_ids)].copy()
s3_pool = cell[cell["CellID"].astype(str).isin(s3_ids)].copy()
s4_pool = cell[cell["CellID"].astype(str).isin(s4_ids)].copy()

# Safety: CONTROLLED and LLM3 must be unlabelled
s3_pool = s3_pool[~s3_pool["is_labelled_cell"]].copy()
s4_pool = s4_pool[~s4_pool["is_labelled_cell"]].copy()

# RANDOM: drop presences if any slipped in
s2_includes_pres = int(s2_pool_all[prez_col].sum())
s2_includes_trueabs = int(s2_pool_all[abs_col].sum())
s2_pool = s2_pool_all[~s2_pool_all[prez_col]].copy()

s2_pool["presence"] = 0
s3_pool["presence"] = 0
s4_pool["presence"] = 0

print("\n=== Pseudo-absence pool sanity ===")
print(f"RANDOM file rows: {len(pa_random)} | matched CellIDs: {len(s2_pool_all)} | presences inside file: {s2_includes_pres} | trueabs inside file: {s2_includes_trueabs}")
print(f"RANDOM used as negatives after dropping presences: {len(s2_pool)}")
print(f"CONTROLLED matched CellIDs (unlabelled only): {len(s3_pool)}")
print(f"LLM3 matched CellIDs (unlabelled only): {len(s4_pool)}")


# -----------------------
# 7) BUILD TRAINING DATASETS (TRUE, RANDOM, CONTROLLED, LLM3)
# -----------------------
if BALANCE_TO_N_POS:
    n_neg_true = int(FORCE_N_NEG_TRUE) if FORCE_N_NEG_TRUE is not None else n_pos
    n_neg_rand = int(FORCE_N_NEG_RANDOM) if FORCE_N_NEG_RANDOM is not None else n_pos
    n_neg_ctrl = int(FORCE_N_NEG_CTRL) if FORCE_N_NEG_CTRL is not None else n_pos
    n_neg_llm3 = int(FORCE_N_NEG_LLM3) if FORCE_N_NEG_LLM3 is not None else n_pos
else:
    n_neg_true = int(FORCE_N_NEG_TRUE) if FORCE_N_NEG_TRUE is not None else len(trueabs)
    n_neg_rand = int(FORCE_N_NEG_RANDOM) if FORCE_N_NEG_RANDOM is not None else len(s2_pool)
    n_neg_ctrl = int(FORCE_N_NEG_CTRL) if FORCE_N_NEG_CTRL is not None else len(s3_pool)
    n_neg_llm3 = int(FORCE_N_NEG_LLM3) if FORCE_N_NEG_LLM3 is not None else len(s4_pool)

S1_TRUE = pd.concat([pres, sample_neg(trueabs, n_neg_true, seed=SEED)], ignore_index=True)
S2_RAND = pd.concat([pres, sample_neg(s2_pool, n_neg_rand, seed=SEED + 1)], ignore_index=True)
S3_CTRL = pd.concat([pres, sample_neg(s3_pool, n_neg_ctrl, seed=SEED + 2)], ignore_index=True)
S4_LLM3 = pd.concat([pres, sample_neg(s4_pool, n_neg_llm3, seed=SEED + 3)], ignore_index=True)

def dropna_predictors(df: pd.DataFrame, name: str, predictors_: list[str]) -> pd.DataFrame:
    df2 = df.dropna(subset=predictors_).copy()
    if len(df2) != len(df):
        print(f"[WARN] {name}: dropped {len(df)-len(df2)} rows due to NA predictors")
    return df2

S1_TRUE = dropna_predictors(S1_TRUE, "S1_TRUE", predictors)
S2_RAND = dropna_predictors(S2_RAND, "S2_RANDOM", predictors)
S3_CTRL = dropna_predictors(S3_CTRL, "S3_CONTROLLED", predictors)
S4_LLM3 = dropna_predictors(S4_LLM3, "S4_LLM3", predictors)

print("\n=== Training dataset sizes (after NA drop) ===")
print("TRUE      :", len(S1_TRUE), " (pos:", int(S1_TRUE["presence"].sum()), ")")
print("RANDOM    :", len(S2_RAND), " (pos:", int(S2_RAND["presence"].sum()), ")")
print("CONTROLLED:", len(S3_CTRL), " (pos:", int(S3_CTRL["presence"].sum()), ")")
print("LLM3      :", len(S4_LLM3), " (pos:", int(S4_LLM3["presence"].sum()), ")")


# -----------------------
# 8) CV PERFORMANCE SUMMARY
# -----------------------
cv_rows = []
for name, df_train in [("TRUE", S1_TRUE), ("RANDOM", S2_RAND), ("CONTROLLED", S3_CTRL), ("LLM3", S4_LLM3)]:
    m = run_cv(df_train, predictors, n_splits=N_SPLITS)
    cv_rows.append({"scenario": name, **m})

cv_df = pd.DataFrame(cv_rows)
cv_path = os.path.join(OUT_DIR, "cv_summary.csv")
cv_df.to_csv(cv_path, index=False)
print("\nWrote:", cv_path)


# -----------------------
# 9) TRAIN FINAL MODELS + PREDICT ALL CELLS
# -----------------------
all_cells = cell.dropna(subset=predictors).copy()

preds = all_cells[["CellID"]].copy()
preds[COL_TRUE] = train_and_predict(S1_TRUE, all_cells, predictors)
preds[COL_RAND] = train_and_predict(S2_RAND, all_cells, predictors)
preds[COL_CTRL] = train_and_predict(S3_CTRL, all_cells, predictors)
preds[COL_LLM3] = train_and_predict(S4_LLM3, all_cells, predictors)

pred_path = os.path.join(OUT_DIR, "predictions_all_cells.csv")
preds.to_csv(pred_path, index=False)
print("\nWrote:", pred_path)

prob_cols = [COL_TRUE, COL_RAND, COL_CTRL, COL_LLM3]
print("SECTION 9 preds cols:", preds.columns.tolist())


# -----------------------
# 10) Prediction distribution diagnostics + histograms
# -----------------------
diag_rows = []
for c in prob_cols:
    x = pd.to_numeric(preds[c], errors="coerce").to_numpy()
    diag_rows.append({
        "prob_col": c,
        "n": int(np.isfinite(x).sum()),
        "mean": float(np.nanmean(x)),
        "sd": float(np.nanstd(x)),
        "p05": float(np.nanpercentile(x, 5)),
        "p25": float(np.nanpercentile(x, 25)),
        "p50": float(np.nanpercentile(x, 50)),
        "p75": float(np.nanpercentile(x, 75)),
        "p95": float(np.nanpercentile(x, 95)),
        "pct_ge_0.9": float(100.0 * np.nanmean(x >= 0.9)),
        "pct_ge_0.5": float(100.0 * np.nanmean(x >= 0.5)),
        "pct_le_0.1": float(100.0 * np.nanmean(x <= 0.1)),
    })

diag_df = pd.DataFrame(diag_rows).sort_values("prob_col")
diag_path = os.path.join(OUT_DIR, "pred_distribution_diagnostics.csv")
diag_df.to_csv(diag_path, index=False)
print("\nWrote:", diag_path)

for c in prob_cols:
    plt.figure(figsize=(6.5, 4.5))
    plt.hist(preds[c].to_numpy(), bins=30, alpha=0.9)
    plt.xlabel(c)
    plt.ylabel("Count")
    plt.title(f"Distribution of {c}")
    plt.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"hist_{c}.png"), dpi=160)
    plt.close()


# -----------------------
# 11) Similarity of predictions (Spearman + scatter)
# -----------------------
corr_rows = []
for i in range(len(prob_cols)):
    for j in range(i + 1, len(prob_cols)):
        a, b = prob_cols[i], prob_cols[j]
        r, p = spearmanr(preds[a].to_numpy(), preds[b].to_numpy())
        corr_rows.append({"a": a, "b": b, "spearman_r": float(r), "p_value": float(p)})

corr_df = pd.DataFrame(corr_rows)
corr_path = os.path.join(OUT_DIR, "pred_similarity_spearman.csv")
corr_df.to_csv(corr_path, index=False)
print("\nWrote:", corr_path)

for i in range(len(prob_cols)):
    for j in range(i + 1, len(prob_cols)):
        a, b = prob_cols[i], prob_cols[j]
        plt.figure(figsize=(6.5, 6))
        plt.scatter(preds[a], preds[b], s=10, alpha=0.35)
        plt.xlabel(a)
        plt.ylabel(b)
        plt.title(f"Prediction comparison: {a} vs {b}")
        plt.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"scatter_{a}_vs_{b}.png"), dpi=160)
        plt.close()


# -----------------------
# 12) Mahalanobis diagnostics (CellID-level) - negatives vs presence profile
# -----------------------
X_pres = pres.dropna(subset=predictors)[predictors].to_numpy(dtype=float)
mu = X_pres.mean(axis=0)

cov = np.cov(X_pres, rowvar=False)
cov = cov + RIDGE * np.eye(cov.shape[0])
inv_cov = np.linalg.inv(cov)

def pool_X(df_pool: pd.DataFrame) -> np.ndarray:
    return df_pool.dropna(subset=predictors)[predictors].to_numpy(dtype=float)

md_rows = [
    summarize_dist("TRUEABS", mahalanobis_dists(pool_X(trueabs), mu, inv_cov)),
    summarize_dist("RANDOM_PA", mahalanobis_dists(pool_X(s2_pool), mu, inv_cov)),
    summarize_dist("CONTROLLED_PA", mahalanobis_dists(pool_X(s3_pool), mu, inv_cov)),
    summarize_dist("LLM3_PA", mahalanobis_dists(pool_X(s4_pool), mu, inv_cov)),
    summarize_dist("UNLABELLED_POOL", mahalanobis_dists(pool_X(unlab_pool), mu, inv_cov)),
]
md_df = pd.DataFrame(md_rows)
md_path = os.path.join(OUT_DIR, "mahalanobis_distance_negatives_to_presence.csv")
md_df.to_csv(md_path, index=False)
print("\nWrote:", md_path)


# -----------------------
# 13) HSA sizes + overlaps (CellID-level)
# -----------------------
ids_true = set(preds.loc[preds[COL_TRUE] >= HSA_THR, "CellID"].astype(str).tolist())
ids_rand = set(preds.loc[preds[COL_RAND] >= HSA_THR, "CellID"].astype(str).tolist())
ids_ctrl = set(preds.loc[preds[COL_CTRL] >= HSA_THR, "CellID"].astype(str).tolist())
ids_llm3 = set(preds.loc[preds[COL_LLM3] >= HSA_THR, "CellID"].astype(str).tolist())

hsa_sizes_df = pd.DataFrame([
    {"scenario": "TRUE", "n_cellids": len(ids_true)},
    {"scenario": "RANDOM", "n_cellids": len(ids_rand)},
    {"scenario": "CONTROLLED", "n_cellids": len(ids_ctrl)},
    {"scenario": "LLM3", "n_cellids": len(ids_llm3)},
])
hsa_sizes_path = os.path.join(OUT_DIR, "hsa_sizes.csv")
hsa_sizes_df.to_csv(hsa_sizes_path, index=False)

def coverage_of_true(other: set, true_set: set) -> float:
    if not true_set:
        return float("nan")
    return 100.0 * len(other.intersection(true_set)) / len(true_set)

pairs = [
    ("TRUE vs RANDOM", ids_true, ids_rand),
    ("TRUE vs CONTROLLED", ids_true, ids_ctrl),
    ("TRUE vs LLM3", ids_true, ids_llm3),
    ("RANDOM vs CONTROLLED", ids_rand, ids_ctrl),
    ("RANDOM vs LLM3", ids_rand, ids_llm3),
    ("CONTROLLED vs LLM3", ids_ctrl, ids_llm3),
]
overlap = []
for label, A, B in pairs:
    row = {"pair": label, "overlap_pct_union": overlap_pct(A, B), "coverage_of_TRUE_pct": np.nan}
    if label.startswith("TRUE vs "):
        row["coverage_of_TRUE_pct"] = coverage_of_true(B, A)
    overlap.append(row)

hsa_overlap_df = pd.DataFrame(overlap)
hsa_overlap_path = os.path.join(OUT_DIR, "hsa_overlap_percent.csv")
hsa_overlap_df.to_csv(hsa_overlap_path, index=False)

print("\nWrote:", hsa_sizes_path)
print("Wrote:", hsa_overlap_path)


# -----------------------
# 14) Table2 predictor-space distances (FID-level) + FID-level overlaps/sizes
# -----------------------
net_fid2 = net_fid.copy()
net_fid2["CellID"] = normalize_cellid(net_fid2["CellID"])

pred2 = preds[["CellID", COL_TRUE, COL_RAND, COL_CTRL, COL_LLM3]].copy()
pred2["CellID"] = normalize_cellid(pred2["CellID"])

fid_df = net_fid2.merge(pred2, on="CellID", how="inner")
fid_df = fid_df.dropna(subset=predictors).copy()

def hsa_frame(frame: pd.DataFrame, prob_col: str, thr: float) -> pd.DataFrame:
    return frame[frame[prob_col] >= thr].copy()

def cov_regularized(X: np.ndarray, ridge: float) -> np.ndarray:
    C = np.cov(X, rowvar=False)
    return C + ridge * np.eye(C.shape[0])

def standardized_euclidean(a: np.ndarray, b: np.ndarray, ref_sd: np.ndarray) -> float:
    z = (a - b) / np.where(ref_sd == 0, 1.0, ref_sd)
    return float(np.sqrt(np.sum(z ** 2)))

def mahalanobis_between_centroids(a: np.ndarray, b: np.ndarray, cov_: np.ndarray) -> float:
    inv = np.linalg.inv(cov_)
    d = a - b
    return float(np.sqrt(d.T @ inv @ d))

def mean_abs_cohens_d(mu1: np.ndarray, mu2: np.ndarray, var1: np.ndarray, var2: np.ndarray) -> float:
    pooled = 0.5 * (var1 + var2)
    pooled = np.where(pooled <= 0, np.nan, pooled)
    d = np.abs(mu1 - mu2) / np.sqrt(pooled)
    return float(np.nanmean(d))

H_TRUE = hsa_frame(fid_df, COL_TRUE, HSA_THR)
H_RAND = hsa_frame(fid_df, COL_RAND, HSA_THR)
H_CTRL = hsa_frame(fid_df, COL_CTRL, HSA_THR)
H_LLM3 = hsa_frame(fid_df, COL_LLM3, HSA_THR)

def stats(frame: pd.DataFrame) -> dict:
    X = frame[predictors].to_numpy(dtype=float)
    return {"mu": X.mean(axis=0), "var": X.var(axis=0), "n": int(len(frame))}

st_true = stats(H_TRUE)
st_rand = stats(H_RAND)
st_ctrl = stats(H_CTRL)
st_llm3 = stats(H_LLM3)

ref_sd = np.sqrt(st_true["var"])
cov_true = cov_regularized(H_TRUE[predictors].to_numpy(dtype=float), RIDGE)

def row_vs_true(name_other: str, st_other: dict) -> dict:
    return {
        "Scenario": f"TRUE compared to {name_other}",
        "Euclidean_distance": standardized_euclidean(st_true["mu"], st_other["mu"], ref_sd),
        "Mahalanobis_distance": mahalanobis_between_centroids(st_true["mu"], st_other["mu"], cov_true),
        "Mean_abs_Cohens_d": mean_abs_cohens_d(st_true["mu"], st_other["mu"], st_true["var"], st_other["var"]),
        "n_FIDs_TRUE": st_true["n"],
        "n_FIDs_other": st_other["n"],
    }

table2_df = pd.DataFrame([
    row_vs_true("RANDOM", st_rand),
    row_vs_true("CONTROLLED", st_ctrl),
    row_vs_true("LLM3", st_llm3),
])
table2_path = os.path.join(OUT_DIR, "table2_predictor_space_distances.csv")
table2_df.to_csv(table2_path, index=False)
print("\nWrote:", table2_path)

fid_id_col = "FID" if "FID" in fid_df.columns else None

def ids_fid(frame: pd.DataFrame):
    if fid_id_col:
        return set(frame[fid_id_col].astype(int).tolist())
    return set(frame.index.tolist())

ids_true_f = ids_fid(H_TRUE)
ids_rand_f = ids_fid(H_RAND)
ids_ctrl_f = ids_fid(H_CTRL)
ids_llm3_f = ids_fid(H_LLM3)

overlap_fid_df = pd.DataFrame([
    {"pair": "TRUE vs RANDOM", "overlap_pct_union": overlap_pct(ids_true_f, ids_rand_f)},
    {"pair": "TRUE vs CONTROLLED", "overlap_pct_union": overlap_pct(ids_true_f, ids_ctrl_f)},
    {"pair": "TRUE vs LLM3", "overlap_pct_union": overlap_pct(ids_true_f, ids_llm3_f)},
    {"pair": "RANDOM vs CONTROLLED", "overlap_pct_union": overlap_pct(ids_rand_f, ids_ctrl_f)},
    {"pair": "RANDOM vs LLM3", "overlap_pct_union": overlap_pct(ids_rand_f, ids_llm3_f)},
    {"pair": "CONTROLLED vs LLM3", "overlap_pct_union": overlap_pct(ids_ctrl_f, ids_llm3_f)},
])
overlap_fid_path = os.path.join(OUT_DIR, "hsa_overlap_percent_FIDlevel.csv")
overlap_fid_df.to_csv(overlap_fid_path, index=False)
print("Wrote:", overlap_fid_path)

hsa_sizes_fid_df = pd.DataFrame([
    {"scenario": "TRUE", "n_fids": int(len(H_TRUE))},
    {"scenario": "RANDOM", "n_fids": int(len(H_RAND))},
    {"scenario": "CONTROLLED", "n_fids": int(len(H_CTRL))},
    {"scenario": "LLM3", "n_fids": int(len(H_LLM3))},
])
hsa_sizes_fid_path = os.path.join(OUT_DIR, "hsa_sizes_FIDlevel.csv")
hsa_sizes_fid_df.to_csv(hsa_sizes_fid_path, index=False)
print("Wrote:", hsa_sizes_fid_path)

print("\n=== DONE ===")
print("OUT_DIR:", OUT_DIR)


# =========================================================
# EXPORT FULL NETWORK WITH ONLY 4 BINARY PREDICTION COLUMNS (0/1)
#   pred_TRUE_01, pred_RANDOM_01, pred_CONTROLLED_01, pred_LLM3_01
# =========================================================
from pathlib import Path
import pandas as pd

def _norm_cellid(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    return s.str.replace(r"\.0$", "", regex=True)

# Start from full FID-level network
net_export = net_fid.copy()
net_export["CellID"] = _norm_cellid(net_export["CellID"])

# Merge CellID-level probabilities (only to compute 0/1; we will drop probs afterward)
pred_export = preds[["CellID", COL_TRUE, COL_RAND, COL_CTRL, COL_LLM3]].copy()
pred_export["CellID"] = _norm_cellid(pred_export["CellID"])

net_export = net_export.merge(pred_export, on="CellID", how="left", validate="many_to_one")

# Compute binary flags (0/1)
net_export["pred_TRUE_01"] = (pd.to_numeric(net_export[COL_TRUE], errors="coerce") >= HSA_THR).astype(int)
net_export["pred_RANDOM_01"] = (pd.to_numeric(net_export[COL_RAND], errors="coerce") >= HSA_THR).astype(int)
net_export["pred_CONTROLLED_01"] = (pd.to_numeric(net_export[COL_CTRL], errors="coerce") >= HSA_THR).astype(int)
net_export["pred_LLM3_01"] = (pd.to_numeric(net_export[COL_LLM3], errors="coerce") >= HSA_THR).astype(int)

# Drop probability columns so ONLY 0/1 columns remain added
net_export = net_export.drop(columns=[COL_TRUE, COL_RAND, COL_CTRL, COL_LLM3], errors="ignore")

# Save
out_xlsx = Path(OUT_DIR) / f"{SPECIES}_full_network_predicted01_only.xlsx"
out_csv  = Path(OUT_DIR) / f"{SPECIES}_full_network_predicted01_only.csv"

try:
    net_export.to_excel(out_xlsx, index=False)
    print(f"[OK] Wrote Excel: {out_xlsx}")
except Exception as e:
    print(f"[WARN] Excel export failed ({e}).")

net_export.to_csv(out_csv, index=False)
print(f"[OK] Wrote CSV: {out_csv}")

# Sanity counts
for c in ["pred_TRUE_01", "pred_RANDOM_01", "pred_CONTROLLED_01", "pred_LLM3_01"]:
    print(c, int(net_export[c].sum()), "/", len(net_export), "FIDs predicted presence")
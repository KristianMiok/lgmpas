#!/usr/bin/env python3
# =========================================================
# Evaluate three scenarios (S1 TRUE, S2 RANDOM, S3 LLM) with 5-fold CV
# Enforce minimum-distance spacing for S2 & S3 (e.g., 10 km)
# Metrics: Accuracy, F1, AUC (ROC), TSS (at 0.5 threshold)
# Outputs: summary CSV/table + per-species scatter plots
# ---------------------------------------------------------
# Expected paths:
#   Aggregated net: /Users/kristianmiok/Desktop/Lucian/LLM/Data/New_FXL/net_aggr_all_clean.csv
#   LLM IDs folder: /Users/kristianmiok/Desktop/Lucian/LLM/Data/New_FXL/IDs   (AUB.txt/rtf, AUT.txt/rtf, FXL.txt/rtf)
#   Results folder: /Users/kristianmiok/Desktop/Lucian/LLM/Data/New_FXL/llm_results/tree
# =========================================================

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix
)

# ----------------------- CONFIG -----------------------
BASE   = "/Users/kristianmiok/Desktop/Lucian/LLM/Data/New_FXL"
NET_CSV = os.path.join(BASE, "net_aggr_all_clean.csv")  # your cleaned aggregate

# Here we make the difference between the RAG which is IDs and without information IDs2
ID_DIR  = os.path.join(BASE, "IDs2")
OUT_DIR = os.path.join(BASE, "llm_results/new2")
os.makedirs(OUT_DIR, exist_ok=True)

# Species and candidate predictors (we’ll keep only those present in the CSV)
SPECIES = ["AUB", "AUT", "FXL"]
CANDIDATE_PRED_COLS = ["RWQ", "ALT", "FFP", "BIO1", "BIO5", "BIO9", "BIO12", "BIO16", "BIO17", "BIO18"]

# Random Forest settings
N_SPLITS = 5
RF_PARAMS = dict(n_estimators=1000, max_features="sqrt", random_state=42, n_jobs=-1)

# Minimum distance (km) spacing for S2 & S3 negatives
MIN_KM_SPACING = 10.0

# Apply biotic constraint for natives (exclude FXL presences from their negative pools)?
BIOTIC_CONSTRAINT_FOR_NATIVES = True  # as in the draft: apply to AUB & AUT only

# -------------------- HELPERS -------------------------
def read_cellids_any(sp_code: str, folder: str) -> list[str]:
    """Read CellIDs (e.g., C_123_456) from the first matching .rtf/.txt in folder."""
    pats = [os.path.join(folder, f"{sp_code}.rtf"),
            os.path.join(folder, f"{sp_code}.RTF"),
            os.path.join(folder, f"{sp_code}.txt"),
            os.path.join(folder, f"{sp_code}.TXT")]
    files = [p for pat in pats for p in glob.glob(pat)]
    if not files:
        print(f"[WARN] No ID file found for {sp_code} in {folder}")
        return []
    path = files[0]
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return []
    ids = re.findall(r"C_\d+_\d+", txt)
    # deduplicate while preserving order
    return list(dict.fromkeys(ids))

def get_pa_tables(net_df: pd.DataFrame, sp: str):
    """Return presence and true-absence tables for species sp."""
    pres_flag = f"has_{sp}_PREZ"
    abs_flag  = f"has_{sp}_TRUEABS"
    if pres_flag not in net_df.columns or abs_flag not in net_df.columns:
        raise ValueError(f"Missing {pres_flag} or {abs_flag} in {NET_CSV}")
    pres = net_df.loc[net_df[pres_flag].astype(bool)].copy()
    pres["presence"] = 1
    trueabs = net_df.loc[net_df[abs_flag].astype(bool)].copy()
    trueabs["presence"] = 0
    return pres, trueabs

# ---- geodesic spacing helpers ----
def haversine_km(lon1, lat1, lon2, lat2):
    """Vectorized Haversine distance in km; (lon1,lat1) vs arrays (lon2,lat2)."""
    R = 6371.0088
    lon1, lat1 = np.radians(lon1), np.radians(lat1)
    lon2, lat2 = np.radians(lon2), np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def thin_against_presences(cand_df: pd.DataFrame, pres_df: pd.DataFrame, min_km: float) -> pd.DataFrame:
    """Drop candidates closer than min_km to ANY presence point."""
    if cand_df.empty or pres_df.empty:
        return cand_df.copy()
    c_lon = cand_df["X_WGS84_DD"].to_numpy()
    c_lat = cand_df["Y_WGS84_DD"].to_numpy()
    p_lon = pres_df["X_WGS84_DD"].to_numpy()
    p_lat = pres_df["Y_WGS84_DD"].to_numpy()

    keep = []
    for i in range(len(cand_df)):
        d = haversine_km(c_lon[i], c_lat[i], p_lon, p_lat)
        keep.append(np.min(d) >= min_km)
    return cand_df.loc[keep].copy()

def greedy_self_thin(df: pd.DataFrame, min_km: float) -> pd.DataFrame:
    """Greedy thinning so retained rows are ≥ min_km from each other."""
    if df.empty:
        return df.copy()
    df = df.sort_values(["X_WGS84_DD", "Y_WGS84_DD"]).reset_index(drop=True)
    kept = []
    for i, r in df.iterrows():
        if not kept:
            kept.append(i)
            continue
        lon_i, lat_i = r["X_WGS84_DD"], r["Y_WGS84_DD"]
        too_close = False
        for j in kept:
            d = haversine_km(lon_i, lat_i, df.at[j, "X_WGS84_DD"], df.at[j, "Y_WGS84_DD"])
            if d < min_km:
                too_close = True
                break
        if not too_close:
            kept.append(i)
    return df.iloc[kept].copy()

# ---- dataset builders (S1, S2, S3) ----
def make_A1_TRUE(pres: pd.DataFrame, trueabs: pd.DataFrame, rng=42):
    """S1: presences vs known TRUE absences (balanced to n_pos)."""
    n_pos = len(pres)
    if n_pos == 0 or len(trueabs) == 0:
        return None
    abs_s = trueabs.sample(n=n_pos, replace=(len(trueabs) < n_pos), random_state=rng)
    return pd.concat([pres, abs_s], ignore_index=True)

def make_A2_RANDOM(pres: pd.DataFrame, net_df: pd.DataFrame, sp: str, rng=42, min_km=10.0):
    """S2: presences vs random unlabeled negatives (exclude target presences & true-absences) + spacing."""
    n_pos = len(pres)
    if n_pos == 0:
        return None
    prez_flag = f"has_{sp}_PREZ"
    abs_flag  = f"has_{sp}_TRUEABS"
    pool = net_df.loc[(~net_df[prez_flag].astype(bool)) & (~net_df[abs_flag].astype(bool))].copy()

    # Optional biotic constraint: for natives exclude FXL presences
    if BIOTIC_CONSTRAINT_FOR_NATIVES and sp in ("AUB", "AUT") and "has_FXL_PREZ" in pool.columns:
        pool = pool[~pool["has_FXL_PREZ"]]

    if pool.empty:
        return None

    # Spacing against presences, then among negatives
    pool = thin_against_presences(pool, pres, min_km=min_km)
    pool = greedy_self_thin(pool, min_km=min_km)
    if pool.empty:
        return None

    pool["presence"] = 0
    neg_s = pool.sample(n=n_pos, replace=(len(pool) < n_pos), random_state=rng)
    return pd.concat([pres, neg_s], ignore_index=True)

def make_A3_LLM(pres: pd.DataFrame, net_df: pd.DataFrame, llm_ids: list[str], rng=42, min_km=10.0, sp=None):
    """S3: presences vs LLM-selected pseudo-absences + spacing."""
    n_pos = len(pres)
    if n_pos == 0 or not llm_ids:
        return None
    cand = net_df[net_df["CellID"].isin(llm_ids)].copy()

    # Optional biotic constraint for natives (exclude FXL presences)
    if BIOTIC_CONSTRAINT_FOR_NATIVES and sp in ("AUB", "AUT") and "has_FXL_PREZ" in cand.columns:
        cand = cand[~cand["has_FXL_PREZ"]]

    if cand.empty:
        return None

    cand = thin_against_presences(cand, pres, min_km=min_km)
    cand = greedy_self_thin(cand, min_km=min_km)
    if cand.empty:
        return None

    cand["presence"] = 0
    abs_s = cand.sample(n=n_pos, replace=(len(cand) < n_pos), random_state=rng)
    return pd.concat([pres, abs_s], ignore_index=True)

# ---- metrics ----
def tss_from_labels(y_true, y_pred):
    """TSS = Sensitivity + Specificity - 1 at threshold 0.5 (here using predicted labels)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sens + spec - 1.0

def run_cv_metrics(df: pd.DataFrame, predictors: list[str], n_splits=5):
    """5-fold CV: return mean±sd for ACC, F1, AUC, TSS."""
    X = df[predictors].to_numpy()
    y = df["presence"].to_numpy().astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    accs, f1s, aucs, tsss = [], [], [], []
    for tr, te in skf.split(X, y):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X_tr, y_tr)
        y_prob = clf.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        accs.append(accuracy_score(y_te, y_pred))
        f1s.append(f1_score(y_te, y_pred))
        try:
            aucs.append(roc_auc_score(y_te, y_prob))
        except ValueError:
            pass
        tsss.append(tss_from_labels(y_te, y_pred))

    def mean_sd(a):
        a = np.array(a, dtype=float)
        return float(np.mean(a)), float(np.std(a))

    acc_m, acc_s = mean_sd(accs)
    f1_m,  f1_s  = mean_sd(f1s)
    tss_m, tss_s = mean_sd(tsss)
    if len(aucs) > 0:
        auc_m, auc_s = mean_sd(aucs)
    else:
        auc_m, auc_s = np.nan, np.nan

    return dict(acc_mean=acc_m, acc_sd=acc_s,
                f1_mean=f1_m,  f1_sd=f1_s,
                auc_mean=auc_m, auc_sd=auc_s,
                tss_mean=tss_m, tss_sd=tss_s)

def format_pm(mean, sd, decimals=3):
    return f"{mean:.{decimals}f} (±{sd:.{decimals}f})"

# ---- plotting ----
def plot_species(sp, pres, a1, a2, a3, out_dir):
    base = a1 if a1 is not None else (a2 if a2 is not None else (a3 if a3 is not None else pres))
    if base is None or base.empty:
        print(f"[{sp}] Nothing to plot.")
        return

    plt.figure(figsize=(8, 7))

    # presences (circles)
    pres_pts = base[base["presence"] == 1] if "presence" in base.columns else pres
    if pres_pts is not None and not pres_pts.empty:
        plt.scatter(pres_pts["X_WGS84_DD"], pres_pts["Y_WGS84_DD"],
                    s=20, marker="o", alpha=0.9, label="Presence")

    # true absences (triangles) from S1
    if a1 is not None:
        tna = a1[a1["presence"] == 0]
        if not tna.empty:
            plt.scatter(tna["X_WGS84_DD"], tna["Y_WGS84_DD"],
                        s=28, marker="^", alpha=0.85, label="True absence (S1)")

    # random negatives (squares) from S2
    if a2 is not None:
        rnd = a2[a2["presence"] == 0]
        if not rnd.empty:
            plt.scatter(rnd["X_WGS84_DD"], rnd["Y_WGS84_DD"],
                        s=32, marker="s", alpha=0.85, label="Random negative (S2, spaced)")

    # LLM pseudo-absences (crosses) from S3
    if a3 is not None:
        pna = a3[a3["presence"] == 0]
        if not pna.empty:
            plt.scatter(pna["X_WGS84_DD"], pna["Y_WGS84_DD"],
                        s=36, marker="x", alpha=0.95, label="Pseudo-absence (S3 LLM, spaced)")

    plt.xlabel("Longitude (X_WGS84_DD)")
    plt.ylabel("Latitude (Y_WGS84_DD)")
    plt.title(f"{sp}: Presence vs. Absence Types (with {MIN_KM_SPACING:.0f} km spacing for S2/S3)")
    plt.legend(loc="best", frameon=True)
    plt.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"{sp}_S1_S2_S3_scatter.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[{sp}] Saved scatter: {fig_path}")

# -------------------- MAIN ---------------------------
def main():
    # Load aggregated network
    net = pd.read_csv(NET_CSV)

    # Choose predictors that actually exist
    pred_cols = [c for c in CANDIDATE_PRED_COLS if c in net.columns]
    if not pred_cols:
        raise ValueError(f"No predictor columns found in {NET_CSV} among {CANDIDATE_PRED_COLS}")
    print("Using predictors:", pred_cols)

    # Read LLM IDs
    llm_ids = {sp: read_cellids_any(sp, ID_DIR) for sp in SPECIES}
    print("LLM IDs:", {k: len(v) for k, v in llm_ids.items()})

    rows = []       # for CSV summary
    printable = []  # for console table

    # Scenario label mapping to your table (S1, S2, S3)
    scenario_order = [("S1", "TRUE"), ("S2", "RANDOM"), ("S3", "LLM")]

    for sp in SPECIES:
        pres, trueabs = get_pa_tables(net, sp)
        if pres.empty:
            print(f"[{sp}] No presences; skipping.")
            continue

        s1 = make_A1_TRUE(pres, trueabs, rng=42)
        s2 = make_A2_RANDOM(pres, net, sp, rng=42, min_km=MIN_KM_SPACING)
        s3 = make_A3_LLM(pres, net, llm_ids.get(sp, []), rng=42, min_km=MIN_KM_SPACING, sp=sp)

        datasets = {"S1": s1, "S2": s2, "S3": s3}

        # Save the negatives actually used (audit)
        for key, df in datasets.items():
            if df is not None and not df.empty:
                negs = df[df["presence"] == 0]
                negs.to_csv(os.path.join(OUT_DIR, f"{sp}_{key}_negatives.csv"), index=False)

        # compute metrics
        for S, name in [("S1","TRUE"), ("S2","RANDOM"), ("S3","LLM")]:
            df = datasets[S]
            if df is None or df.empty:
                print(f"[{sp}] {S} ({name}) has no data; skipping.")
                continue
            metrics = run_cv_metrics(df, pred_cols, n_splits=N_SPLITS)
            rows.append({
                "species": sp,
                "scenario": S,
                "name": name,
                "accuracy": format_pm(metrics["acc_mean"], metrics["acc_sd"]),
                "f1":       format_pm(metrics["f1_mean"],  metrics["f1_sd"]),
                "auc":      format_pm(metrics["auc_mean"], metrics["auc_sd"]) if not np.isnan(metrics["auc_mean"]) else "NA",
                "tss":      format_pm(metrics["tss_mean"], metrics["tss_sd"]),
                # raw numbers too
                "acc_mean": metrics["acc_mean"], "acc_sd": metrics["acc_sd"],
                "f1_mean":  metrics["f1_mean"],  "f1_sd":  metrics["f1_sd"],
                "auc_mean": metrics["auc_mean"], "auc_sd": metrics["auc_sd"],
                "tss_mean": metrics["tss_mean"], "tss_sd": metrics["tss_sd"],
                "n_pos": int((df["presence"]==1).sum()),
                "n_neg": int((df["presence"]==0).sum())
            })
            printable.append([sp, S,
                              format_pm(metrics["acc_mean"], metrics["acc_sd"]),
                              format_pm(metrics["f1_mean"], metrics["f1_sd"]),
                              format_pm(metrics["auc_mean"], metrics["auc_sd"]) if not np.isnan(metrics["auc_mean"]) else "NA",
                              format_pm(metrics["tss_mean"], metrics["tss_sd"])])

        # plot (now spaced)
        plot_species(sp, pres, s1, s2, s3, OUT_DIR)

    # Save full summary CSV
    out_csv = os.path.join(OUT_DIR, "summary_rf_S1_S2_S3_spaced.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nSaved summary to: {out_csv}")

    # Pretty print & save compact table for manuscript
    if printable:
        df_print = pd.DataFrame(printable, columns=["Species", "S", "Accuracy", "F1-score", "AUC", "TSS"])
        print("\n=== Manuscript-ready summary ===")
        print(df_print.to_string(index=False))
        compact_csv = os.path.join(OUT_DIR, "summary_compact_table_spaced.csv")
        df_print.to_csv(compact_csv, index=False)
        print(f"\nSaved compact table to: {compact_csv}")
    else:
        print("No printable results (check inputs).")

if __name__ == "__main__":
    main()


    # --- Count pseudo-absences per species & scenario ---
    counts = []
    for sp in SPECIES:
        for sc, name in [("S1","TRUE"), ("S2","RANDOM"), ("S3","LLM")]:
            neg_path = os.path.join(OUT_DIR, f"{sp}_{sc}_negatives.csv")
            if os.path.exists(neg_path):
                n_neg = sum(1 for _ in open(neg_path)) - 1  # subtract header
            else:
                n_neg = 0
            counts.append({"species": sp, "scenario": sc, "name": name, "n_neg": n_neg})

    counts_df = pd.DataFrame(counts)
    print("\n=== Number of negatives (pseudo-absences) per species & scenario ===")
    print(counts_df.to_string(index=False))

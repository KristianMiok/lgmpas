# make_prompts.py
# Build eco-narrative prompt batches for LLM selection of pseudo-absences (LOOSE only)

from pathlib import Path
import pandas as pd
import numpy as np
from textwrap import dedent
import math

# ============ CONFIG ============
ROOT_DIR   = Path("/Users/kristianmiok/Desktop/Lucian/LLM/Data/New_FXL")
IN_FILE    = ROOT_DIR / "net_aggr_all.csv"                 # produced by your R aggregation script
PROMPT_DIR = ROOT_DIR / "Prompts"                          # output: .txt prompts you paste into ChatGPT
PROMPT_DIR.mkdir(parents=True, exist_ok=True)

# Species codes in your data
SPECIES = ["AUB", "AUT", "FXL"]

# Predictors to show in narratives (make sure these exist in net_aggr_all.csv)
PRED_COLS = ["RWQ", "ALT", "FFP", "BIO1"]

# Batch size for each prompt file (adjust if ChatGPT context is tight)
BATCH_SIZE = 200

# Target total pseudo-absences per species (upper bound). We’ll ask per-batch for ~this/num_batches.
MAX_TOTAL_PSEUDO = 100
# =================================

# Species knowledge snippets (very brief—feel free to edit text)
SPECIES_INFO = {
    "AUB": {
        "scientific": "Austropotamobius bihariensis",
        "common": "Idle crayfish",
        "ecology": (
            "Endemic to NW Romania (Apuseni Mountains). Prefers montane to sub-montane headwaters with "
            "cool to cold water, high oxygen, stable flow, and clear substrates. Sensitive to pollution, "
            "habitat alteration and crayfish plague."
        )
    },
    "AUT": {
        "scientific": "Austropotamobius torrentium",
        "common": "Stone crayfish",
        "ecology": (
            "Native crayfish of Central/Eastern Europe. Occupies small streams and foothill rivers with "
            "clear, well-oxygenated water, coarse substrates, and moderate to cool climates. Avoids "
            "warm, eutrophic, or degraded waters."
        )
    },
    "FXL": {
        "scientific": "Faxonius limosus",
        "common": "Spiny-cheek crayfish (invasive)",
        "ecology": (
            "Invasive North American crayfish. Tolerant generalist of lowland rivers, lakes and canals, "
            "including warmer, eutrophic and sometimes polluted habitats; often linked to crayfish plague spread."
        )
    }
}


def species_profile_text(df: pd.DataFrame, sp: str) -> str:
    """Median + range (min, max) for chosen predictors among presence cells of the target species."""
    prez_flag = f"has_{sp}_PREZ"
    pres = df[df[prez_flag].astype(bool)]
    if pres.empty:
        return "No known presences in the network summary; rely on general species ecology."
    parts = []
    for v in PRED_COLS:
        if v not in pres.columns:
            continue
        med = pres[v].median()
        vmin = pres[v].min()
        vmax = pres[v].max()
        parts.append(f"{v}: median={med:.3f}, range=[{vmin:.3f}, {vmax:.3f}]")
    return "; ".join(parts) if parts else "Presence predictors unavailable; rely on general species ecology."


def candidate_pool_loose(df: pd.DataFrame, sp: str) -> pd.DataFrame:
    """
    LOOSE: exclude only presences of the focal species.
    Other species' presences may remain in the pool.
    """
    mask = ~df[f"has_{sp}_PREZ"].astype(bool)
    return df.loc[mask].copy()


# ----- Narrative helpers -----
def q_rwq(x):
    if pd.isna(x): return "unknown water quality"
    if x < 0.2:    return "poor water quality"
    if x < 0.5:    return "moderate water quality"
    return "good to very good water quality"

def q_alt(x):
    if pd.isna(x): return "unknown elevation/relief"
    if x < 200:    return "lowland plain context"
    if x < 800:    return "foothill to sub-montane context"
    return "montane to high-elevation headwater context"

def q_bio1(x):
    if pd.isna(x): return "unknown mean annual temperature"
    if x < 5:      return "cold mean annual temperature"
    if x < 8:      return "cool mean annual temperature"
    if x < 12:     return "temperate mean annual temperature"
    return "warm mean annual temperature"

def q_ffp(x):
    if pd.isna(x): return "unknown flow period"
    if x < 60:     return "very short flow period"
    if x < 120:    return "short to moderate flow period"
    if x < 240:    return "moderate to long flow period"
    return "long flow period"


def make_narrative_row(row: pd.Series) -> str:
    # Raw values (rounded for readability)
    lat = float(row["Y_WGS84_DD"])
    lon = float(row["X_WGS84_DD"])
    rwq = row.get("RWQ", np.nan)
    alt = row.get("ALT", np.nan)
    ffp = row.get("FFP", np.nan)
    bio1 = row.get("BIO1", np.nan)
    ths = "unknown substrate" if pd.isna(row.get("THS")) else str(row.get("THS"))
    tys = "unknown stream type" if pd.isna(row.get("TYS")) else str(row.get("TYS"))

    return (
        f"CellID {row['CellID']} — latitude {lat:.5f}, longitude {lon:.5f}. "
        f"Hydro-ecological context: {q_alt(alt)} (ALT≈{np.round(alt, 3)} m), "
        f"{q_rwq(rwq)} (RWQ≈{np.round(rwq, 3)}), {q_ffp(ffp)} (FFP≈{np.round(ffp, 3)} days/yr), "
        f"and {q_bio1(bio1)} (BIO1≈{np.round(bio1, 3)} °C). "
        f"Local descriptors: THS='{ths}', TYS='{tys}'. Treat as unlabeled; use ecological reasoning to judge suitability."
    )


def chunk_df(df: pd.DataFrame, size: int) -> list[pd.DataFrame]:
    if df.empty:
        return []
    idx = np.arange(len(df))
    bins = np.floor(idx / size).astype(int)
    return [df.iloc[bins == b].copy() for b in np.unique(bins)]


def rules_block(per_batch_k: int, sp: str) -> str:
    return dedent(f"""\
        - Selection setting: LOOSE (exclude only known presences of {sp}).
        - All candidates are river-network cells; do not assume off-network habitats.
        - Prefer multi-variable profiles atypical for the PRESENCE profile of {sp} (outside or near edges of the ranges).
        - Avoid cells that cluster around presence medians across several predictors simultaneously.
        - If two cells are similar, choose the less suitable one considering elevation, temperature, water quality, flow period and local descriptors (THS/TYS).
        - Return at most N={per_batch_k} CellIDs for THIS BATCH. If fewer clearly unsuitable cells exist, return fewer.
        - Output ONLY a CSV with one header column named CellID and one CellID per line. No extra text.
    """)


def prompt_text(sp: str, profile_text: str, rules_text: str, batch_df: pd.DataFrame) -> str:
    info = SPECIES_INFO[sp]
    lines = "\n".join([" - " + make_narrative_row(r) for _, r in batch_df.iterrows()])
    return dedent(f"""\
        Task: From the following river-network cell narratives, select pseudo-absence CellIDs for species {sp}
        using the LOOSE pool (cells without known presences of {sp}).

        Target species:
        - Scientific name: {info['scientific']}
        - Common name: {info['common']}
        - Ecology: {info['ecology']}

        Presence profile (summary from known presences of {sp} across the study network):
        {profile_text}

        Decision rules:
        {rules_text}

        Candidate narratives:
        {lines}

        Output format (CSV, no commentary):
        CellID
        C_xxx_yyy
        C_aaa_bbb
        ...
    """)


def target_totals_for_species(df: pd.DataFrame, sp: str) -> tuple[int, int]:
    """
    Returns:
      presence_count: number of presence cells for the species
      total_target: min(MAX_TOTAL_PSEUDO, presence_count)
    """
    prez_flag = f"has_{sp}_PREZ"
    presence_count = int(df[prez_flag].sum())
    total_target = min(MAX_TOTAL_PSEUDO, presence_count if presence_count > 0 else MAX_TOTAL_PSEUDO)
    return presence_count, total_target


def build_prompts_for_species(df: pd.DataFrame, sp: str):
    prof = species_profile_text(df, sp)
    presence_count, total_target = target_totals_for_species(df, sp)

    pool = candidate_pool_loose(df, sp)
    need_cols = ["CellID", "Y_WGS84_DD", "X_WGS84_DD", "THS", "TYS"] + [c for c in PRED_COLS if c in df.columns]
    pool = pool.loc[:, [c for c in need_cols if c in pool.columns]].reset_index(drop=True)

    batches = chunk_df(pool, BATCH_SIZE)
    if not batches:
        print(f"No LOOSE candidates for {sp}.")
        return

    # Ask the LLM for fewer IDs per batch so totals ≈ total_target across all batches
    per_batch_k = max(1, math.ceil(total_target / len(batches)))

    for i, bt in enumerate(batches, start=1):
        txt = prompt_text(
            sp,
            profile_text=prof,
            rules_text=rules_block(per_batch_k, sp),
            batch_df=bt
        )
        out_file = PROMPT_DIR / f"{sp}_loose_batch{i:03d}.txt"
        out_file.write_text(txt, encoding="utf-8")

    print(f"Wrote {len(batches)} LOOSE prompts for {sp}. Requested ~{total_target} total IDs (~{per_batch_k}/batch).")


def main():
    df = pd.read_csv(IN_FILE)

    # basic sanity
    need_cols = ["CellID", "Y_WGS84_DD", "X_WGS84_DD"] + PRED_COLS + \
                ["has_AUB_PREZ", "has_AUT_PREZ", "has_FXL_PREZ"]
    for c in need_cols:
        if c not in df.columns:
            # only warn for optional predictors; error for core columns
            if c in ["CellID", "Y_WGS84_DD", "X_WGS84_DD", "has_AUB_PREZ", "has_AUT_PREZ", "has_FXL_PREZ"]:
                raise ValueError(f"Missing required column: {c}")
            else:
                print(f"Warning: optional predictor missing: {c}")

    for sp in SPECIES:
        build_prompts_for_species(df, sp)

    print("\nPrompts saved in:", PROMPT_DIR)
    print("Open the first file (e.g., AUB_loose_batch001.txt), paste into ChatGPT, and request a CSV with header 'CellID'.")


if __name__ == "__main__":
    main()
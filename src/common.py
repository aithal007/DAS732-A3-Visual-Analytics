"""
common.py - shared configuration, data loading and the FEEDBACK-LOOP data transformation.

Every team member's module imports from here so that the three parts of the workflow use
ONE definition of: the country panel, the region (continent) mapping used in A1, the
leakage-safe column lists, the artefact-cleaning rules and the engineered features.

Workflow position (see Fig. V0, the unrolled diamond workflow):
    Iteration 1  : load_panel()                    -> raw A1 data, no imputation, no cleaning
    Feedback loop: detect_artifacts() + transform_for_iteration2()
                   (inferences K1 from iteration 1 turned into a data transformation)
    Iteration 2  : the transformed panel feeds trajectory clustering, the engineered
                   supervised model and the forecast.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
GEO = ROOT / "data" / "geo" / "countries.geojson"
IMG = ROOT / "images"
OUT = ROOT / "outputs"
for _p in (PROC, IMG, OUT):
    _p.mkdir(parents=True, exist_ok=True)

MASTER_CSV = RAW / "master_dataset.csv"
SEED = 42

# --------------------------------------------------------------------------------------
# Columns
# --------------------------------------------------------------------------------------
TARGET = "internet_users_share_pct"
# exact duplicates / deterministic transforms found in the data audit (max |diff| = 0)
DUPLICATES = {
    "internet_users_share_pct_ict": TARGET,
    "broadband_penetration_per100": "broadband_subs_per100",
}
# counts: scale with population, and internet_users_count IS the target times population
COUNT_COLS = [
    "internet_users_count", "broadband_subs_count", "broadband_10mbps_plus_count",
    "broadband_2to10mbps_count", "broadband_256kbto2mbps_count",
]
RAW_INDICATORS = [
    "internet_users_share_pct", "mobile_subs_per100", "mobile_phone_ownership_share_pct",
    "telephone_lines_per100", "broadband_subs_per100",
]
SHORT = {
    "internet_users_share_pct": "Internet users %",
    "mobile_subs_per100": "Mobile subs /100",
    "mobile_phone_ownership_share_pct": "Phone owners %",
    "telephone_lines_per100": "Landlines /100",
    "broadband_subs_per100": "Fixed broadband /100",
}

# --------------------------------------------------------------------------------------
# Harmonised thresholds (one value for the whole team; justified in V3 of the report)
# --------------------------------------------------------------------------------------
MULTI_SIM_RATIO = 1.5      # SIMs per phone owner: >= 1 extra SIM for every 2 owners
ARTIFACT_LEVEL_SHIFT = 2.5  # value > 2.5x the country's median of the previous 5 years
ARTIFACT_SPIKE = 1.8        # isolated one-year jump/drop by >= 1.8x that reverts next year
ARTIFACT_REVERT = 1.25      # "reverts" = next year within 1.25x of the year before the spike

# --------------------------------------------------------------------------------------
# Regions: the SAME continent lookup used throughout A1 (data/raw/region_lookup.csv)
# --------------------------------------------------------------------------------------
_REGION_LOOKUP = pd.read_csv(RAW / "region_lookup.csv").set_index("code")["region"].to_dict()
REGION_ORDER = ["Africa", "Asia", "Europe", "North America", "South America", "Oceania"]
REGION_COLORS = {  # colour-blind-safe (Okabe-Ito), same order everywhere
    "Africa": "#E69F00", "Asia": "#56B4E9", "Europe": "#009E73",
    "North America": "#CC79A7", "South America": "#D55E00", "Oceania": "#0072B2",
    "Other": "#999999",
}


def region_of(codes: pd.Series) -> pd.Series:
    return codes.map(_REGION_LOOKUP).fillna("Other")


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def load_panel(verbose: bool = True) -> pd.DataFrame:
    """Country-level panel (aggregates removed), duplicates and counts dropped, A1 regions."""
    df = pd.read_csv(MASTER_CSV)
    # duplicate audit is asserted, not assumed
    for dup, orig in DUPLICATES.items():
        diff = (df[dup] - df[orig]).abs().max()
        assert diff == 0 or np.isnan(diff), f"{dup} is no longer an exact copy of {orig}"
    df = df[(~df["is_aggregate"].astype(bool)) & df["code"].notna()].copy()
    df = df[(df["code"].str.len() == 3) | df["code"].eq("OWID_KOS")]
    df = df.drop(columns=[c for c in list(DUPLICATES) + COUNT_COLS + ["is_aggregate"] if c in df])
    df["region"] = region_of(df["code"])
    df = df.sort_values(["code", "year"]).reset_index(drop=True)
    if verbose:
        print(f"[load_panel] {len(df)} country-years, {df.code.nunique()} countries, "
              f"{df.year.min()}-{df.year.max()}")
    return df


# --------------------------------------------------------------------------------------
# FEEDBACK LOOP, step 1: artefact detection (inference K1.1 from the anomaly detector)
# --------------------------------------------------------------------------------------
def detect_artifacts(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per implausible cell.

    Rule A (level shift, mobile only): a mature mobile market cannot multiply its
      subscriptions in a year; value > ARTIFACT_LEVEL_SHIFT x median(previous 5 years)
      and that median >= 90 (market already saturated, so this is not a boom year).  This is what Iteration-1 anomaly detection surfaced for Fiji
      (111 -> 361 -> 574 subs/100 in 2023-24 while ownership stayed at ~82%).
    Rule B (spike-and-revert, every indicator): x_t / x_{t-1} >= 1.8 (or <= 1/1.8) and
      x_{t+1} is back within 1.25x of x_{t-1}: a one-year recording glitch.
    """
    rows = []
    g = df.groupby("code", sort=False)
    # Rule A
    med5 = g["mobile_subs_per100"].transform(lambda s: s.shift(1).rolling(5, min_periods=3).median())
    mA = (med5 >= 90) & (df["mobile_subs_per100"] > ARTIFACT_LEVEL_SHIFT * med5)
    for i in df.index[mA]:
        rows.append((df.at[i, "code"], df.at[i, "entity"], int(df.at[i, "year"]), "mobile_subs_per100",
                     df.at[i, "mobile_subs_per100"], med5[i], "A: level shift vs 5-yr median"))
    # Rule B
    for col, floor in [("mobile_subs_per100", 20), ("broadband_subs_per100", 2),
                       ("telephone_lines_per100", 2), ("internet_users_share_pct", 5)]:
        prev = g[col].shift(1)
        nxt = g[col].shift(-1)
        r = df[col] / prev
        back = nxt / prev
        mB = ((prev >= floor) & ((r >= ARTIFACT_SPIKE) | (r <= 1 / ARTIFACT_SPIKE))
              & (back <= ARTIFACT_REVERT) & (back >= 1 / ARTIFACT_REVERT))
        for i in df.index[mB]:
            rows.append((df.at[i, "code"], df.at[i, "entity"], int(df.at[i, "year"]), col,
                         df.at[i, col], prev[i], "B: one-year spike that reverts"))
    out = pd.DataFrame(rows, columns=["code", "entity", "year", "variable", "value",
                                      "reference", "rule"])
    return out.drop_duplicates(["code", "year", "variable"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# FEEDBACK LOOP, step 2: the data transformation for Iteration 2
# --------------------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Domain features suggested by Iteration-1 inferences.  Predictor-only, past-only:
    no feature uses the target, and no feature at year t uses data after year t."""
    d = df.sort_values(["code", "year"]).copy()
    g = d.groupby("code", sort=False)

    # K1.2  multi-SIM inflation: subscriptions overstate people (A1 Set 2, V3)
    own = d["mobile_phone_ownership_share_pct"]
    d["sim_per_owner"] = np.where(own > 0, d["mobile_subs_per100"] / own.clip(lower=1), np.nan)
    d["multi_sim_flag"] = np.where(d["sim_per_owner"].notna(),
                                   (d["sim_per_owner"] >= MULTI_SIM_RATIO).astype(float), np.nan)
    d["mobile_capped"] = d["mobile_subs_per100"].clip(upper=100)  # a person needs one line

    # K1.3  leapfrogging: same landline level today, opposite histories (A1 Set 1)
    d["landline_peak"] = g["telephone_lines_per100"].cummax()            # expanding max, past only
    d["landline_peak"] = d.groupby("code")["landline_peak"].ffill()
    d["landline_decline"] = (d["landline_peak"] - d["telephone_lines_per100"]).clip(lower=0)
    d["leapfrog_index"] = d["mobile_capped"] / (d["landline_peak"] + 1.0)
    mob_gt = (d["mobile_subs_per100"] > d["telephone_lines_per100"]).astype(float)
    mob_gt[d["mobile_subs_per100"].isna() | d["telephone_lines_per100"].isna()] = np.nan
    first_cross = (d.assign(_gt=mob_gt).loc[lambda x: x["_gt"] == 1]
                   .groupby("code")["year"].min())
    fc = d["code"].map(first_cross)
    d["years_since_mobile_gt_landline"] = np.where(fc.notna() & (d["year"] >= fc), d["year"] - fc, 0.0)

    # K1.4  broadband floor effect: non-linear, near-zero for a third of country-years (V2)
    bb = d["broadband_subs_per100"].clip(lower=0)
    d["log1p_broadband"] = np.log1p(bb)
    d["broadband_zero"] = np.where(bb.notna(), (bb < 1.0).astype(float), np.nan)

    # K1.5  history matters more than the snapshot (trajectory clusters, V8): momentum
    for col, short in [("broadband_subs_per100", "broadband"), ("mobile_capped", "mobile")]:
        gc = d.groupby("code", sort=False)[col]
        d[f"{short}_yoy"] = d[col] - gc.shift(1)
        d[f"{short}_3yr"] = d[col] - gc.shift(3)
    return d


ENGINEERED = [
    "sim_per_owner", "multi_sim_flag", "mobile_capped", "landline_peak", "landline_decline",
    "leapfrog_index", "years_since_mobile_gt_landline", "log1p_broadband", "broadband_zero",
    "broadband_yoy", "broadband_3yr", "mobile_yoy", "mobile_3yr",
]


def transform_for_iteration2(df: pd.DataFrame, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the full feedback-loop transformation: clean artefacts -> engineer features."""
    arts = detect_artifacts(df)
    clean = df.copy()
    for _, r in arts.iterrows():
        clean.loc[(clean.code == r.code) & (clean.year == r.year), r.variable] = np.nan
    clean = engineer_features(clean)
    if verbose:
        print(f"[feedback] {len(arts)} artefact cells set to NaN "
              f"({arts.groupby('variable').size().to_dict()}); "
              f"{len(ENGINEERED)} engineered features added")
    return clean, arts


# --------------------------------------------------------------------------------------
# Plot style (shared so all figures read as one report)
# --------------------------------------------------------------------------------------
INK, MUTED, GRID = "#1F2328", "#6E7781", "#D0D7DE"
ACCENT, ACCENT2, BAD = "#2A78D6", "#E69F00", "#C0392B"


def set_style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 220, "font.size": 10, "axes.titlesize": 11.5,
        "axes.titleweight": "bold", "axes.labelsize": 10, "axes.edgecolor": "#57606A",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "grid.alpha": 0.7, "legend.frameon": False,
        "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "font.family": "DejaVu Sans",
    })


def savefig(fig, name: str):
    path = IMG / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [img] {path.name}")
    return path


set_style()

if __name__ == "__main__":
    p = load_panel()
    c, a = transform_for_iteration2(p)
    print(a.to_string())

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DAS732 Assignment 3 - Sindhoor Ganapathi Hegde
Unsupervised Mining (Iteration 1), Feedback Loop, Trajectory Clustering (Iteration 2)

Usage:  python src/p2_sindhoor_unsupervised.py   (or via src/run_all.py)

Outputs
    outputs/*.csv           tables (clusters, validation, anomalies, trajectories, feedback_features.csv)
    outputs/diagnostics/    quick-look plots; report figures V4, V5, V6, V8 come from p2_sindhoor_figures.py

Design rules
    * fixed seeds (SEED)  * no imputation for Iteration 1 (complete cases only)
    * leakage columns are never features (asserted)  * optional libs have offline fallbacks
"""
from __future__ import annotations

import glob
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist, squareform
from scipy.stats import rankdata
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from common import IMG as _IMG, MASTER_CSV, MULTI_SIM_RATIO as _MSR, OUT as _OUT, region_of
from common import transform_for_iteration2

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# 0. CONFIG
# =============================================================================
SEED = 42
SNAPSHOT_YEAR = 2024            # 2025 is sparse -> 2024 default cross-section
IMG_DIR, OUT_DIR = str(_OUT / "diagnostics"), str(_OUT)   # quick-look plots; report figures: p2_sindhoor_figures.py
DPI = 300

CORE = [
    "internet_users_share_pct",
    "mobile_subs_per100",
    "mobile_phone_ownership_share_pct",
    "telephone_lines_per100",
    "broadband_subs_per100",
]
DUPLICATE_COLS = ["internet_users_share_pct_ict", "broadband_penetration_per100"]
FORBIDDEN = [
    "internet_users_count", "broadband_subs_count", "broadband_10mbps_plus_count",
    "broadband_2to10mbps_count", "broadband_256kbto2mbps_count",
]
SHORT = {
    "internet_users_share_pct": "Internet %",
    "mobile_subs_per100": "Mobile subs/100",
    "mobile_phone_ownership_share_pct": "Phone owners %",
    "telephone_lines_per100": "Landlines/100",
    "broadband_subs_per100": "Broadband/100",
}

K_RANGE = range(2, 9)           # k scanned in validation panel
K_SELECT = range(3, 7)          # k eligible for final choice (avoid trivial k=2)
N_BOOT = 200                    # bootstrap resamples for ARI stability
TOP_N_ANOM = 15
MULTI_SIM_RATIO = _MSR          # harmonised team threshold (1.5 SIMs per owner), see V3
MULTI_SIM_PER100 = 150          # raw subs/100 used only when ownership is unobserved

# Iteration 2
TRAJ_Y0, TRAJ_Y1 = 2000, 2025
MIN_OBS = 18                    # of 26 years must be observed
EDGE_LIMIT = 3                  # max years to carry first/last observation outward
DTW_BAND = 5                    # Sakoe-Chiba radius (years)

# Paste the real A1 figure URLs/paths here; they are printed in the V6 block.
A1_LINKS = {  # appendix = A1 report
    "A1 Fig. 5":  "same landline level today, opposite histories -> landline_peak feature",
    "A1 Fig. 10": "multi-SIM inflation by region -> multi-SIM anomaly tag (V6), V3",
    "A1 Fig. 17": "every country's two-decade climb -> trajectory clusters (V8)",
    "A1 Fig. 18": "owning a phone is not being online -> sim_per_owner",
    "A1 Fig. 23": "two infrastructures, four kinds of country -> iteration-1 clusters (V5)",
    "A1 Fig. 26": "correlation structure of the five measures -> PCA (V4)",
}

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": DPI, "axes.titleweight": "bold"})


# =============================================================================
# 1. UTILITIES
# =============================================================================
def ensure_dirs():
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)


def save_fig(fig, name):
    path = os.path.join(IMG_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [saved] {path}")


def save_csv(df, name, index=True):
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path, index=index)
    print(f"  [saved] {path}  ({len(df)} rows)")


def show(title, df, index=True):
    print(f"\n=== {title} ===")
    print(df.to_string(index=index))


def assert_no_leakage(features):
    bad = set(features) & (set(FORBIDDEN) | set(DUPLICATE_COLS))
    assert not bad, f"Leakage/duplicate columns used as features: {bad}"


def relabel_by(labels, values):
    """Relabel clusters 0..k-1 in ascending order of mean(values) (0 = least connected)."""
    s = pd.Series(values).groupby(np.asarray(labels)).mean().sort_values()
    mapping = {old: new for new, old in enumerate(s.index)}
    return np.array([mapping[l] for l in labels])


def place_labels(ax, xs, ys, texts, fontsize=7):
    """Non-overlapping labels: adjustText if installed, else greedy offset search."""
    if len(texts) == 0:
        return
    try:
        from adjustText import adjust_text
        ts = [ax.text(x, y, t, fontsize=fontsize) for x, y, t in zip(xs, ys, texts)]
        adjust_text(ts, ax=ax, arrowprops=dict(arrowstyle="-", color="grey", lw=0.4))
        return
    except Exception:
        pass
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    offsets = [(6, 6), (6, -12), (-6, 6), (-6, -12), (12, 0), (-12, 0), (0, 14), (0, -16),
               (18, 10), (-18, 10), (18, -16), (-18, -16), (26, 4), (-26, 4), (0, 26), (0, -28)]
    placed = []
    for x, y, t in zip(xs, ys, texts):
        done = False
        for dx, dy in offsets:
            ann = ax.annotate(t, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=fontsize,
                              ha="left" if dx >= 0 else "right",
                              arrowprops=dict(arrowstyle="-", lw=0.4, color="grey"))
            bb = ann.get_window_extent(r)
            if not any(bb.overlaps(p) for p in placed):
                placed.append(bb)
                done = True
                break
            ann.remove()
        if not done:
            ax.annotate(t, (x, y), xytext=offsets[0], textcoords="offset points", fontsize=fontsize)


# =============================================================================
# 2. DATA + REGION
# =============================================================================
_REGION_LISTS = {
    "Africa": "DZA AGO BEN BWA BFA BDI CPV CMR CAF TCD COM COG COD CIV DJI EGY GNQ ERI SWZ ETH GAB GMB GHA GIN GNB KEN "
              "LSO LBR LBY MDG MWI MLI MRT MUS MAR MOZ NAM NER NGA RWA STP SEN SYC SLE SOM ZAF SSD SDN TZA TGO TUN UGA "
              "ZMB ZWE ESH REU MYT SHN",
    "Asia": "AFG ARM AZE BHR BGD BTN BRN KHM CHN CYP GEO HKG IND IDN IRN IRQ ISR JPN JOR KAZ KWT KGZ LAO LBN MAC MYS MDV "
            "MNG MMR NPL PRK OMN PAK PSE PHL QAT SAU SGP KOR LKA SYR TWN TJK THA TLS TUR TKM ARE UZB VNM YEM",
    "Europe": "ALB AND AUT BLR BEL BIH BGR HRV CZE DNK EST FRO FIN FRA DEU GIB GRC GGY HUN ISL IRL IMN ITA JEY XKX LVA "
              "LIE LTU LUX MLT MDA MCO MNE NLD MKD NOR POL PRT ROU RUS SMR SRB SVK SVN ESP SWE CHE UKR GBR VAT ALA SJM",
    "North America": "ATG BHS BRB BLZ CAN CRI CUB DMA DOM SLV GRD GTM HTI HND JAM MEX NIC PAN KNA LCA VCT TTO USA ABW AIA "
                     "BMU CYM CUW GRL GLP MTQ MSR PRI SXM TCA VGB VIR BES BLM MAF SPM",
    "South America": "ARG BOL BRA CHL COL ECU GUY PRY PER SUR URY VEN FLK GUF",
    "Oceania": "AUS FJI KIR MHL FSM NRU NZL PLW PNG WSM SLB TON TUV VUT ASM COK GUM NCL NIU NFK MNP PYF PCN TKL WLF",
}
ISO3_TO_REGION = {c: r for r, s in _REGION_LISTS.items() for c in s.split()}


def derive_region(codes: pd.Series) -> pd.Series:
    """Static ISO3->continent map (offline, deterministic); country_converter fills any unmapped code."""
    reg = codes.map(ISO3_TO_REGION)
    missing = codes[reg.isna()].dropna().unique().tolist()
    if missing:
        try:
            import country_converter as coco
            cc = coco.CountryConverter()
            conv = cc.convert(names=missing, src="ISO3", to="continent", not_found=None)
            conv = [conv] if isinstance(conv, str) else conv
            fix = {"America": "Americas"}
            mp = {m: fix.get(c, c) for m, c in zip(missing, conv)}
            reg = reg.fillna(codes.map(mp))
        except Exception:
            pass
    return reg.fillna("Other/Unknown")


def find_master(path=None):
    if path is None and os.path.exists(MASTER_CSV):
        return str(MASTER_CSV)
    cands = ([path] if path else []) + ["master_dataset.csv", "./input/master_dataset.csv"]
    cands += glob.glob("/kaggle/input/**/master_dataset.csv", recursive=True)
    for c in cands:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError("master_dataset.csv not found (checked cwd and /kaggle/input).")


def load_countries(path=None):
    p = find_master(path)
    df = pd.read_csv(p, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    agg = df["is_aggregate"].map(lambda v: str(v).strip().lower() in {"true", "1", "1.0", "yes"})
    df = df[(~agg) & df["code"].notna()].copy()
    df["code"] = df["code"].astype(str).str.strip().str.upper()
    df["year"] = df["year"].astype(int)
    name_col = next((c for c in ["country", "entity", "country_name", "name"] if c in df.columns), None)
    df["name"] = df[name_col] if name_col else df["code"]
    df = df.drop(columns=[c for c in DUPLICATE_COLS + FORBIDDEN if c in df.columns])
    for c in CORE:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["region"] = region_of(df["code"])   # A1 continent lookup (shared)
    print(f"Loaded {p}: {len(df)} country-year rows, {df['code'].nunique()} countries, "
          f"years {df['year'].min()}-{df['year'].max()}")
    return df


# =============================================================================
# 3. ITERATION 1  (V4, V5, V6)
# =============================================================================
def snapshot_coverage(df, years=(2021, 2022, 2023, 2024, 2025)):
    rows = []
    for y in years:
        s = df[df.year == y]
        r = {"year": y, "countries_in_file": s.code.nunique(),
             "complete_core_cases": int(s[CORE].notna().all(axis=1).sum())}
        r.update({f"nn_{SHORT[c]}": int(s[c].notna().sum()) for c in CORE})
        rows.append(r)
    return pd.DataFrame(rows).set_index("year")


def prepare_iteration1(df, year=SNAPSHOT_YEAR):
    assert_no_leakage(CORE)
    cov = snapshot_coverage(df)
    show("Snapshot coverage (why 2024 is used; 2025 sparse)", cov)
    save_csv(cov, "iter1_snapshot_coverage.csv")

    s = df[df.year == year].set_index("code")
    n_rows = len(s)
    snap = s.dropna(subset=CORE).copy()
    print(f"\nIteration 1 snapshot {year}: {n_rows} country rows -> n = {len(snap)} complete cases "
          f"across {len(CORE)} core features ({n_rows - len(snap)} dropped, NO imputation).")
    show("Region counts (complete cases)", snap.region.value_counts().to_frame("n"))

    # edge-case flags (reported, not removed)
    edge = {
        "mobile_subs_per100 > 100 (multi-SIM markets)": int((snap.mobile_subs_per100 > 100).sum()),
        "mobile_subs_per100 > 150": int((snap.mobile_subs_per100 > 150).sum()),
        "broadband == 0": int((snap.broadband_subs_per100 == 0).sum()),
        "landlines < 1 per 100": int((snap.telephone_lines_per100 < 1).sum()),
        "phone ownership >= 99%": int((snap.mobile_phone_ownership_share_pct >= 99).sum()),
    }
    show("Edge cases in snapshot (kept; flagged)", pd.Series(edge, name="n").to_frame())

    # context columns used only for anomaly TAGGING (never as clustering features)
    piv = df.pivot_table(index="code", columns="year", values="internet_users_share_pct")
    prev = year - 10
    snap["internet_delta10"] = (snap["internet_users_share_pct"] - piv[prev].reindex(snap.index)
                                if prev in piv.columns else np.nan)
    own = snap.mobile_phone_ownership_share_pct.where(snap.mobile_phone_ownership_share_pct > 0)
    snap["sim_per_owner"] = snap.mobile_subs_per100 / own

    scaler = StandardScaler()
    X = scaler.fit_transform(snap[CORE].values)
    return {"snap": snap, "X": X, "scaler": scaler, "year": year, "n": len(snap)}


# ---------- V4 ----------
def run_v4_pca(it1):
    snap, X = it1["snap"], it1["X"]
    pca = PCA(random_state=SEED).fit(X)
    Z = pca.transform(X)
    it1["pca"], it1["Z"] = pca, Z
    ev = pca.explained_variance_ratio_

    regions = sorted(snap.region.unique())
    pal = dict(zip(regions, sns.color_palette("tab10", len(regions))))
    fig, ax = plt.subplots(figsize=(11, 8))
    for r in regions:
        m = (snap.region == r).values
        ax.scatter(Z[m, 0], Z[m, 1], s=38, alpha=0.8, color=pal[r], label=f"{r} ({m.sum()})",
                   edgecolor="white", linewidth=0.4)
    load = pca.components_[:2].T * np.sqrt(pca.explained_variance_[:2])
    scale = 0.85 * min(np.abs(Z[:, 0]).max(), np.abs(Z[:, 1]).max()) / np.abs(load).max()
    xr, yr = np.ptp(Z[:, 0]), np.ptp(Z[:, 1])
    placed_tips = []
    for (lx, ly), f in zip(load * scale, CORE):
        ax.annotate("", xy=(lx, ly), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.6))
        tx, ty = lx * 1.06, ly * 1.06
        shift = 0
        while any(abs(tx - px) < 0.22 * xr and abs(ty - py) < 0.05 * yr for px, py in placed_tips) and shift < 6:
            ty += 0.055 * yr          # stagger labels of near-collinear arrows
            shift += 1
        placed_tips.append((tx, ty))
        ax.text(tx, ty, SHORT[f], fontsize=9, weight="bold", ha="left" if lx >= 0 else "right",
                va="bottom" if ly >= 0 else "top",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    far = np.argsort(-np.hypot(Z[:, 0], Z[:, 1]))[:8]
    place_labels(ax, Z[far, 0], Z[far, 1], [snap.name.iloc[i] for i in far])
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel(f"PC1 ({ev[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1%} var)")
    ax.set_title(f"V4 - PCA biplot, {it1['year']} cross-section (n = {it1['n']}, complete cases)")
    ax.legend(title="Region", loc="upper left", fontsize=8, framealpha=0.9)
    ins = ax.inset_axes([0.72, 0.66, 0.26, 0.28])   # empty upper-right corner, clear of loading labels
    ins.bar(range(1, len(ev) + 1), ev, color="#4c72b0")
    ins.plot(range(1, len(ev) + 1), np.cumsum(ev), "o-", color="#c44e52", ms=3)
    ins.set_xticks(range(1, len(ev) + 1))
    ins.set_ylim(0, 1.05)
    ins.set_title("Scree / cumulative var", fontsize=8)
    ins.tick_params(labelsize=7)
    ins.patch.set_alpha(0.9)
    save_fig(fig, "V4_pca_biplot.png")

    load_df = pd.DataFrame(pca.components_.T, index=[SHORT[c] for c in CORE],
                           columns=[f"PC{i+1}" for i in range(len(CORE))]).round(3)
    show("PCA loadings", load_df)
    show("Explained variance", pd.DataFrame({"var_ratio": ev.round(4), "cum": np.cumsum(ev).round(4)},
                                            index=[f"PC{i+1}" for i in range(len(ev))]))
    save_csv(load_df, "iter1_pca_loadings.csv")


# ---------- V5 ----------
def _fit_methods(X, k, seed=SEED):
    km = KMeans(k, n_init=25, random_state=seed).fit(X)
    gmm = GaussianMixture(k, covariance_type="full", n_init=5, reg_covar=1e-4, random_state=seed).fit(X)
    ward = AgglomerativeClustering(k, linkage="ward").fit(X)
    return km, gmm, ward


def validate_k(X):
    rows = []
    for k in K_RANGE:
        km, gmm, ward = _fit_methods(X, k)
        rows.append({
            "k": k,
            "sil_kmeans": silhouette_score(X, km.labels_),
            "sil_gmm": silhouette_score(X, gmm.predict(X)),
            "sil_ward": silhouette_score(X, ward.labels_),
            "min_size_kmeans": int(np.bincount(km.labels_).min()),
            "inertia": km.inertia_, "gmm_bic": gmm.bic(X), "gmm_aic": gmm.aic(X),
        })
    return pd.DataFrame(rows).set_index("k")


def _assign_to_centroids(X, Xb, lab_b):
    ids = np.unique(lab_b)
    cents = np.vstack([Xb[lab_b == c].mean(0) for c in ids])
    return ids[cdist(X, cents).argmin(1)]


def bootstrap_ari(X, k, ref, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(X)
    out = {"kmeans": [], "gmm": [], "ward": []}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        Xb = X[idx]
        km = KMeans(k, n_init=5, random_state=seed + b).fit(Xb)
        out["kmeans"].append(adjusted_rand_score(ref["kmeans"], km.predict(X)))
        gm = GaussianMixture(k, n_init=2, reg_covar=1e-4, random_state=seed + b).fit(Xb)
        out["gmm"].append(adjusted_rand_score(ref["gmm"], gm.predict(X)))
        wd = AgglomerativeClustering(k, linkage="ward").fit(Xb)
        out["ward"].append(adjusted_rand_score(ref["ward"], _assign_to_centroids(X, Xb, wd.labels_)))
    return pd.DataFrame(out)


def run_v5_validation(it1):
    snap, X = it1["snap"], it1["X"]
    val = validate_k(X)
    min_sz = max(3, int(0.03 * len(X)))        # reject degenerate (tiny-cluster) solutions
    cand = val.loc[list(K_SELECT)]
    ok = cand[cand.min_size_kmeans >= min_sz]
    cand = ok if len(ok) else cand
    k = int(cand.sil_kmeans.idxmax())
    it1["k"] = k
    print(f"\nChosen k = {k} (max k-means silhouette within k in {min(K_SELECT)}..{max(K_SELECT)} with every cluster >= {min_sz}; "
          f"GMM-BIC minimum at k = {int(val.gmm_bic.idxmin())}).")
    show("Cluster validation by k", val.round(3))
    save_csv(val.round(4), "iter1_cluster_validation.csv")

    km, gmm, ward = _fit_methods(X, k)
    inet = snap["internet_users_share_pct"].values
    labs = {"kmeans": relabel_by(km.labels_, inet),
            "gmm": relabel_by(gmm.predict(X), inet),
            "ward": relabel_by(ward.labels_, inet)}
    it1["labels"] = labs
    it1["gmm_post"] = gmm.predict_proba(X).max(1)

    boot = bootstrap_ari(X, k, labs)
    bs = boot.agg(["mean", "std", lambda s: s.quantile(.05)]).T
    bs.columns = ["ARI_mean", "ARI_sd", "ARI_p05"]
    show(f"Bootstrap ARI stability (B = {N_BOOT})", bs.round(3))
    save_csv(bs.round(4), "iter1_bootstrap_ari.csv")

    names = list(labs)
    agree = pd.DataFrame([[adjusted_rand_score(labs[a], labs[b]) for b in names] for a in names],
                         index=names, columns=names)
    show("Between-method ARI (final k)", agree.round(3))
    sizes = pd.DataFrame({m: pd.Series(l).value_counts().sort_index() for m, l in labs.items()})
    sizes.index.name = "cluster(0=low connectivity)"
    show("Cluster sizes", sizes)
    save_csv(sizes, "iter1_cluster_sizes.csv")
    final_sil = pd.Series({m: silhouette_score(X, l) for m, l in labs.items()}, name="silhouette")
    show("Silhouette at final k", final_sil.round(3).to_frame())

    # --- validation panel ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    ax = axs[0, 0]
    for c, lab in [("sil_kmeans", "k-means"), ("sil_gmm", "GMM"), ("sil_ward", "Ward")]:
        ax.plot(val.index, val[c], "o-", label=lab)
    ax.axvline(k, ls="--", color="grey")
    ax.set(title="Silhouette vs k", xlabel="k", ylabel="mean silhouette")
    ax.legend()
    ax = axs[0, 1]
    ax.plot(val.index, val.inertia, "o-", color="#55a868")
    ax.axvline(k, ls="--", color="grey")
    ax.set(title="Inertia / elbow (k-means)", xlabel="k", ylabel="inertia")
    ax = axs[1, 0]
    ax.plot(val.index, val.gmm_bic, "o-", color="#c44e52", label="BIC")
    ax.plot(val.index, val.gmm_aic, "s--", color="#8172b2", label="AIC")
    ax.axvline(k, ls="--", color="grey")
    ax.set(title="GMM information criteria", xlabel="k", ylabel="criterion (lower = better)")
    ax.legend()
    ax = axs[1, 1]
    sns.boxplot(data=boot, ax=ax, palette="Set2", width=0.5)
    ax.set(title=f"Bootstrap ARI stability (B = {N_BOOT}, k = {k})", ylabel="ARI vs reference", ylim=(0, 1.02))
    fig.suptitle(f"V5a - Cluster validation, {it1['year']} (n = {it1['n']})", fontsize=14, weight="bold")
    fig.tight_layout()
    save_fig(fig, "V5a_cluster_validation.png")

    # --- parallel coordinates: medians + IQR band ---
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.6), sharey=True)
    pal = sns.color_palette("viridis", k)
    xs = np.arange(len(CORE))
    for ax, (m, lab) in zip(axs, labs.items()):
        for c in range(k):
            Xc = X[lab == c]
            if len(Xc) == 0:
                continue
            for row in Xc:
                ax.plot(xs, row, color=pal[c], alpha=0.05, lw=0.8)
            q1, med, q3 = np.percentile(Xc, [25, 50, 75], axis=0)
            ax.fill_between(xs, q1, q3, color=pal[c], alpha=0.22)
            ax.plot(xs, med, "o-", color=pal[c], lw=2.4, label=f"C{c} (n={len(Xc)})")
        ax.set_xticks(xs)
        ax.set_xticklabels([SHORT[c] for c in CORE], rotation=25, ha="right")
        ax.set_title(f"{m.upper()} - median + IQR (z-scores)")
        ax.legend(fontsize=8, loc="upper left")
    axs[0].set_ylabel("standardised value")
    fig.suptitle("V5b - Parallel-coordinate cluster profiles (C0 = lowest internet share)",
                 fontsize=14, weight="bold")
    fig.tight_layout()
    save_fig(fig, "V5b_parallel_profiles.png")

    prof = snap[CORE].assign(c=labs["kmeans"]).groupby("c").median().round(2)
    prof.insert(0, "n", pd.Series(labs["kmeans"]).value_counts().sort_index())
    show("K-means cluster medians (raw units)", prof)
    save_csv(prof, "iter1_kmeans_cluster_medians.csv")


# ---------- V6 ----------
def _tag(row, p10, p90):
    tags = []
    spo = row.sim_per_owner
    if (pd.notna(spo) and spo >= MULTI_SIM_RATIO) or (pd.isna(spo) and row.mobile_subs_per100 >= MULTI_SIM_PER100):
        tags.append("multi-SIM inflation")
    if row.z_landline > 1.0 and row.z_internet < 0:
        tags.append("high landline / low internet")
    if pd.notna(row.internet_delta10):
        if row.internet_delta10 >= p90:
            tags.append("fast climber")
        elif row.internet_delta10 <= p10 and row.internet_users_share_pct < 80:
            tags.append("laggard")          # a small gain at the 100% ceiling is saturation, not lag
    ext = [SHORT[c] for c in CORE if abs(row[f"z_{c}"]) > 2.5]
    if ext and not tags:
        tags.append("extreme " + ", ".join(ext))
    return " + ".join(tags) if tags else "multivariate outlier"


def run_v6_anomalies(it1):
    snap, X, Z = it1["snap"], it1["X"], it1["Z"]
    n = len(X)
    iso = IsolationForest(n_estimators=500, contamination="auto", random_state=SEED).fit(X)
    s_if = -iso.score_samples(X)
    lof = LocalOutlierFactor(n_neighbors=min(15, n - 1))
    lof.fit_predict(X)
    s_lof = -lof.negative_outlier_factor_
    r_if, r_lof = rankdata(s_if) / n, rankdata(s_lof) / n
    comb = (r_if + r_lof) / 2

    sc = pd.DataFrame({"if_score": s_if, "lof_score": s_lof, "if_rank_pct": r_if,
                       "lof_rank_pct": r_lof, "anomaly_combined": comb}, index=snap.index)
    sc["anomaly_rank"] = sc.anomaly_combined.rank(ascending=False, method="first").astype(int)
    ctx = snap.copy()
    ctx["z_landline"] = X[:, CORE.index("telephone_lines_per100")]
    ctx["z_internet"] = X[:, CORE.index("internet_users_share_pct")]
    for j, c in enumerate(CORE):
        ctx[f"z_{c}"] = X[:, j]
    d = ctx["internet_delta10"].dropna()
    p10, p90 = (d.quantile(.10), d.quantile(.90)) if len(d) else (np.nan, np.nan)
    sc["reason"] = ctx.apply(_tag, axis=1, args=(p10, p90))
    it1["scores"] = sc

    top = sc.sort_values("anomaly_rank").head(TOP_N_ANOM)
    top_tbl = top.join(snap[["name", "region"] + CORE + ["sim_per_owner", "internet_delta10"]])
    top_tbl["in_top15_IF"] = top_tbl.if_score >= np.sort(s_if)[-TOP_N_ANOM]
    top_tbl["in_top15_LOF"] = top_tbl.lof_score >= np.sort(s_lof)[-TOP_N_ANOM]
    cols = ["anomaly_rank", "name", "region", "reason", "anomaly_combined", "if_score", "lof_score",
            "in_top15_IF", "in_top15_LOF", "mobile_subs_per100", "sim_per_owner",
            "telephone_lines_per100", "internet_users_share_pct", "internet_delta10"]
    show(f"Top {TOP_N_ANOM} anomalies (IF + LOF rank-average)", top_tbl[cols].round(2))
    save_csv(top_tbl[cols].round(4), "iter1_anomalies_top15.csv")
    both = int((top_tbl.in_top15_IF & top_tbl.in_top15_LOF).sum())
    print(f"\nTop-{TOP_N_ANOM} agreement between IF and LOF top-{TOP_N_ANOM} lists: {both} countries "
          f"| Spearman(IF, LOF) = {pd.Series(s_if).corr(pd.Series(s_lof), method='spearman'):.2f}")

    # --- lollipop ---
    cats = sorted({r.split(" + ")[0] for r in top.reason})
    cpal = dict(zip(cats, sns.color_palette("Set1", len(cats))))
    t = top.iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ypos = np.arange(len(t))
    colors = [cpal[r.split(" + ")[0]] for r in t.reason]
    ax.hlines(ypos, 0, t.anomaly_combined, color=colors, lw=2)
    ax.scatter(t.anomaly_combined, ypos, color=colors, s=90, zorder=3, edgecolor="k")
    ax.scatter(t.if_rank_pct, ypos, marker="^", color="k", s=28, zorder=4)
    ax.scatter(t.lof_rank_pct, ypos, marker="s", color="white", edgecolor="k", s=26, zorder=4)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{snap.name[c]}  [{r}]" for c, r in zip(t.index, t.reason)], fontsize=8)
    ax.set_xlim(0.5, 1.02)
    ax.set_xlabel("anomaly percentile (rank-average of Isolation Forest and LOF)")
    ax.set_title(f"V6a - Top {TOP_N_ANOM} anomalies, {it1['year']} (n = {n})")
    handles = [Line2D([], [], marker="o", color=cpal[c], ls="", label=c) for c in cats]
    handles += [Line2D([], [], marker="^", color="k", ls="", label="IF percentile"),
                Line2D([], [], marker="s", mfc="white", mec="k", color="k", ls="", label="LOF percentile")]
    ax.legend(handles=handles, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    save_fig(fig, "V6a_anomaly_lollipop.png")

    # --- PCA overlay ---
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(Z[:, 0], Z[:, 1], s=18, color="lightgrey", label="all countries")
    isin = sc.index.isin(top.index)
    sizes = 40 + 260 * (comb[isin] - comb.min()) / (comb.max() - comb.min() + 1e-9)
    ax.scatter(Z[isin, 0], Z[isin, 1], s=sizes, c=[cpal[r.split(" + ")[0]] for r in sc.reason[isin]],
               edgecolor="k", zorder=3, alpha=0.9)
    place_labels(ax, Z[isin, 0], Z[isin, 1], [f"{snap.name[c]}" for c in sc.index[isin]], fontsize=7.5)
    handles = [Line2D([], [], marker="o", color=cpal[c], ls="", label=c) for c in cats]
    ax.legend(handles=handles, title="Primary tag", fontsize=8, loc="upper left")
    ev = it1["pca"].explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]:.1%})")
    ax.set_ylabel(f"PC2 ({ev[1]:.1%})")
    ax.set_title(f"V6b - Anomalies on PCA space (marker size = anomaly score), n = {n}")
    save_fig(fig, "V6b_anomaly_pca_overlay.png")

    print("\nLinks to Assignment-1 figures (edit A1_LINKS in config):")
    for k_, v in A1_LINKS.items():
        print(f"  {k_}: {v}")
    print("  Fig. 6/9/10 -> relate to mobile vs landline vs internet patterns; "
          "Fig. 17/18 -> relate to multi-SIM / growth-trajectory cases (check against your A1 captions).")


def run_iteration1(df):
    print("\n" + "=" * 78 + "\nITERATION 1\n" + "=" * 78)
    it1 = prepare_iteration1(df)
    run_v4_pca(it1)
    run_v5_validation(it1)
    run_v6_anomalies(it1)
    lab = pd.DataFrame({"iter1_kmeans": it1["labels"]["kmeans"], "iter1_gmm": it1["labels"]["gmm"],
                        "iter1_ward": it1["labels"]["ward"], "iter1_gmm_maxprob": it1["gmm_post"]},
                       index=it1["snap"].index)
    it1["assign"] = lab.join(it1["scores"])
    save_csv(it1["snap"][["name", "region"] + CORE].join(it1["assign"]), "iter1_assignments.csv")
    return it1


# =============================================================================
# 4. FEEDBACK LOOP FEATURES
# =============================================================================
def _crossover_years(g, snap_year):
    """Years since mobile subs/100 permanently exceeded landlines/100 (NaN if not yet / no overlap)."""
    s = g[["mobile_subs_per100", "telephone_lines_per100"]].dropna()
    if s.empty:
        return np.nan
    gt = (s.mobile_subs_per100 > s.telephone_lines_per100).values
    yrs = s.index.values
    if not gt[-1]:
        return np.nan
    false_idx = np.where(~gt)[0]
    cross = yrs[0] if len(false_idx) == 0 else yrs[false_idx[-1] + 1]
    return float(snap_year - cross)


def build_feedback_features(df, it1, year=SNAPSHOT_YEAR):
    print("\n" + "=" * 78 + "\nFEEDBACK LOOP FEATURES\n" + "=" * 78)
    hist_all = df[df.year <= year].sort_values(["code", "year"])
    rows = []
    for code, g in hist_all.groupby("code"):
        g = g.set_index("year")
        cur = g.loc[year] if year in g.index else pd.Series(dtype=float)
        mob = cur.get("mobile_subs_per100", np.nan)
        own = cur.get("mobile_phone_ownership_share_pct", np.nan)
        bb = cur.get("broadband_subs_per100", np.nan)
        ll = g["telephone_lines_per100"].dropna()
        ll_peak = ll.max() if len(ll) else np.nan
        ll_peak_year = ll.idxmax() if len(ll) else np.nan
        ll_cur = ll.iloc[-1] if len(ll) else np.nan          # latest valid <= snapshot year
        ll_cur_year = ll.index[-1] if len(ll) else np.nan
        spo = mob / own if (pd.notna(mob) and pd.notna(own) and own > 0) else np.nan
        rows.append({
            "code": code, "name": g["name"].iloc[0], "region": g["region"].iloc[0], "snapshot_year": year,
            "sim_per_owner": spo,
            "multi_SIM": (pd.NA if pd.isna(spo) else int(spo > MULTI_SIM_RATIO)),
            "landline_peak": ll_peak, "landline_peak_year": ll_peak_year,
            "landline_current": ll_cur, "landline_current_year": ll_cur_year,
            "landline_decline": ll_peak - ll_cur if pd.notna(ll_peak) else np.nan,
            # leapfrog: mobile density today relative to the landline base ever built (+1 avoids /0)
            "leapfrog_index": mob / (ll_peak + 1) if (pd.notna(mob) and pd.notna(ll_peak)) else np.nan,
            "years_since_mobile_gt_landline": _crossover_years(g, year),
            "broadband_log1p": np.log1p(bb) if pd.notna(bb) and bb >= 0 else np.nan,
            "broadband_zero_flag": (pd.NA if pd.isna(bb) else int(bb == 0)),
        })
    ff = pd.DataFrame(rows).set_index("code")
    ff["multi_SIM"] = ff["multi_SIM"].astype("Int8")
    ff["broadband_zero_flag"] = ff["broadband_zero_flag"].astype("Int8")

    ff = ff.join(it1["assign"][["iter1_kmeans", "iter1_gmm", "iter1_ward", "if_score", "lof_score",
                                "anomaly_combined", "anomaly_rank", "reason"]]
                 .rename(columns={"reason": "anomaly_reason"}))
    ff["in_iter1_sample"] = ff["iter1_kmeans"].notna()

    eng = ["sim_per_owner", "multi_SIM", "landline_peak", "landline_decline", "leapfrog_index",
           "years_since_mobile_gt_landline", "broadband_log1p", "broadband_zero_flag"]
    cov = pd.DataFrame({"non_null": ff[eng].notna().sum(), "coverage_%": (ff[eng].notna().mean() * 100).round(1)})
    print(f"Countries in feedback table: {len(ff)} | in Iteration 1 sample: {int(ff.in_iter1_sample.sum())}")
    show("Engineered feature coverage", cov)
    show("Engineered feature summary", ff[eng].astype(float).describe().T.round(2))
    print(f"\nmulti_SIM flagged (ratio > {MULTI_SIM_RATIO}): {int((ff.multi_SIM == 1).sum())} of "
          f"{int(ff.multi_SIM.notna().sum())} with ownership data")
    print(f"landline_current from a year earlier than {year}: "
          f"{int((ff.landline_current_year < year).sum())} countries (see landline_current_year)")
    save_csv(ff, "feedback_features.csv")
    return ff


# =============================================================================
# 5. ITERATION 2  (trajectories + V8)
# =============================================================================
def build_trajectories(df):
    yrs = list(range(TRAJ_Y0, TRAJ_Y1 + 1))
    piv = (df.pivot_table(index="code", columns="year", values="internet_users_share_pct")
             .reindex(columns=yrs))
    n_any = int(piv.notna().any(axis=1).sum())
    obs = piv.notna().sum(axis=1)
    keep = piv[obs >= MIN_OBS].copy()
    n_thr = len(keep)
    # transparent gap handling: interior linear interpolation, limited edge carry-out, then drop leftovers
    keep = keep.interpolate(axis=1, limit_area="inside")
    keep = keep.ffill(axis=1, limit=EDGE_LIMIT).bfill(axis=1, limit=EDGE_LIMIT)
    still = keep.isna().any(axis=1)
    keep = keep[~still].clip(0, 100)
    filled = (piv.loc[keep.index].isna()).sum().sum()
    print(f"Trajectory panel {TRAJ_Y0}-{TRAJ_Y1}: {n_any} countries with any data -> {n_thr} with >= {MIN_OBS} "
          f"observed years -> n = {len(keep)} after gap handling ({int(still.sum())} dropped for unfillable edges). "
          f"Cells filled: {int(filled)} (interior linear interpolation; edge carry <= {EDGE_LIMIT} yrs).")
    return keep


def _dtw_pair(a, b, w):
    n, m = len(a), len(b)
    w = max(w, abs(n - m))
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - w), min(m, i + w) + 1):
            c = (a[i - 1] - b[j - 1]) ** 2
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return np.sqrt(D[n, m])


def dtw_distance_matrix(X, band=DTW_BAND):
    """DTW matrix: tslearn -> dtaidistance -> pure NumPy/Python fallback. Returns (D, backend)."""
    try:
        from tslearn.metrics import cdist_dtw
        return cdist_dtw(X[:, :, None], sakoe_chiba_radius=band, n_jobs=-1), "tslearn"
    except Exception:
        pass
    try:
        from dtaidistance import dtw
        D = dtw.distance_matrix_fast(X.astype(np.double), window=band)
        D = np.where(np.isinf(D), D.T, D)
        np.fill_diagonal(D, 0)
        return D, "dtaidistance"
    except Exception:
        pass
    n = len(X)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = _dtw_pair(X[i], X[j], band)
    return D, "numpy-fallback"


def cluster_trajectories(traj):
    X = traj.values
    D, backend = dtw_distance_matrix(X)
    print(f"DTW backend: {backend}  (Sakoe-Chiba radius {DTW_BAND} yrs; raw percentage-point scale, not z-normalised)")
    Z = linkage(squareform(D, checks=False), method="average")
    n = len(X)
    min_sz = max(3, int(0.03 * n))
    rows, labs = [], {}
    for k in K_RANGE:
        l = fcluster(Z, k, criterion="maxclust") - 1
        labs[k] = l
        if len(np.unique(l)) < 2:
            continue
        rows.append({"k": k, "silhouette_dtw": silhouette_score(D, l, metric="precomputed"),
                     "min_cluster_size": int(np.bincount(l).min())})
    val = pd.DataFrame(rows).set_index("k")
    ok = val[(val.index.isin(list(K_SELECT))) & (val.min_cluster_size >= min_sz)]
    if ok.empty:
        ok = val[val.index.isin(list(K_SELECT))]
    k2 = int(ok.silhouette_dtw.idxmax())
    show(f"Trajectory clustering: k scan (average-linkage on DTW; min cluster size >= {min_sz})", val.round(3))
    print(f"Chosen k2 = {k2}")
    final = traj.mean(axis=1).values
    lab = relabel_by(labs[k2], np.asarray(traj.iloc[:, -1].values))
    lab_k1 = labs  # raw dict for matched-k analysis
    return {"D": D, "labels": lab, "k": k2, "val": val, "backend": backend, "Z": Z, "raw_labels": lab_k1}


def draw_alluvial(ax, ct, left_title, right_title):
    L, R = list(ct.index), list(ct.columns)
    total = ct.values.sum()
    gap = 0.04 * total
    colL = sns.color_palette("viridis", len(L))
    colR = sns.color_palette("magma_r", len(R) + 1)[1:]

    def layout(sizes):
        pos, y = {}, 0.0
        for key, s in sizes.items():
            pos[key] = [y, y + s]
            y += s + gap
        return pos, y - gap
    lpos, lh = layout({i: ct.loc[i].sum() for i in L})
    rpos, rh = layout({j: ct[j].sum() for j in R})
    H = max(lh, rh)
    lo, ro = (H - lh) / 2, (H - rh) / 2
    lpos = {k: [a + lo, b + lo] for k, (a, b) in lpos.items()}
    rpos = {k: [a + ro, b + ro] for k, (a, b) in rpos.items()}
    lcur = {i: lpos[i][0] for i in L}
    rcur = {j: rpos[j][0] for j in R}
    xs = np.linspace(0, 1, 60)
    s = 3 * xs ** 2 - 2 * xs ** 3
    w = 0.07
    for i in L:
        for j in R:
            v = ct.loc[i, j]
            if v == 0:
                continue
            y0, y1 = lcur[i], rcur[j]
            lcur[i] += v
            rcur[j] += v
            bot = y0 + (y1 - y0) * s
            ax.fill_between(w + xs * (1 - 2 * w), bot, bot + v, color=colL[L.index(i)], alpha=0.45, lw=0)
    for i in L:
        a, b = lpos[i]
        ax.add_patch(plt.Rectangle((0, a), w, b - a, color=colL[L.index(i)]))
        ax.text(-0.015, (a + b) / 2, f"C{i}\nn={ct.loc[i].sum()}", ha="right", va="center", fontsize=9)
    for j in R:
        a, b = rpos[j]
        ax.add_patch(plt.Rectangle((1 - w, a), w, b - a, color=colR[R.index(j)]))
        ax.text(1.015, (a + b) / 2, f"T{j}\nn={ct[j].sum()}", ha="left", va="center", fontsize=9)
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.02 * H, 1.10 * H)
    ax.axis("off")
    ax.text(w / 2, 1.03 * H, left_title, ha="left", va="bottom", weight="bold", fontsize=10)
    ax.text(1 - w / 2, 1.03 * H, right_title, ha="right", va="bottom", weight="bold", fontsize=10)


def sankey_plotly_optional(ct):
    try:
        import plotly.graph_objects as go
        L, R = list(ct.index), list(ct.columns)
        src, tgt, val = [], [], []
        for i, a in enumerate(L):
            for j, b in enumerate(R):
                if ct.loc[a, b] > 0:
                    src.append(i); tgt.append(len(L) + j); val.append(int(ct.loc[a, b]))
        fig = go.Figure(go.Sankey(node=dict(label=[f"C{a}" for a in L] + [f"T{b}" for b in R]),
                                  link=dict(source=src, target=tgt, value=val)))
        path = os.path.join(OUT_DIR, "V8_sankey_interactive.html")
        fig.write_html(path)
        print(f"  [saved] {path} (optional interactive Sankey)")
    except Exception:
        print("  plotly unavailable -> matplotlib alluvial only (fine).")


def run_v8(traj, t2, it1):
    codes, lab2, k2 = traj.index, t2["labels"], t2["k"]
    t2_df = pd.DataFrame({"traj_cluster": lab2}, index=codes)
    t2_df["name"] = it1["snap"]["name"].reindex(codes)
    common = codes.intersection(it1["snap"].index)
    l1 = pd.Series(it1["labels"]["kmeans"], index=it1["snap"].index).loc[common]
    l2 = t2_df.loc[common, "traj_cluster"]
    print(f"\nOverlap Iteration 1 x Iteration 2: n = {len(common)} "
          f"(Iter1 n = {it1['n']}, Iter2 n = {len(codes)}).")

    ct = pd.crosstab(l1, l2)
    ct.index.name, ct.columns.name = "iter1_cluster", "traj_cluster"
    show("Contingency: Iteration 1 clusters x trajectory clusters", ct)
    save_csv(ct, "iter2_crosstab.csv")

    # ---- before / after comparison in a shared sample ----
    Xs = pd.DataFrame(it1["X"], index=it1["snap"].index).loc[common].values
    pos = [list(codes).index(c) for c in common]
    Dc = t2["D"][np.ix_(pos, pos)]

    def sil(feat, lab, metric=None):
        if len(np.unique(lab)) < 2 or len(np.unique(lab)) >= len(lab):
            return np.nan
        return silhouette_score(feat, lab, metric=metric) if metric else silhouette_score(feat, lab)
    comp = pd.DataFrame({
        "Iter-1 feature space (2024 z-scores)": [sil(Xs, l1.values), sil(Xs, l2.values)],
        "Trajectory space (DTW 2000-25)": [sil(Dc, l1.values, "precomputed"), sil(Dc, l2.values, "precomputed")],
    }, index=["Iter-1 clusters (k-means)", "Iter-2 trajectory clusters"])
    ari = adjusted_rand_score(l1, l2)
    k1 = it1["k"]
    lab_k1 = pd.Series(t2["raw_labels"][k1], index=codes).loc[common]
    ari_matched = adjusted_rand_score(l1, lab_k1)
    show(f"Silhouette comparison (shared n = {len(common)})", comp.round(3))
    print(f"ARI(Iter1 k={k1}, Iter2 k={k2}) = {ari:.3f} | ARI with Iter2 forced to k={k1}: {ari_matched:.3f}")
    comp2 = comp.copy()
    comp2["ARI_iter1_vs_iter2"] = [ari, ari]
    comp2["ARI_matched_k"] = [ari_matched, ari_matched]
    save_csv(comp2.round(4), "iter2_comparison.csv")

    # ---- V8a alluvial ----
    fig, ax = plt.subplots(figsize=(10, 7))
    draw_alluvial(ax, ct, f"Iter 1 (2024), k={k1}", f"Trajectories (2000-25), k={k2}")
    fig.suptitle(f"V8a - Cluster flow: cross-section -> trajectory (n = {len(common)}, ARI = {ari:.2f})",
                 fontsize=13, weight="bold", y=0.97)
    save_fig(fig, "V8a_alluvial.png")
    sankey_plotly_optional(ct)

    # ---- V8b small multiples ----
    ncol = min(k2, 3)
    nrow = int(np.ceil(k2 / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.9 * nrow), sharex=True, sharey=True, squeeze=False)
    yrs = traj.columns.values
    pal = sns.color_palette("magma_r", k2 + 1)[1:]
    summ = []
    for c in range(k2):
        ax = axs[c // ncol, c % ncol]
        idx = np.where(lab2 == c)[0]
        Xc = traj.values[idx]
        for row in Xc:
            ax.plot(yrs, row, color=pal[c], alpha=0.22, lw=0.9)
        q1, q3 = np.percentile(Xc, [25, 75], axis=0)
        ax.fill_between(yrs, q1, q3, color=pal[c], alpha=0.30, label="IQR")
        ax.plot(yrs, Xc.mean(0), "--", color="black", lw=1.6, label="mean")
        ax.plot(yrs, np.median(Xc, 0), "-", color="black", lw=2.4, label="median")
        sub = t2["D"][np.ix_(idx, idx)]
        med_i = idx[sub.sum(1).argmin()]
        ax.plot(yrs, traj.values[med_i], color="crimson", lw=1.8, label="medoid")
        ax.set_title(f"T{c}  n={len(idx)}  medoid: {t2_df['name'].iloc[med_i]}", fontsize=10)
        ax.set_ylim(0, 100)
        if c == 0:
            ax.legend(fontsize=8, loc="upper left")
        summ.append({"traj_cluster": c, "n": len(idx), "medoid": t2_df["name"].iloc[med_i],
                     "median_2000": np.median(Xc[:, 0]), "median_2010": np.median(Xc[:, list(yrs).index(2010)]),
                     "median_2025": np.median(Xc[:, -1]),
                     "mean_gain_pp": (Xc[:, -1] - Xc[:, 0]).mean()})
    for k_ in range(k2, nrow * ncol):
        axs[k_ // ncol, k_ % ncol].axis("off")
    for ax in axs[-1]:
        ax.set_xlabel("year")
    for ax in axs[:, 0]:
        ax.set_ylabel("internet users (% pop.)")
    fig.suptitle(f"V8b - Trajectory clusters {TRAJ_Y0}-{TRAJ_Y1} (n = {len(codes)}, DTW + average linkage)",
                 fontsize=14, weight="bold")
    fig.tight_layout()
    save_fig(fig, "V8b_trajectory_small_multiples.png")
    summ = pd.DataFrame(summ).set_index("traj_cluster")
    show("Trajectory cluster summary", summ.round(1))
    save_csv(summ.round(3), "iter2_cluster_summary.csv")

    # ---- V8c before / after ----
    fig, ax = plt.subplots(figsize=(8, 5))
    comp.T.plot(kind="bar", ax=ax, color=["#4c72b0", "#dd8452"], rot=0, width=0.65)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("mean silhouette")
    ax.set_title(f"V8c - Before/after silhouette (shared n = {len(common)})\n"
                 f"ARI(Iter1, Iter2) = {ari:.2f}  |  matched-k ARI = {ari_matched:.2f}")
    ax.legend(title="Labels evaluated", fontsize=8)
    plt.setp(ax.get_xticklabels(), fontsize=9)
    save_fig(fig, "V8c_before_after_silhouette_ari.png")

    out = t2_df.copy()
    out["iter1_cluster"] = l1.reindex(out.index)
    save_csv(out, "iter2_assignments.csv")
    return t2_df


def run_iteration2(df, it1, ff):
    print("\n" + "=" * 78 + "\nITERATION 2 - TRAJECTORIES\n" + "=" * 78)
    traj = build_trajectories(df)
    t2 = cluster_trajectories(traj)
    t2_df = run_v8(traj, t2, it1)
    # feed trajectory cluster back to the shared feature table
    ff2 = ff.join(t2_df[["traj_cluster"]])
    save_csv(ff2, "feedback_features.csv")
    return t2_df


# =============================================================================
# 6. MAIN
# =============================================================================
def main(path=None):
    np.random.seed(SEED)
    ensure_dirs()
    df = load_countries(path)
    it1 = run_iteration1(df)                       # iteration 1: RAW data (anomalies found here)
    ff = build_feedback_features(df, it1)
    # FEEDBACK LOOP: the artefacts flagged by V6 + the A1 inferences become a data transformation
    clean, arts = transform_for_iteration2(df, verbose=True)
    run_iteration2(clean, it1, ff)                 # iteration 2: TRANSFORMED data
    print("\nDone. Figures in ./images, tables in ./outputs "
          "(feedback_features.csv lists the engineered features per country).")


if __name__ == "__main__":
    main()

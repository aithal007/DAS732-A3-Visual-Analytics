# %% [markdown]
# # DAS732 A3 — Unsupervised pillar (Sindhoor): 8 viva-ready diagnostics & visuals
# Input : OWID Internet panel `master_dataset.csv` (1960–2025)
# Output: ./images/*.png (300 DPI)  |  ./outputs/*.csv (summary tables)
#
# Design rules enforced in code
#   * countries only: (~is_aggregate) & code.notna()
#   * StandardScaler before every distance / PCA / clustering step
#   * default snapshot = 2024, exact n reported, NO imputation of cross-sectional features
#   * leakage guard: *_count, *_ict, broadband_subs_per100 can never enter FEATURES
#   * trajectories: only interior gaps interpolated (counted + reported), coverage threshold
#   * fixed seeds; optional libs (numba, umap) have offline-safe fallbacks

# %% [1] Imports & configuration
from __future__ import annotations

import glob
import random
import re
import warnings
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, leaves_list, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.manifold import MDS, TSNE
from sklearn.metrics import adjusted_rand_score, pairwise_distances, silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SEED = 42
SNAPSHOT_YEAR = 2024
K = 3                                   # chosen number of clusters
N_BOOT = 200                            # bootstrap runs (V1)
TRAJ_START, TRAJ_END = 1990, 2024       # trajectory window for DTW
MIN_TRAJ_COVERAGE = 0.80                # share of window years that must be observed
DTW_BAND = 10                           # Sakoe-Chiba radius (years)
ANOM_YEARS = list(range(2010, 2025))    # years scanned for anomaly persistence
ANOM_MIN_N = 100                        # skip years with fewer complete-case countries (coverage jumps in 2019)
ANOM_CONTAM = 0.05                      # IsolationForest contamination per year
TOP_ANOM_STRIP = 25                     # rows in persistence strip
LAG_TARGET, LAG_TOL = 8, 1.5            # DTW teaching pair: lag in years (+/- tol)
EMBED_SPACE = "snapshot"                # "snapshot" (5 features) or "dtw" (MDS on DTW matrix)

CORE = [
    "internet_users_share_pct",
    "mobile_subs_per100",
    "mobile_phone_ownership_share_pct",
    "telephone_lines_per100",
    "broadband_penetration_per100",
]
FEATURES = CORE.copy()
LEAK_PATTERNS = [r"^internet_users_count$", r"^broadband_.*_count$", r"_ict$", r"^broadband_subs_per100$"]
REGION_CANDIDATES = ["region", "world_region", "continent", "un_region", "owid_region", "who_region"]

IMG_DIR, OUT_DIR = Path("./images"), Path("./outputs")
pass  # directories are created by the caller

I1_NAMES = ["Low adoption", "Mid adoption", "High adoption"]   # ordered by mean internet share
I2_NAMES = ["Early adopters", "Mid adopters", "Late adopters"]  # ordered by mean years-to-50%
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 11,
})

SUMMARY: dict = {}
REGION_COLORS: dict = {}   # optional override (team palette)   # key diagnostics -> outputs/run_summary.csv


_CONT = {
 "Africa": "DZA AGO BEN BWA BFA BDI CPV CMR CAF TCD COM COG COD CIV DJI EGY GNQ ERI SWZ ETH GAB GMB GHA GIN GNB KEN LSO LBR LBY MDG MWI MLI MRT MUS MAR MOZ NAM NER NGA RWA STP SEN SYC SLE SOM ZAF SSD SDN TZA TGO TUN UGA ZMB ZWE ESH REU MYT SHN",
 "Asia": "AFG ARM AZE BHR BGD BTN BRN KHM CHN CYP GEO HKG IND IDN IRN IRQ ISR JPN JOR KAZ KWT KGZ LAO LBN MAC MYS MDV MNG MMR NPL PRK OMN PAK PSE PHL QAT SAU SGP KOR LKA SYR TWN TJK THA TLS TUR TKM ARE UZB VNM YEM",
 "Europe": "ALB AND AUT BLR BEL BIH BGR HRV CZE DNK EST FRO FIN FRA DEU GIB GRC GGY HUN ISL IRL IMN ITA JEY OWID_KOS LVA LIE LTU LUX MLT MDA MCO MNE NLD MKD NOR POL PRT ROU RUS SMR SRB SVK SVN ESP SWE CHE UKR GBR VAT",
 "North America": "ATG BHS BRB BLZ CAN CRI CUB DMA DOM SLV GRD GTM HTI HND JAM MEX NIC PAN KNA LCA VCT TTO USA ABW BMU CYM CUW SXM TCA VGB VIR PRI GRL MAF BES AIA MSR GLP MTQ BLM SPM",
 "South America": "ARG BOL BRA CHL COL ECU GUY PRY PER SUR URY VEN FLK GUF",
 "Oceania": "AUS FJI KIR MHL FSM NRU NZL PLW PNG WSM SLB TON TUV VUT ASM GUM MNP NCL PYF COK NIU TKL WLF",
}
ISO3_TO_CONTINENT = {c: r for r, codes in _CONT.items() for c in codes.split()}   # offline fallback


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def assert_no_leakage(cols) -> None:
    bad = [c for c in cols if any(re.search(p, c) for p in LEAK_PATTERNS)]
    assert not bad, f"Leakage columns in feature set: {bad}"


def save_fig(fig, name: str) -> None:
    path = IMG_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[saved] {path}")
    plt.show()
    plt.close(fig)


def save_csv(df: pd.DataFrame, name: str, index: bool = True) -> None:
    path = OUT_DIR / name
    df.to_csv(path, index=index)
    print(f"[saved] {path}")


def show(df: pd.DataFrame, title: str, n: int = 25) -> None:
    print(f"\n=== {title} ===")
    print(df.head(n).to_string())


def names_for(k: int, names: list[str]) -> list[str]:
    return names if k == len(names) else [f"C{i}" for i in range(k)]


assert_no_leakage(FEATURES)


# %% [2] Data loading, snapshot, Iteration-1 clustering
def find_data(path: str | None = None) -> str:
    cands = ([path] if path else []) + ["master_dataset.csv", "./data/master_dataset.csv"]
    cands += glob.glob("/kaggle/input/**/master_dataset.csv", recursive=True)
    for c in cands:
        if c and Path(c).exists():
            return c
    raise FileNotFoundError("master_dataset.csv not found (checked cwd, ./data, /kaggle/input).")


def load_panel(path: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(find_data(path))
    ccol = next((c for c in ["country", "Entity", "entity", "name"] if c in df.columns), None)
    assert ccol is not None, "No country-name column found."
    df = df.rename(columns={ccol: "country"})
    agg = df["is_aggregate"].astype(str).str.lower().isin(["true", "1", "1.0"])
    df = df[(~agg) & df["code"].notna()].copy()            # countries only
    df["year"] = df["year"].astype(int)
    rcol = next((c for c in REGION_CANDIDATES if c in df.columns), None)
    if rcol:
        df["region"] = df[rcol].fillna("Unknown")
    else:                                                    # no region column -> built-in ISO3 -> continent map
        df["region"] = df["code"].map(ISO3_TO_CONTINENT).fillna("Other")
        rcol = "continent (built-in ISO3 map)"
        unm = sorted(df.loc[df["region"] == "Other", "code"].unique())
        print(f"  region fallback used; unmapped codes -> 'Other': {unm}")
    missing = [c for c in CORE if c not in df.columns]
    assert not missing, f"Missing core columns: {missing}"
    print(f"[panel] {df['code'].nunique()} countries, years {df['year'].min()}–{df['year'].max()}, "
          f"region col = {rcol}")
    return df


def relabel_by_mean(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Relabel clusters 0..k-1 by ascending mean of `values` (stable, interpretable colours)."""
    labs = np.unique(labels)
    means = {l: (np.nanmean(values[labels == l]) if np.any(~np.isnan(values[labels == l])) else np.inf)
             for l in labs}
    order = sorted(labs, key=lambda l: means[l])
    mp = {old: new for new, old in enumerate(order)}
    return np.array([mp[l] for l in labels])


def build_snapshot(panel: pd.DataFrame, year: int = SNAPSHOT_YEAR) -> SimpleNamespace:
    sub = panel[panel["year"] == year][["code", "country", "region"] + FEATURES].copy()
    n_raw = len(sub)
    miss = sub[FEATURES].isna().mean().round(3)
    sub = sub.dropna(subset=FEATURES).set_index("code")     # complete-case: NO imputation
    scaler = StandardScaler()
    X = scaler.fit_transform(sub[FEATURES])
    print(f"[snapshot {year}] rows={n_raw} -> complete-case n={len(sub)} (dropped {n_raw-len(sub)})")
    print("  missing share per feature before drop:", miss.to_dict())
    if len(sub) < 30:
        print("  WARNING: small snapshot; consider documenting an alternative year.")
    SUMMARY.update(snapshot_year=year, snapshot_n_raw=n_raw, snapshot_n=len(sub))
    return SimpleNamespace(df=sub, X=X, scaler=scaler, year=year, codes=sub.index.values)


def fit_iter1(snap: SimpleNamespace, k: int = K) -> None:
    km = KMeans(k, n_init=50, random_state=SEED).fit(snap.X)
    lab = relabel_by_mean(snap.df["internet_users_share_pct"].values, km.labels_)
    snap.df["cluster_iter1"] = lab
    snap.labels = lab
    cnt = snap.df["cluster_iter1"].value_counts().sort_index()
    cnt.index = [names_for(k, I1_NAMES)[i] for i in cnt.index]
    show(cnt.to_frame("n_countries"), f"Iteration-1 KMeans (k={k}) cluster counts")
    prof = snap.df.groupby("cluster_iter1")[FEATURES].mean().round(2)
    show(prof, "Iteration-1 cluster profiles (raw means)")
    SUMMARY["iter1_silhouette_k%d" % k] = round(silhouette_score(snap.X, lab), 4)


# %% [3] V1 — Bootstrap co-assignment heatmap
def bootstrap_coassignment(X, ref_labels, k=K, B=N_BOOT, seed=SEED):
    """Resample countries with replacement, refit KMeans, assign ALL countries to nearest centroid.
    co[i,j] = fraction of runs where i and j land in the same cluster (label-switching invariant)."""
    rng = np.random.default_rng(seed)
    n = len(X)
    co = np.zeros((n, n))
    aris = []
    for b in range(B):
        idx = rng.integers(0, n, n)
        km = KMeans(k, n_init=5, random_state=seed + b).fit(X[idx])
        lab = km.predict(X)
        co += (lab[:, None] == lab[None, :])
        aris.append(adjusted_rand_score(ref_labels, lab))
    return co / B, np.array(aris)


def plot_coassignment(snap, co, aris, k=K):
    lab, n = snap.labels, len(snap.labels)
    order = []
    for c in range(k):                                   # sort by cluster, then by block-internal similarity
        idx = np.where(lab == c)[0]
        if len(idx) > 2:
            Zc = linkage(squareform(1 - co[np.ix_(idx, idx)], checks=False), "average")
            idx = idx[leaves_list(Zc)]
        order.extend(idx)
    order = np.array(order)
    M, ls = co[np.ix_(order, order)], lab[order]
    cmap_c = ListedColormap(PALETTE[:k])

    fig = plt.figure(figsize=(9, 9))
    gs = fig.add_gridspec(2, 3, width_ratios=[0.025, 1, 0.03], height_ratios=[0.025, 1], wspace=0.02, hspace=0.02)
    ax, axl, axt, cax = (fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 0]),
                         fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 2]))
    im = ax.imshow(M, cmap="magma_r", vmin=0, vmax=1, interpolation="nearest", aspect="equal")
    axl.imshow(ls[:, None], cmap=cmap_c, aspect="auto", interpolation="nearest")
    axt.imshow(ls[None, :], cmap=cmap_c, aspect="auto", interpolation="nearest")
    for a in (ax, axl, axt):
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
    bounds = np.cumsum([np.sum(ls == c) for c in range(k)])[:-1] - 0.5
    for b in bounds:
        ax.axhline(b, color="white", lw=1.2); ax.axvline(b, color="white", lw=1.2)
    fig.colorbar(im, cax=cax).set_label("Co-assignment fraction")
    nm = names_for(k, I1_NAMES)
    axt.set_title(f"Bootstrap co-assignment (B={len(aris)}, k={k}, n={n}, {snap.year})\n"
                  f"mean ARI vs reference = {aris.mean():.2f} ± {aris.std():.2f}", pad=8)
    ax.legend(handles=[Patch(color=PALETTE[c], label=nm[c]) for c in range(k)],
              loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=k, frameon=False)
    save_fig(fig, "v1_bootstrap_coassignment.png")

    # ---- diagnostics
    block = pd.DataFrame([[co[np.ix_(lab == a, lab == b)].mean() for b in range(k)] for a in range(k)],
                         index=nm, columns=nm).round(3)
    show(block, "Mean co-assignment between cluster blocks (diag = within-block)")
    off = co[np.triu_indices(n, 1)]
    unstable = float(((off > 0.1) & (off < 0.9)).mean())
    stab = np.array([(co[i, lab == lab[i]].sum() - 1) / max((lab == lab[i]).sum() - 1, 1) for i in range(n)])
    per_c = pd.DataFrame({"country": snap.df["country"].values, "cluster": [nm[l] for l in lab],
                          "stability": stab.round(3)}, index=snap.codes).sort_values("stability")
    print(f"\n[V1] mean ARI={aris.mean():.3f} (sd {aris.std():.3f}); "
          f"share of country pairs with 0.1<co<0.9 = {unstable:.3f}")
    show(per_c, "Least stable countries (bootstrap co-assignment with own cluster)", 12)
    save_csv(per_c, "v1_country_stability.csv")
    save_csv(block, "v1_block_coassignment.csv")
    SUMMARY.update(boot_ari_mean=round(aris.mean(), 4), boot_ari_sd=round(aris.std(), 4),
                   boot_unstable_pair_share=round(unstable, 4))


# %% [4] V2 — Silhouette knife plots (k=2 vs k=3)
def neighbour_cluster(D, lab):
    """Second-best cluster for each point (the 'border' partner used in silhouette b(i))."""
    n, ks = len(lab), np.unique(lab)
    out = np.zeros(n, dtype=int)
    for i in range(n):
        means = {}
        for c in ks:
            m = lab == c
            if c == lab[i]:
                continue
            means[c] = D[i, m].mean()
        out[i] = min(means, key=means.get)
    return out


def plot_silhouette_knife(snap, ks=(2, 3), n_label=8, border_thr=0.10):
    X, n = snap.X, len(snap.X)
    D = pairwise_distances(X)
    share = snap.df["internet_users_share_pct"].values
    fig, axes = plt.subplots(1, len(ks), figsize=(6.5 * len(ks), max(8, 0.075 * n)), sharex=True)
    per_country = pd.DataFrame(index=snap.codes)
    per_country["country"] = snap.df["country"].values
    rows = []
    for ax, k in zip(np.atleast_1d(axes), ks):
        lab = relabel_by_mean(share, KMeans(k, n_init=50, random_state=SEED).fit_predict(X))
        s = silhouette_samples(X, lab)
        mean_s = s.mean()
        nb = neighbour_cluster(D, lab)
        per_country[f"cluster_k{k}"], per_country[f"sil_k{k}"], per_country[f"neighbour_k{k}"] = lab, s.round(4), nb
        y, ypos, ytick, ylab = 0, np.zeros(n), [], []
        for c in range(k):
            idx = np.where(lab == c)[0]
            idx = idx[np.argsort(-s[idx])]
            ypos[idx] = np.arange(y, y + len(idx))
            ax.barh(ypos[idx], s[idx], height=1.0, color=PALETTE[c], edgecolor="none")
            ytick.append(y + len(idx) / 2); ylab.append(f"C{c} (n={len(idx)})")
            y += len(idx) + 6
        ax.axvline(mean_s, color="k", ls="--", lw=1.2)
        ax.text(mean_s, -4, f"mean = {mean_s:.3f}", ha="center", va="bottom", fontsize=9)
        ax.axvline(0, color="0.4", lw=0.8)
        ax.set_yticks(ytick); ax.set_yticklabels(ylab); ax.invert_yaxis()
        ax.set_xlabel("Silhouette width"); ax.set_title(f"k = {k}  (mean {mean_s:.3f}, "
                                                         f"{(s < 0).sum()} negative)")
        for i in np.argsort(s)[:n_label]:                  # label the borderline countries
            if s[i] < border_thr:
                left = s[i] < 0
                ax.text(s[i] + (-0.01 if left else 0.01), ypos[i], snap.df["country"].iloc[i],
                        fontsize=6.5, ha="right" if left else "left", va="center")
        SUMMARY[f"silhouette_mean_k{k}"] = round(mean_s, 4)
        rows.append({"k": k, "mean_silhouette": round(mean_s, 4), "n_negative": int((s < 0).sum()),
                     "n_below_%.2f" % border_thr: int((s < border_thr).sum()),
                     "sizes": np.bincount(lab).tolist()})
    fig.suptitle(f"Per-country silhouettes — k=2 vs k=3 (n={n}, {snap.year}, standardized features)", y=1.0)
    fig.tight_layout()
    save_fig(fig, "v2_silhouette_knife_k2_vs_k3.png")
    show(pd.DataFrame(rows), "Silhouette summary (k=2 vs k=3)")
    kmax = max(ks)
    border = per_country[per_country[f"sil_k{kmax}"] < border_thr].sort_values(f"sil_k{kmax}")
    b = border[["country", f"cluster_k{kmax}", f"neighbour_k{kmax}", f"sil_k{kmax}"]].copy()
    b[f"cluster_k{kmax}"] = b[f"cluster_k{kmax}"].map(lambda i: names_for(kmax, I1_NAMES)[i])
    b[f"neighbour_k{kmax}"] = b[f"neighbour_k{kmax}"].map(lambda i: names_for(kmax, I1_NAMES)[i])
    show(b, f"Border countries at k={kmax} (sil < {border_thr})", 30)
    save_csv(per_country, "v2_silhouette_per_country.csv")
    return per_country


# %% [5] Trajectories, DTW distance (numba -> pure-python fallback), clustering
def _dtw_sq(a, b, band):
    n, m = len(a), len(b)
    INF = 1e18
    D = np.full((n + 1, m + 1), INF)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        jlo, jhi = max(1, i - band), min(m, i + band)
        for j in range(jlo, jhi + 1):
            c = (a[i - 1] - b[j - 1]) ** 2
            best = D[i - 1, j - 1]
            if D[i - 1, j] < best:
                best = D[i - 1, j]
            if D[i, j - 1] < best:
                best = D[i, j - 1]
            D[i, j] = c + best
    return D[n, m]


def _dtw_all(S, band):
    n = S.shape[0]
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(_dtw_sq(S[i], S[j], band))
            out[i, j] = d
            out[j, i] = d
    return out


try:                                                    # fast path if numba is installed (Kaggle: yes)
    from numba import njit
    _dtw_sq = njit(_dtw_sq)
    _dtw_all = njit(_dtw_all)
    DTW_BACKEND = "numba"
except Exception:
    DTW_BACKEND = "pure-python (slower)"


def dtw_path(a, b, band=None):
    band = DTW_BAND if band is None else band
    """DTW distance + warping path (squared-diff cost, sqrt at the end; same convention as _dtw_all)."""
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - band), min(m, i + band) + 1):
            D[i, j] = (a[i - 1] - b[j - 1]) ** 2 + min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
        i, j = (i - 1, j - 1) if step == 0 else ((i - 1, j) if step == 1 else (i, j - 1))
    return np.sqrt(D[n, m]), path[::-1]


def first_year_crossing(years, vals, thr=50.0):
    """First year the series crosses `thr` (linear interpolation). NaN if never crossed or
    already >= thr at the first observation (left-censored)."""
    m = ~np.isnan(vals)
    y, v = np.asarray(years)[m], np.asarray(vals)[m]
    if len(v) == 0 or v.max() < thr or v[0] >= thr:
        return np.nan
    i = int(np.argmax(v >= thr))
    y0, y1, v0, v1 = y[i - 1], y[i], v[i - 1], v[i]
    return float(y0 + (thr - v0) * (y1 - y0) / (v1 - v0)) if v1 != v0 else float(y1)


def build_trajectories(panel: pd.DataFrame) -> SimpleNamespace:
    wide_full = panel.pivot_table(index="code", columns="year", values="internet_users_share_pct", aggfunc="first")
    t50 = pd.Series({c: first_year_crossing(wide_full.columns.values, wide_full.loc[c].values)
                     for c in wide_full.index}, name="years_to_50pct")
    cols = list(range(TRAJ_START, TRAJ_END + 1))
    w = wide_full.reindex(columns=cols)
    cov = w.notna().mean(axis=1)
    keep = cov >= MIN_TRAJ_COVERAGE
    W = w[keep].copy()
    n_obs = W.notna().sum().sum()
    W = W.interpolate(axis=1, limit_direction="both")            # interior = linear; edges = nearest obs
    filled = W.size - n_obs
    print(f"[trajectories] window {TRAJ_START}–{TRAJ_END}: kept {keep.sum()}/{len(w)} countries "
          f"(coverage >= {MIN_TRAJ_COVERAGE:.0%}); filled {filled} of {W.size} cells ({filled/W.size:.1%})")
    meta = panel.drop_duplicates("code").set_index("code")[["country", "region"]]
    S = (W.values / 100.0).astype(float)                          # share in [0,1]
    print(f"[DTW] backend={DTW_BACKEND}, Sakoe-Chiba radius={DTW_BAND}, n={len(S)}")
    D = _dtw_all(S, DTW_BAND)
    codes = W.index.values
    SUMMARY.update(traj_n=len(codes), traj_filled_share=round(filled / W.size, 4), dtw_backend=DTW_BACKEND)
    return SimpleNamespace(W=W, S=S, D=D, codes=codes, years=np.array(cols), t50_all=t50,
                           t50=t50.reindex(codes).values, meta=meta.reindex(codes))


def cluster_trajectories(tr: SimpleNamespace, k=K):
    Z = linkage(squareform(tr.D, checks=False), method="average")
    lab_dtw = relabel_by_mean(tr.t50, fcluster(Z, k, "maxclust") - 1)
    lab_euc = relabel_by_mean(tr.t50, KMeans(k, n_init=50, random_state=SEED).fit_predict(tr.S))
    tr.Z, tr.lab_dtw, tr.lab_euc = Z, lab_dtw, lab_euc
    cnt = pd.Series(lab_dtw).value_counts().sort_index()
    cnt.index = [names_for(k, I2_NAMES)[i] for i in cnt.index]
    show(cnt.to_frame("n_countries"), f"Iteration-2 DTW hierarchical (average, k={k}) counts")
    print(f"ARI(DTW vs Euclidean-KMeans) = {adjusted_rand_score(lab_dtw, lab_euc):.3f}")
    SUMMARY["ari_dtw_vs_euclid"] = round(adjusted_rand_score(lab_dtw, lab_euc), 4)


# %% [6] V3 — Years-to-50% ridgeline (DTW clusters vs Euclidean clusters)
def eta_squared(values, labels):
    m = ~np.isnan(values)
    v, l = values[m], labels[m]
    ss_tot = ((v - v.mean()) ** 2).sum()
    ss_b = sum(((l == c).sum()) * (v[l == c].mean() - v.mean()) ** 2 for c in np.unique(l))
    return ss_b / ss_tot if ss_tot > 0 else np.nan


def _ridge(ax, t50, lab, k, names, xs, rng, title, min_kde=5):
    """One ridge per cluster (early on top), each scaled to its own peak so shapes are comparable.
    Clusters with < min_kde crossed countries show dots only (a KDE on 3 points is not meaningful)."""
    ticks, tlabs = [], []
    for c in range(k):
        base = (k - 1 - c) * 1.0
        v = t50[(lab == c) & ~np.isnan(t50)]
        n_not = int(((lab == c) & np.isnan(t50)).sum())
        ticks.append(base + 0.3)
        tlabs.append(f"{names[c]}\n{len(v)} crossed · {n_not} not yet >50%" + (f"\nmedian {np.median(v):.0f}" if len(v) else ""))
        if len(v) == 0:
            continue
        if len(v) >= min_kde and v.std() > 0:
            dens = gaussian_kde(v, bw_method=0.6)(xs)
            h = dens / dens.max() * 1.3
            ax.fill_between(xs, base, base + h, color=PALETTE[c], alpha=0.75, zorder=2 + c, lw=0)
            ax.plot(xs, base + h, color="white", lw=1, zorder=2 + c)
        ax.scatter(v, base - 0.14 + rng.uniform(-0.06, 0.06, len(v)), s=11, color=PALETTE[c],
                   alpha=0.85, zorder=10, edgecolor="none")
        ax.plot([np.median(v)] * 2, [base, base + 1.0], color="k", ls="--", lw=0.9, zorder=11)
    ax.set_yticks(ticks); ax.set_yticklabels(tlabs, fontsize=8.5)
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlabel("Year internet-user share first crosses 50%")
    ax.set_title(title)


def plot_ridgeline(tr: SimpleNamespace, k=K):
    t50 = tr.t50
    ok = ~np.isnan(t50)
    xs = np.linspace(np.nanmin(t50) - 4, np.nanmax(t50) + 4, 400)
    eta_d, eta_e = eta_squared(t50, tr.lab_dtw), eta_squared(t50, tr.lab_euc)
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharex=True)
    _ridge(axes[0], t50, tr.lab_dtw, k, names_for(k, I2_NAMES), xs, rng,
           f"DTW trajectory clusters (η² of T50 = {eta_d:.2f})")
    _ridge(axes[1], t50, tr.lab_euc, k, names_for(k, I2_NAMES), xs, rng,
           f"Euclidean K-means clusters (η² of T50 = {eta_e:.2f})")
    n_cens = int((~ok).sum())
    fig.suptitle("Years-to-50% internet adoption by trajectory cluster — shifted S-curves "
                 f"(n={ok.sum()} crossed; {n_cens} not crossed/left-censored, excluded)", y=1.02)
    fig.tight_layout()
    save_fig(fig, "v3_years_to_50pct_ridgeline.png")
    tab = pd.DataFrame({
        "country": tr.meta["country"].values, "region": tr.meta["region"].values,
        "years_to_50pct": np.round(t50, 2),
        "cluster_dtw": [names_for(k, I2_NAMES)[i] for i in tr.lab_dtw],
        "cluster_euclid": [names_for(k, I2_NAMES)[i] for i in tr.lab_euc]}, index=tr.codes)
    g = tab.groupby("cluster_dtw")["years_to_50pct"].agg(["count", "mean", "std", "min", "median", "max"]).round(2)
    show(g, "T50 by DTW cluster")
    g2 = tab.groupby("cluster_euclid")["years_to_50pct"].agg(["count", "mean", "std", "min", "median", "max"]).round(2)
    show(g2, "T50 by Euclidean cluster")
    print(f"[V3] eta^2: DTW={eta_d:.3f} vs Euclid={eta_e:.3f}  (higher = clusters align with adoption timing)")
    save_csv(tab, "v3_years_to_50pct_by_country.csv")
    SUMMARY.update(eta2_dtw=round(eta_d, 4), eta2_euclid=round(eta_e, 4), n_never_crossed=n_cens)


# %% [7] V4 — DTW alignment teaching figure (same shape, ~8-year lag)
def pick_lag_pair(tr: SimpleNamespace, target=LAG_TARGET, tol=LAG_TOL):
    t, D, S = tr.t50, tr.D, tr.S
    for tol_i in (tol, 2.5, 4.0, 8.0):
        cand = [(i, j) for i in range(len(t)) for j in range(len(t))
                if not np.isnan(t[i]) and not np.isnan(t[j]) and abs((t[j] - t[i]) - target) <= tol_i]
        if len(cand) >= 5:
            break
    if not cand:
        raise RuntimeError("No country pair with a usable adoption lag.")
    d = np.array([D[i, j] for i, j in cand])
    cut = np.quantile(d, 0.40)                                    # similar shape under DTW ...
    pool = [(c, np.linalg.norm(S[c[0]] - S[c[1]]) / (D[c] + 1e-9)) for c, dd in zip(cand, d) if dd <= cut]
    (i, j), _ = max(pool, key=lambda x: x[1])                     # ... but Euclid much larger
    return i, j


def plot_dtw_alignment(tr: SimpleNamespace):
    i, j = pick_lag_pair(tr)
    a, b, yrs = tr.S[i], tr.S[j], tr.years
    d_dtw, path = dtw_path(a, b)
    d_euc = float(np.linalg.norm(a - b))
    n1, n2 = tr.meta["country"].iloc[i], tr.meta["country"].iloc[j]
    lag = tr.t50[j] - tr.t50[i]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax in axes:
        ax.plot(yrs, a * 100, color=PALETTE[0], lw=2.2, label=f"{n1} (early, T50≈{tr.t50[i]:.0f})", zorder=3)
        ax.plot(yrs, b * 100, color=PALETTE[1], lw=2.2, label=f"{n2} (late, T50≈{tr.t50[j]:.0f})", zorder=3)
        ax.set_xlabel("Year"); ax.axhline(50, color="0.7", lw=0.8, ls=":")
    axes[0].set_ylabel("Internet users (% of population)")
    for (p, q) in path:                                           # warping connectors
        axes[0].plot([yrs[p], yrs[q]], [a[p] * 100, b[q] * 100], color="0.55", lw=0.5, alpha=0.6, zorder=1)
    axes[0].set_title(f"DTW alignment (distance = {d_dtw:.3f})")
    for p in range(len(yrs)):                                     # time-locked comparison
        axes[1].plot([yrs[p]] * 2, [a[p] * 100, b[p] * 100], color="0.55", lw=0.6, alpha=0.6, zorder=1)
    axes[1].fill_between(yrs, a * 100, b * 100, color="0.85", alpha=0.6, zorder=0)
    axes[1].set_title(f"Euclidean / same-year comparison (distance = {d_euc:.3f})")
    axes[0].legend(loc="upper left", frameon=False)
    fig.suptitle(f"Same S-curve, ~{lag:.0f}-year lag: DTW sees a {d_euc/d_dtw:.1f}× smaller distance "
                 f"than Euclidean", y=1.02)
    fig.tight_layout()
    save_fig(fig, "v4_dtw_alignment_illustration.png")
    mean_shift = np.mean([q - p for p, q in path])
    out = pd.DataFrame([{"early": n1, "late": n2, "T50_early": round(tr.t50[i], 2), "T50_late": round(tr.t50[j], 2),
                         "lag_years": round(lag, 2), "dtw_dist": round(d_dtw, 4), "euclid_dist": round(d_euc, 4),
                         "euclid_over_dtw": round(d_euc / d_dtw, 2), "mean_path_shift_years": round(mean_shift, 2)}])
    show(out, "DTW teaching pair")
    save_csv(out, "v4_dtw_pair.csv", index=False)


# %% [8] V5 — Hierarchical dendrogram on DTW distances
def plot_dendrogram(tr: SimpleNamespace, k_cut=K, k_max=5):
    Z, n = tr.Z, len(tr.codes)
    h = Z[:, 2]
    cut_h = lambda k: (h[n - k - 1] + h[n - k]) / 2               # height separating k from k+1 clusters
    names = tr.meta["country"].tolist()
    reg = tr.meta["region"].tolist()
    regs = sorted(set(reg))
    rcol = {r: (REGION_COLORS.get(r) if REGION_COLORS else None) or plt.cm.tab10(i % 10) for i, r in enumerate(regs)}
    name2reg = dict(zip(names, reg))
    fig, ax = plt.subplots(figsize=(max(14, n * 0.12), 7.5))
    dendrogram(Z, labels=names, link_color_func=lambda _: "0.35",     # neutral branches: colour = region only
               leaf_rotation=90, leaf_font_size=5.5, ax=ax)
    for lbl in ax.get_xmajorticklabels():
        lbl.set_color(rcol[name2reg.get(lbl.get_text(), regs[0])])
    for k_, ls, al in [(k_cut, "-", 1.0)] + [(k, "--", 0.5) for k in range(k_cut + 1, k_max + 1)]:
        ax.axhline(cut_h(k_), color="black", ls=ls, lw=1.4 if k_ == k_cut else 0.9, alpha=al)
        ax.text(ax.get_xlim()[1], cut_h(k_), f" k={k_}", color="black", va="center", fontsize=9)
    ax.set_ylabel("Average-linkage DTW distance")
    ax.set_title(f"Hierarchical clustering on DTW distances (n={n}); solid line = cut at k={k_cut}, dashed = k={k_cut+1}–{k_max}")
    ax.legend(handles=[Patch(color=rcol[r], label=r) for r in regs], title="Region (leaf label colour)",
              loc="upper right", frameon=False, ncol=2, fontsize=8)
    save_fig(fig, "v5_dtw_dendrogram.png")

    # ---- does k=4/5 split meaningfully?
    cond = squareform(tr.D, checks=False)
    coph = cophenet(Z, cond)[0]
    rows, prev = [], None
    for k in range(2, 7):
        lab = fcluster(Z, k, "maxclust")
        sil = silhouette_score(tr.D, lab, metric="precomputed") if 1 < len(np.unique(lab)) < n else np.nan
        row = {"k": k, "silhouette_dtw": round(sil, 4), "sizes": sorted(np.bincount(lab)[1:].tolist(), reverse=True),
               "cut_height": round(cut_h(k), 4)}
        if prev is not None:
            ct = pd.crosstab(prev, lab)
            par = ct.index[(ct > 0).sum(axis=1) > 1][0]
            kids = sorted(ct.loc[par][ct.loc[par] > 0].values.tolist(), reverse=True)
            row["split_of_k-1"] = kids
            row["meaningful(min_child>=max(5,5%n))"] = bool(min(kids) >= max(5, 0.05 * n))
        rows.append(row); prev = lab
    tab = pd.DataFrame(rows)
    print(f"\n[V5] cophenetic correlation = {coph:.3f}")
    show(tab, "Dendrogram cut diagnostics (k=2..6)")
    save_csv(tab, "v5_dendrogram_cut_diagnostics.csv", index=False)
    assign = pd.DataFrame({"country": names, "region": reg, "dtw_cluster_k3": tr.lab_dtw}, index=tr.codes)
    reg_ct = pd.crosstab(assign["region"], assign["dtw_cluster_k3"].map(lambda i: names_for(K, I2_NAMES)[i]))
    show(reg_ct, "Region x DTW cluster (does geography explain the tree?)")
    save_csv(reg_ct, "v5_region_by_dtw_cluster.csv")
    SUMMARY["cophenetic_corr"] = round(coph, 4)


# %% [9] Anomalies — per-year IsolationForest on standardized core features
def yearly_anomalies(panel: pd.DataFrame, years=ANOM_YEARS) -> pd.DataFrame:
    out, skipped = [], []
    for y in years:
        sub = panel[panel["year"] == y].dropna(subset=FEATURES)       # complete-case, no imputation
        if len(sub) < ANOM_MIN_N:
            skipped.append((y, len(sub)))
            continue
        Xy = StandardScaler().fit_transform(sub[FEATURES])
        iso = IsolationForest(n_estimators=300, contamination=ANOM_CONTAM, random_state=SEED).fit(Xy)
        out.append(pd.DataFrame({"code": sub["code"].values, "country": sub["country"].values, "year": y,
                                 "score": -iso.score_samples(Xy), "flag": iso.predict(Xy) == -1}))
    print(f"[anomaly] scanned {[int(o['year'].iloc[0]) for o in out]}; skipped (n complete < {ANOM_MIN_N}): {skipped}")
    SUMMARY["anomaly_years"] = f"{out[0]['year'].iloc[0]}-{out[-1]['year'].iloc[0]}"
    return pd.concat(out, ignore_index=True)


def _longest_run(flags: np.ndarray) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def anomaly_stats(an: pd.DataFrame) -> pd.DataFrame:
    flags = an.pivot(index="code", columns="year", values="flag")
    score = an.pivot(index="code", columns="year", values="score")
    st = pd.DataFrame({
        "country": an.drop_duplicates("code").set_index("code")["country"],
        "n_obs": flags.notna().sum(axis=1),
        "n_flag": flags.fillna(False).astype(bool).sum(axis=1),
        "longest_run": [_longest_run(flags.loc[c].fillna(False).astype(bool).values) for c in flags.index],
        "max_score": score.max(axis=1).round(3), "mean_score": score.mean(axis=1).round(3)})
    st["frac_flagged"] = (st["n_flag"] / st["n_obs"]).round(3)
    st["category"] = np.select(
        [st.n_flag == 0, (st.frac_flagged >= 0.6) & (st.n_obs >= 3) & (st.longest_run >= np.minimum(4, st.n_obs)),
         (st.n_flag <= 2) | (st.frac_flagged <= 0.3)],
        ["never flagged", "persistent (structural)", "transient (artefact?)"], default="recurrent")
    return st


# %% [10] V6 — Anomaly persistence strip
def plot_persistence_strip(an: pd.DataFrame, st: pd.DataFrame, top_n=TOP_ANOM_STRIP, n_trans=6):
    head = st[st.n_flag > 0].sort_values(["n_flag", "mean_score"], ascending=False).head(top_n - n_trans)
    trans = (st[(st.category == "transient (artefact?)") & (st.n_flag > 0) & ~st.index.isin(head.index)]
             .sort_values("max_score", ascending=False).head(n_trans))
    sel = pd.concat([head, trans])
    cat_rank = {"persistent (structural)": 0, "recurrent": 1, "transient (artefact?)": 2}
    sel = sel.assign(_r=sel.category.map(cat_rank)).sort_values(["_r", "frac_flagged", "mean_score"],
                                                                 ascending=[True, False, False])
    flags = an.pivot(index="code", columns="year", values="flag").loc[sel.index]
    years = flags.columns.values
    code_map = {"persistent (structural)": 4, "recurrent": 3, "transient (artefact?)": 2}
    M = np.zeros(flags.shape, dtype=int)                               # 0 = no data
    for r, c in enumerate(sel.index):
        for j, y in enumerate(years):
            v = flags.loc[c, y]
            M[r, j] = 0 if pd.isna(v) else (1 if not v else code_map[sel.loc[c, "category"]])
    cols = ["#d9d9d9", "#ffffff", "#E5B93B", "#E07B39", "#B2182B"]
    fig, ax = plt.subplots(figsize=(10, 0.34 * len(sel) + 2.2))
    ax.imshow(M, cmap=ListedColormap(cols), vmin=0, vmax=4, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(years))); ax.set_xticklabels(years, rotation=90)
    ax.set_yticks(np.arange(len(sel)))
    ax.set_yticklabels([f"{sel.loc[c,'country']}  ({sel.loc[c,'n_flag']}/{sel.loc[c,'n_obs']})" for c in sel.index])
    ax.set_xticks(np.arange(-.5, len(years)), minor=True); ax.set_yticks(np.arange(-.5, len(sel)), minor=True)
    ax.grid(which="minor", color="0.85", lw=0.6); ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(handles=[Patch(fc=cols[4], label="flagged — persistent (structural)"),
                       Patch(fc=cols[3], label="flagged — recurrent"),
                       Patch(fc=cols[2], label="flagged — transient (artefact?)"),
                       Patch(fc=cols[1], ec="0.6", label="observed, not flagged"),
                       Patch(fc=cols[0], label="no complete-case data")],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=8)
    ax.set_title(f"Anomaly persistence: IsolationForest per year ({int(years.min())}–{int(years.max())}, "
                 f"top {int((100*ANOM_CONTAM))}% flagged)\nrows = flagged countries (flagged/observed years)")
    save_fig(fig, "v6_anomaly_persistence_strip.png")
    show(sel.drop(columns="_r"), "Anomaly persistence table (rows shown in strip)", 40)
    save_csv(st.sort_values(["n_flag", "mean_score"], ascending=False), "v6_anomaly_persistence_all.csv")
    cat = st.category.value_counts()    # includes "never flagged" so the denominator is explicit
    print("\n[V6] category counts (all countries scanned):", cat.to_dict())
    print("     -> persistent = candidates for structural features (e.g. multi-SIM), "
          "transient = candidates for T1 cleaning review.")


# %% [11] V7 — Anomaly "why" bullet chart (diverging z-scores vs cluster mean)
def plot_anomaly_why(snap, an, st, top=5):
    cur = an[an.year == snap.year].set_index("code")
    top_codes = cur.sort_values("score", ascending=False).head(top).index.tolist()
    Z = pd.DataFrame(snap.X, index=snap.codes, columns=FEATURES)       # global z-scores
    cl = snap.df["cluster_iter1"]
    cl_mean = Z.groupby(cl).mean()
    rows = []
    fig, axes = plt.subplots(top, 1, figsize=(9, 1.55 * top + 1), sharex=True)
    lim = 0
    diffs = {}
    for c in top_codes:
        d = Z.loc[c] - cl_mean.loc[cl.loc[c]]
        diffs[c] = d
        lim = max(lim, np.abs(d).max())
    lim = np.ceil(lim * 2) / 2 + 0.3
    for ax, c in zip(np.atleast_1d(axes), top_codes):
        d = diffs[c]
        ax.barh(range(len(FEATURES)), d.values, color=["#B2182B" if v > 0 else "#2166AC" for v in d.values], height=0.6)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(range(len(FEATURES))); ax.set_yticklabels([f.replace("_", " ") for f in FEATURES], fontsize=7.5)
        ax.invert_yaxis(); ax.set_xlim(-lim, lim)
        for i, v in enumerate(d.values):
            ax.text(v + (0.05 if v >= 0 else -0.05), i, f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right", fontsize=7)
        cname = names_for(K, I1_NAMES)[int(cl.loc[c])]
        pers = f", flagged {st.loc[c,'n_flag']}/{st.loc[c,'n_obs']} yrs" if c in st.index else ""
        ax.set_title(f"{snap.df.loc[c,'country']} — {cname}, score {cur.loc[c,'score']:.2f}{pers}", fontsize=9, loc="left")
        for f in FEATURES:
            rows.append({"code": c, "country": snap.df.loc[c, "country"], "cluster": cname, "feature": f,
                         "raw_value": snap.df.loc[c, f], "z_global": round(Z.loc[c, f], 3),
                         "z_minus_cluster_mean": round(d[f], 3), "anomaly_score": round(cur.loc[c, "score"], 4)})
    np.atleast_1d(axes)[-1].set_xlabel("Feature z-score minus cluster-mean z-score (global SD units)")
    fig.suptitle(f"Why are they anomalous? Top {top} IsolationForest scores, {snap.year}", y=1.0)
    fig.tight_layout()
    save_fig(fig, "v7_anomaly_why_bullets.png")
    tab = pd.DataFrame(rows)
    show(tab.pivot(index="country", columns="feature", values="z_minus_cluster_mean").round(2),
         "Top anomalies: feature deviation from own cluster mean (z units)")
    save_csv(tab, "v7_anomaly_why.csv", index=False)


# %% [12] V8 — Side-by-side embeddings (Iteration 1 vs Iteration 2 clusters)
def embed_2d(X, seed=SEED):
    n = len(X)
    try:
        import umap
        return umap.UMAP(n_neighbors=min(15, n - 1), min_dist=0.25, random_state=seed).fit_transform(X), "UMAP"
    except Exception:
        pass
    try:
        return TSNE(2, perplexity=max(5, min(30, (n - 1) / 3)), init="pca", random_state=seed,
                    learning_rate="auto").fit_transform(X), "t-SNE"
    except Exception:
        return PCA(2, random_state=seed).fit_transform(X), "PCA"


def plot_embeddings(snap, tr, sil_table, label_n=18):
    common = [c for c in snap.codes if c in set(tr.codes)]
    i1 = pd.Series(snap.labels, index=snap.codes).loc[common].values
    i2 = pd.Series(tr.lab_dtw, index=tr.codes).loc[common].values
    # Hungarian match: re-colour Iteration-2 clusters so they align with Iteration-1 colours
    cm = np.zeros((K, K), dtype=int)
    for a, b in zip(i1, i2):
        cm[a, b] += 1
    r, c = linear_sum_assignment(-cm)
    m2to1 = {cj: ri for ri, cj in zip(r, c)}
    i2_al = np.array([m2to1[v] for v in i2])
    mover = i1 != i2_al
    if EMBED_SPACE == "dtw":
        pos = {code: p for p, code in enumerate(tr.codes)}
        idx = [pos[x] for x in common]
        Dsub = tr.D[np.ix_(idx, idx)]
        xy, method = MDS(2, dissimilarity="precomputed", random_state=SEED, n_init=4).fit_transform(Dsub), "MDS (DTW)"
    else:
        Xc = pd.DataFrame(snap.X, index=snap.codes).loc[common].values
        xy, method = embed_2d(Xc)
    names = snap.df.loc[common, "country"].values

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), sharex=True, sharey=True)
    n1, n2 = names_for(K, I1_NAMES), names_for(K, I2_NAMES)
    inv = {v: kk for kk, v in m2to1.items()}                      # aligned colour -> native DTW cluster
    n2_by_colour = [n2[inv[c]] for c in range(K)]                 # legend text must follow the colour
    for ax, lab, title, nm in [(axes[0], i1, "Iteration 1: snapshot K-means", n1),
                               (axes[1], i2_al, "Iteration 2: DTW trajectory clusters", n2_by_colour)]:
        for c in range(K):
            m = lab == c
            ax.scatter(xy[m, 0], xy[m, 1], s=34, color=PALETTE[c], alpha=0.85, edgecolor="white", lw=0.4, label=nm[c])
        ax.scatter(xy[mover, 0], xy[mover, 1], s=110, facecolors="none", edgecolors="k", lw=1.4, zorder=5)
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
        ax.legend(frameon=False, loc="best", fontsize=8)
    # label the most borderline movers (lowest Iteration-1 silhouette)
    sil = sil_table.loc[common, f"sil_k{K}"].values
    for ii in [i for i in np.argsort(sil) if mover[i]][:label_n]:
        axes[1].annotate(names[ii], xy[ii], xytext=(4, 4), textcoords="offset points", fontsize=6.5)
    fig.suptitle(f"{method} on {'standardized snapshot features' if EMBED_SPACE!='dtw' else 'DTW distances'} "
                 f"(n={len(common)}) — ringed = changed cluster ({mover.sum()}), colours matched by Hungarian assignment",
                 y=1.0)
    fig.tight_layout()
    save_fig(fig, "v8_embedding_iter1_vs_iter2.png")
    ct = pd.crosstab(pd.Series([n1[v] for v in i1], name="Iter1"), pd.Series([n2[v] for v in i2], name="Iter2"))
    show(ct, "Crosstab Iteration-1 x Iteration-2 (Sankey source table)")
    ari = adjusted_rand_score(i1, i2)
    print(f"[V8] ARI(Iter1, Iter2) = {ari:.3f}; movers = {mover.sum()}/{len(common)} ({mover.mean():.1%}); "
          f"embedding = {method}")
    movers = pd.DataFrame({"country": names, "iter1": [n1[v] for v in i1],
                           "iter2_aligned": [n1[v] for v in i2_al], "iter2_native": [n2[v] for v in i2],
                           "changed": mover, "sil_iter1": sil.round(3)}, index=common)
    show(movers[movers.changed].sort_values("sil_iter1"), "Countries that changed cluster", 40)
    save_csv(movers, "v8_cluster_movers.csv"); save_csv(ct, "v8_iter1_iter2_crosstab.csv")
    SUMMARY.update(ari_iter1_iter2=round(ari, 4), n_movers=int(mover.sum()), n_common=len(common),
                   embedding_method=method)


# %% [13] Run everything
def main(path: str | None = None):
    seed_everything()
    panel = load_panel(path)
    snap = build_snapshot(panel)
    fit_iter1(snap)

    # V1 / V2 (cross-sectional)
    co, aris = bootstrap_coassignment(snap.X, snap.labels)
    plot_coassignment(snap, co, aris)
    sil_table = plot_silhouette_knife(snap)

    # trajectories + DTW (V3 / V4 / V5)
    tr = build_trajectories(panel)
    cluster_trajectories(tr)
    plot_ridgeline(tr)
    plot_dtw_alignment(tr)
    plot_dendrogram(tr)

    # anomalies (V6 / V7)
    an = yearly_anomalies(panel)
    st = anomaly_stats(an)
    plot_persistence_strip(an, st)
    plot_anomaly_why(snap, an, st)

    # V8
    plot_embeddings(snap, tr, sil_table)

    save_csv(pd.Series(SUMMARY, name="value").to_frame(), "run_summary.csv")
    show(pd.Series(SUMMARY, name="value").to_frame(), "RUN SUMMARY (viva cheat-sheet)", 40)


if __name__ == "__main__":
    main()   # pass an explicit path, e.g. main("/kaggle/input/<your-dataset>/master_dataset.csv")

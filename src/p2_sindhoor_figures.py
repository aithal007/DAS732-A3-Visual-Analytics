"""
p2_sindhoor_figures.py  -  publication figures for the unsupervised part (V4, V5, V6, V8),
drawn in the shared team style from the tables written by p2_sindhoor_unsupervised.py.
(The pipeline's own quick-look plots go to outputs/diagnostics/.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from common import (ACCENT, ACCENT2, BAD, INK, MUTED, OUT, PROC, REGION_COLORS, REGION_ORDER, SEED,
                    savefig)
import p2_sindhoor_unsupervised as p2
from common import set_style

set_style()   # p2 sets a seaborn theme on import; restore the shared team style

CORE = p2.CORE
LAB = {"internet_users_share_pct": "Internet %", "mobile_subs_per100": "Mobile /100",
       "mobile_phone_ownership_share_pct": "Phone owners %", "telephone_lines_per100": "Landlines /100",
       "broadband_subs_per100": "Broadband /100"}
CL_COL = ["#E69F00", "#56B4E9", "#009E73"]            # archetype colours: low -> high connectivity
CL_NAME = {0: "C0 mobile-first, low", 1: "C1 mixed", 2: "C2 fixed-line legacy"}
TR_NAME = {0: "T0 late starters", 1: "T1 S-curve majority", 2: "T2 early saturators"}
TAG_COL = {"multi-SIM inflation": "#CC79A7", "extreme": BAD, "fast climber": "#009E73",
           "laggard": ACCENT2, "high landline / low internet": "#0072B2", "multivariate outlier": MUTED}


def _tagcol(reason: str) -> str:
    first = reason.split(" + ")[0]
    return TAG_COL["extreme"] if first.startswith("extreme") else TAG_COL.get(first, MUTED)


def load():
    a = pd.read_csv(OUT / "iter1_assignments.csv").set_index("code")
    X = StandardScaler().fit_transform(a[CORE].values)
    pca = PCA(random_state=SEED).fit(X)
    Z = pca.transform(X)
    val = pd.read_csv(OUT / "iter1_cluster_validation.csv").set_index("k")
    return a, X, pca, Z, val


# ---------------------------------------------------------------- V4
def fig_v4(a, pca, Z):
    ev = pca.explained_variance_ratio_
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axs[0]
    for r in REGION_ORDER:
        m = (a.region == r).values
        ax.scatter(Z[m, 0], Z[m, 1], s=34, color=REGION_COLORS[r], alpha=0.85, edgecolor="white", lw=0.4,
                   label=f"{r} ({m.sum()})")
    load = pca.components_[:2].T * np.sqrt(pca.explained_variance_[:2]) * 2.6
    tips = sorted(zip(load, CORE), key=lambda t: -t[0][1])
    ys_lab = [2.6, 1.55, 1.05, 0.55, 0.05]
    for ((lx, ly), f), yl in zip(tips, ys_lab):
        ax.annotate("", xy=(lx, ly), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
        xl = 2.75 if f != "mobile_subs_per100" else lx + 0.15
        ax.annotate(LAB[f], xy=(lx, ly), xytext=(xl, yl), fontsize=8.5, weight="bold", color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5),
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    for nm, dxy in [("Fiji", (10, -4)), ("Hong Kong", (8, 4)), ("Monaco", (-50, 8)), ("Chad", (-10, -14)),
                    ("Kiribati", (6, -12))]:
        i = np.where(a.name.values == nm)[0]
        if len(i):
            ax.annotate(nm, (Z[i[0], 0], Z[i[0], 1]), xytext=dxy, textcoords="offset points", fontsize=8.5,
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.axhline(0, color=MUTED, lw=0.6); ax.axvline(0, color=MUTED, lw=0.6)
    ax.set(xlabel=f"PC1 ({ev[0]:.1%}): overall connectivity", ylabel=f"PC2 ({ev[1]:.1%}): mobile subscriptions",
           title="(a) Biplot, 2024 snapshot (n = 135, z-scored)")
    ax.legend(loc="upper left", fontsize=8)
    ax = axs[1]
    ld = pd.DataFrame(pca.components_[:3].T, index=[LAB[c] for c in CORE], columns=["PC1", "PC2", "PC3"])
    im = ax.imshow(ld.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(3), [f"PC{i+1}\n{ev[i]:.0%}" for i in range(3)])
    ax.set_yticks(range(5), ld.index)
    ax.grid(False)
    for i in range(5):
        for j in range(3):
            ax.text(j, i, f"{ld.values[i, j]:+.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(ld.values[i, j]) > 0.6 else INK)
    fig.colorbar(im, ax=ax, fraction=0.05, label="loading")
    ax.set_title("(b) Loadings: PC2 is mobile alone (0.96)")
    fig.suptitle("V4  Two axes describe 82% of 2024 connectivity: a general one, and a mobile one that Fiji and "
                 "Hong Kong stretch on their own", x=0.01, ha="left", fontsize=12.5, weight="bold")
    fig.tight_layout()
    savefig(fig, "V4_pca_biplot.png")


# ---------------------------------------------------------------- V5
def fig_v5(a, X, val):
    k = 3
    labs = {"kmeans": a.iter1_kmeans.values, "gmm": a.iter1_gmm.values, "ward": a.iter1_ward.values}
    boot = p2.bootstrap_ari(X, k, labs, n_boot=200)
    fig = plt.figure(figsize=(16, 8.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1])
    ax = fig.add_subplot(gs[0, 0])
    for c, lab, col in [("sil_kmeans", "k-means", ACCENT), ("sil_gmm", "GMM", ACCENT2), ("sil_ward", "Ward", MUTED)]:
        ax.plot(val.index, val[c], "o-", color=col, label=lab, lw=1.8)
    ax.axvspan(2.5, 6.5, color=ACCENT, alpha=0.06)
    ax.axvline(3, color=BAD, ls="--", lw=1)
    ax.annotate("Ward k=3: 0.47, but one\ncluster = Fiji + Hong Kong", (3, val.loc[3, "sil_ward"]),
                xytext=(4.6, 0.43), fontsize=8, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.set(xlabel="k", ylabel="mean silhouette", title="(a) Silhouette: k=3 best in the eligible range 3-6")
    ax.legend(fontsize=8, loc="lower left")
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(val.index, val.gmm_bic, "o-", color=BAD, label="GMM BIC")
    ax.plot(val.index, val.gmm_aic, "s--", color=MUTED, label="GMM AIC")
    ax2 = ax.twinx(); ax2.plot(val.index, val.inertia, "^-", color=ACCENT, label="k-means inertia")
    ax2.grid(False); ax2.set_ylabel("inertia", color=ACCENT)
    ax.axvline(3, color=BAD, ls="--", lw=1)
    ax.set(xlabel="k", ylabel="information criterion", title="(b) BIC minimum at k=5; elbow near k=3-4")
    ax.legend(fontsize=8, loc="upper center")
    ax = fig.add_subplot(gs[0, 2])
    bp = ax.boxplot([boot[m] for m in ["kmeans", "gmm", "ward"]], patch_artist=True, widths=0.5, showfliers=False)
    for b, col in zip(bp["boxes"], [ACCENT, ACCENT2, MUTED]):
        b.set(facecolor=col + "55", edgecolor=col)
    for m in bp["medians"]:
        m.set(color=INK)
    ax.set_xticks([1, 2, 3], ["k-means", "GMM", "Ward"])
    for i, m in enumerate(["kmeans", "gmm", "ward"]):
        ax.text(i + 1, 0.04, f"mean {boot[m].mean():.2f}", ha="center", fontsize=8)
    ax.set(ylim=(0, 1.02), ylabel="ARI vs full-data labels", title="(c) Bootstrap stability (B=200): moderate")
    ax = fig.add_subplot(gs[1, :])
    Xs = pd.DataFrame(X, columns=CORE)
    xs = np.arange(len(CORE))
    for c in range(3):
        m = a.iter1_kmeans.values == c
        q1, med, q3 = np.percentile(Xs[m].values, [25, 50, 75], axis=0)
        for row in Xs[m].values:
            ax.plot(xs, row, color=CL_COL[c], alpha=0.07, lw=0.8)
        ax.fill_between(xs, q1, q3, color=CL_COL[c], alpha=0.25)
        ax.plot(xs, med, "o-", color=CL_COL[c], lw=2.8, label=f"{CL_NAME[c]} (n={m.sum()})")
    raw_med = a.groupby("iter1_kmeans")[CORE].median()
    for j, col in enumerate(CORE):
        ax.text(j, -2.95, " / ".join(f"{raw_med.loc[c, col]:.0f}" for c in range(3)), ha="center", fontsize=8.5,
                color=INK)
    ax.text(-0.45, -2.95, "raw medians\nC0/C1/C2:", fontsize=8, color=MUTED, va="center")
    ax.set_xticks(xs, [LAB[c] for c in CORE])
    ax.set(ylim=(-3.3, 3.3), ylabel="z-score", title="(d) k-means profiles (median, IQR band, all countries faint): "
           "the archetypes differ most on landlines and broadband, least on mobile")
    ax.legend(loc="upper left", fontsize=8.5, ncol=3)
    fig.suptitle("V5  Three 2024 archetypes ordered by connectivity, as in A1 Fig. 23, but only moderately stable",
                 x=0.01, ha="left", fontsize=12.5, weight="bold")
    fig.tight_layout()
    savefig(fig, "V5_cluster_validation_profiles.png")
    return boot


# ---------------------------------------------------------------- V6
def fig_v6(a, Z):
    top = a.sort_values("anomaly_rank").head(15).iloc[::-1]
    fig, axs = plt.subplots(1, 2, figsize=(16, 6.4), gridspec_kw={"width_ratios": [1, 1.05]})
    ax = axs[0]
    y = np.arange(len(top))
    cols = [_tagcol(r) for r in top.reason]
    ax.hlines(y, 0.5, top.anomaly_combined, color=cols, lw=2.2)
    ax.scatter(top.anomaly_combined, y, s=90, color=cols, edgecolor=INK, zorder=3)
    ax.scatter(top.if_rank_pct, y, marker="^", s=30, color=INK, zorder=4)
    ax.scatter(top.lof_rank_pct, y, marker="s", s=28, facecolor="white", edgecolor=INK, zorder=4)
    ax.set_yticks(y, [f"{n}  [{r}]" for n, r in zip(top.name, top.reason)], fontsize=8.3)
    ax.set(xlim=(0.5, 1.02), xlabel="anomaly percentile (rank-average of Isolation Forest and LOF)",
           title="(a) Top-15 anomalies with rule-based reason tags")
    h = [Line2D([], [], marker="^", ls="", color=INK, label="Isolation Forest"),
         Line2D([], [], marker="s", ls="", mfc="white", mec=INK, label="LOF")]
    ax.legend(handles=h, loc="lower right", fontsize=8)
    ax = axs[1]
    ax.scatter(Z[:, 0], Z[:, 1], s=16, color="#D0D7DE")
    idx = [a.index.get_loc(c) for c in top.index]
    sz = 40 + 400 * (top.anomaly_combined.values - 0.85).clip(0) / 0.15
    ax.scatter(Z[idx, 0], Z[idx, 1], s=sz, color=cols, edgecolor=INK, alpha=0.9, zorder=3)
    for c, (dx, dy) in {"FJI": (10, -6), "HKG": (8, 6), "MCO": (-46, 10), "KIR": (8, -12), "AND": (8, 4),
                        "UGA": (-10, -14), "MUS": (8, 6)}.items():
        if c in a.index:
            i = a.index.get_loc(c)
            ax.annotate(a.name.iloc[i], (Z[i, 0], Z[i, 1]), xytext=(dx, dy), textcoords="offset points",
                        fontsize=8.5, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    h = [Line2D([], [], marker="o", ls="", color=c, label=t) for t, c in TAG_COL.items()
         if t in {"multi-SIM inflation", "extreme", "fast climber", "multivariate outlier"}]
    ax.legend(handles=h, loc="upper right", fontsize=8, title="primary tag")
    ax.set(xlabel="PC1 (connectivity)", ylabel="PC2 (mobile)", title="(b) Same anomalies in PCA space (size = score)")
    agree = 7
    fig.suptitle(f"V6  Both detectors rank Fiji first; most anomalies are multi-SIM markets "
                 f"(IF and LOF share {agree} of their top 15, Spearman 0.77)", x=0.01, ha="left", fontsize=12.5,
                 weight="bold")
    fig.tight_layout()
    savefig(fig, "V6_anomalies.png")


# ---------------------------------------------------------------- V8
def fig_v8():
    clean = pd.read_parquet(PROC / "panel_iteration2.parquet")
    traj = p2.build_trajectories(clean)
    t2 = pd.read_csv(OUT / "iter2_assignments.csv").set_index("code")
    summ = pd.read_csv(OUT / "iter2_cluster_summary.csv").set_index("traj_cluster")
    cmp_ = pd.read_csv(OUT / "iter2_comparison.csv", index_col=0)
    ct = pd.read_csv(OUT / "iter2_crosstab.csv", index_col=0)
    regmap = clean.drop_duplicates("code").set_index("code").region
    fig = plt.figure(figsize=(17, 9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1])
    yrs = traj.columns.astype(int)
    for c in range(3):
        ax = fig.add_subplot(gs[0, c])
        codes = t2.index[t2.traj_cluster == c].intersection(traj.index)
        M = traj.loc[codes].values
        for row in M:
            ax.plot(yrs, row, color=CL_COL[c], alpha=0.15, lw=0.8)
        q1, med, q3 = np.percentile(M, [25, 50, 75], axis=0)
        ax.fill_between(yrs, q1, q3, color=CL_COL[c], alpha=0.3)
        ax.plot(yrs, med, color=INK, lw=2.4, label="median")
        mname = summ.loc[c, "medoid"]
        mc = clean.loc[clean.entity == mname, "code"].iloc[0]
        if mc in traj.index:
            ax.plot(yrs, traj.loc[mc].values, color=BAD, lw=1.8, ls="--", label=f"medoid: {mname}")
        ax.set(ylim=(0, 100), xlim=(2000, 2025), title=f"{TR_NAME[c]} (n={len(codes)})",
               xlabel="year", ylabel="internet users (%)" if c == 0 else None)
        ax.text(2001, 92, f"median 2000 → 2025:\n{summ.loc[c, 'median_2000']:.0f}% → {summ.loc[c, 'median_2025']:.0f}%",
                fontsize=8.5, va="top")
        ax.legend(loc="lower right", fontsize=8)
    ax = fig.add_subplot(gs[0, 3])
    comp = pd.crosstab(t2.traj_cluster.map(TR_NAME), t2.index.map(regmap))
    comp = comp.reindex(columns=[r for r in REGION_ORDER if r in comp.columns])
    share = comp / comp.sum(axis=0)
    bottom = np.zeros(share.shape[1])
    for c in range(3):
        v = share.loc[TR_NAME[c]].values
        ax.bar(range(share.shape[1]), v * 100, bottom=bottom, color=CL_COL[c], label=TR_NAME[c])
        bottom += v * 100
    ax.set_xticks(range(share.shape[1]), [r.replace(" ", "\n") for r in share.columns], fontsize=8)
    ax.set(ylabel="% of region's countries", title="Archetype mix by region")
    ax.legend(fontsize=7.5, loc="lower right")
    ax = fig.add_subplot(gs[1, :2])
    ctv = ct.values
    im = ax.imshow(ctv, cmap="Blues", aspect="auto")
    ax.grid(False)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, ctv[i, j], ha="center", va="center", fontsize=13, weight="bold",
                    color="white" if ctv[i, j] > 30 else INK)
    ax.set_xticks(range(3), [TR_NAME[j] for j in range(3)])
    ax.set_yticks(range(3), [CL_NAME[i] for i in range(3)])
    ax.set(xlabel="iteration 2: trajectory archetype (2000-25, DTW)", ylabel="iteration 1: 2024 snapshot cluster",
           title=f"Snapshot vs trajectory (n={ctv.sum()}): {ctv.sum() - np.trace(ctv)} countries change group, "
                 f"ARI {cmp_.iloc[0]['ARI_iter1_vs_iter2']:.2f}")
    ax = fig.add_subplot(gs[1, 2:])
    xs = np.arange(2)
    v1 = cmp_.iloc[0, :2].values
    v2 = cmp_.iloc[1, :2].values
    ax.bar(xs - 0.18, v1, 0.36, color=MUTED, label="iteration-1 snapshot labels")
    ax.bar(xs + 0.18, v2, 0.36, color=ACCENT, label="iteration-2 trajectory labels")
    for x, a_, b_ in zip(xs, v1, v2):
        ax.text(x - 0.18, a_ + 0.01, f"{a_:.2f}", ha="center", fontsize=9)
        ax.text(x + 0.18, b_ + 0.01, f"{b_:.2f}", ha="center", fontsize=9)
    ax.set_xticks(xs, ["scored in 2024 feature space", "scored in trajectory (DTW) space"])
    ax.set(ylim=(0, 0.62), ylabel="mean silhouette", title="Each labelling is coherent only in its own space")
    ax.legend(fontsize=8.5, loc="upper left")
    fig.suptitle("V8  Iteration 2 groups countries by PATH: three S-curve archetypes that cut across the 2024 "
                 "snapshot clusters", x=0.01, ha="left", fontsize=12.5, weight="bold")
    fig.tight_layout()
    savefig(fig, "V8_trajectory_archetypes.png")
    return comp


# ---------------------------------------------------------------- V8b
def fig_v8b_dendrogram_tsne():
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform
    from sklearn.manifold import TSNE
    clean = pd.read_parquet(PROC / "panel_iteration2.parquet")
    traj = p2.build_trajectories(clean)
    cache = OUT / "dtw_distance_matrix.npy"
    if cache.exists():
        D = np.load(cache)
    else:
        D, _ = p2.dtw_distance_matrix(traj.values)
        np.save(cache, D)
    t2 = pd.read_csv(OUT / "iter2_assignments.csv").set_index("code").reindex(traj.index)
    lab = t2.traj_cluster.values.astype(int)
    names = clean.drop_duplicates("code").set_index("code").entity.reindex(traj.index)
    Z = linkage(squareform(D, checks=False), method="average")
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1])
    ax = fig.add_subplot(gs[0, 0])
    cut = Z[-3, 2] + 1e-6
    from scipy.cluster.hierarchy import set_link_color_palette
    set_link_color_palette(CL_COL)          # branches coloured left-to-right = T0, T1, T2
    dn = dendrogram(Z, ax=ax, no_labels=True, color_threshold=cut, above_threshold_color=MUTED)
    set_link_color_palette(None)
    ax.axhline(cut, color=BAD, ls="--", lw=1.2)
    ax.text(5, cut * 1.03, "cut for k = 3", color=BAD, fontsize=9)
    leaves = dn["leaves"]
    for i, leaf in enumerate(leaves):
        ax.add_patch(plt.Rectangle((i * 10, -cut * 0.06), 10, cut * 0.04, color=CL_COL[lab[leaf]], lw=0))
    ax.set_ylim(-cut * 0.08, None)
    ax.grid(False)
    ax.set(ylabel="DTW distance (average linkage)", title="(a) Dendrogram of 181 trajectories; colour strip = archetype")
    ax.set_xticks([])
    ax = fig.add_subplot(gs[0, 1])
    emb = TSNE(n_components=2, metric="precomputed", init="random", perplexity=25, random_state=SEED).fit_transform(D)
    last = traj.iloc[:, -1].values
    pd.DataFrame({"code": traj.index, "entity": names.values, "tsne_x": emb[:, 0], "tsne_y": emb[:, 1],
                  "traj_cluster": lab, "internet_2025": last}).to_csv(OUT / "iter2_tsne_embedding.csv", index=False)
    for c in range(3):
        m = lab == c
        ax.scatter(emb[m, 0], emb[m, 1], s=20 + last[m] * 0.8, color=CL_COL[c], alpha=0.85, edgecolor="white",
                   lw=0.5, label=f"{TR_NAME[c]} (n={m.sum()})")
    for nm in ["India", "Nigeria", "South Korea", "Germany", "Brazil", "Tanzania", "Maldives", "Kiribati", "China",
               "United States", "Ethiopia", "Indonesia"]:
        idx = np.where(names.values == nm)[0]
        if len(idx):
            ax.annotate(nm, emb[idx[0]], xytext=(4, 4), textcoords="offset points", fontsize=8.5)
    ax.set(xticks=[], yticks=[], xlabel="t-SNE 1", ylabel="t-SNE 2",
           title="(b) t-SNE map of DTW distances (size = 2025 level):\narchetypes form a continuum ordered by timing")
    ax.legend(loc="best", fontsize=8.5)
    fig.suptitle("V8b  The trajectory archetypes are real structure in DTW space, and late starters sit at one end of "
                 "a single continuum", x=0.01, ha="left", fontsize=12.5, weight="bold")
    fig.tight_layout()
    savefig(fig, "../outputs/diagnostics/V8b_dendrogram_tsne.png")   # superseded in the report by V8b_dtw_dendrogram + V8d


def main():
    a, X, pca, Z, val = load()
    fig_v4(a, pca, Z)
    b = fig_v5(a, X, val); print(b.mean().round(3).to_dict())
    fig_v6(a, Z)
    print(fig_v8().to_string())
    fig_v8b_dendrogram_tsne()


if __name__ == "__main__":
    main()

"""
p2_sindhoor_evidence.py - Workflow W2 evidence figures (Sindhoor), run on the SHARED pipeline.

Sindhoor's diagnostic library (p2_sindhoor_diagnostics.py) draws eight evidence views for the
unsupervised workflow. This adapter feeds it the team's data instead of its own loader, so every
number agrees with the rest of the report:

  * iteration-1 snapshot  = the same 2024 complete-case table and k-means labels as V4-V6
                            (outputs/iter1_assignments.csv, A1 regions, raw data);
  * anomaly persistence   = per-year Isolation Forest on the RAW panel (iteration 1);
  * iteration-2 curves    = the same 2000-2025 trajectories, DTW matrix (Sakoe-Chiba radius 5)
                            and archetype labels as V8 (transformed panel, artefacts removed).

Figures (images/):
  V5b_bootstrap_coassignment.png   iteration 1 - how stable is each snapshot cluster?
  V5c_silhouette_knife.png         iteration 1 - which countries sit on a cluster border?
  V6b_anomaly_persistence.png      iteration 1 - persistent (structural) vs transient (artefact) anomalies
  V6c_anomaly_why.png              iteration 1 - which indicator makes each top anomaly unusual?
  V8a_dtw_alignment.png            transformation - why DTW: same S-curve, shifted in time
  V8b_dtw_dendrogram.png           iteration 2 - the DTW hierarchy and its k = 3 cut
  V8c_takeoff_ridgeline.png        iteration 2 - year of crossing 50 % by archetype
  V8d_iteration_embedding.png      iteration 2 - which countries changed group between iterations

    python src/p2_sindhoor_evidence.py
"""
from __future__ import annotations

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import p2_sindhoor_diagnostics as sv
import p2_sindhoor_unsupervised as p2
from common import IMG, OUT, PROC, RAW_INDICATORS, REGION_COLORS, SEED, load_panel, set_style

CL_COL = ["#E69F00", "#56B4E9", "#009E73"]          # same archetype colours as V5 and V8
I1 = ["C0 mobile-first, low", "C1 mixed", "C2 fixed-line legacy"]
I2 = ["T0 late starters", "T1 S-curve majority", "T2 early saturators"]

RENAME = {
    "v1_bootstrap_coassignment.png": "V5b_bootstrap_coassignment.png",
    "v2_silhouette_knife_k2_vs_k3.png": "V5c_silhouette_knife.png",
    "v6_anomaly_persistence_strip.png": "V6b_anomaly_persistence.png",
    "v7_anomaly_why_bullets.png": "V6c_anomaly_why.png",
    "v4_dtw_alignment_illustration.png": "V8a_dtw_alignment.png",
    "v5_dtw_dendrogram.png": "V8b_dtw_dendrogram.png",
    "v3_years_to_50pct_ridgeline.png": "V8c_takeoff_ridgeline.png",
    "v8_embedding_iter1_vs_iter2.png": "V8d_iteration_embedding.png",
}
_PENDING: dict = {}


def _hold(fig, name):            # keep the figure open until its finding title is known
    _PENDING[name] = fig


def _csv(df, name, index=True):
    df.to_csv(OUT / f"p2_{name}", index=index)


def _finish(name: str, finding: str, keep_subtitle: bool = True):
    """Finding as a bold title (wrapped to the figure width); the library's own title becomes a
    regular-weight subtitle underneath. Offsets are in points so tall and wide figures look alike."""
    import textwrap
    from matplotlib.transforms import offset_copy
    fig = _PENDING.pop(name)
    vid = RENAME[name].split("_")[0]
    old = fig._suptitle.get_text() if fig._suptitle is not None and keep_subtitle else ""
    if fig._suptitle is not None:
        fig._suptitle.set_text("")
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())        # inches, all drawn content
    top = bb.y1 / fig.get_size_inches()[1]
    width = int(fig.get_size_inches()[0] * 9.5)
    head = textwrap.fill(f"{vid}  {finding}", width)
    sub = textwrap.fill(old.replace("\n", " "), int(width * 1.15)) if old else ""
    y = 6
    if sub:
        fig.text(0.01, top, sub, ha="left", va="bottom", fontsize=10, color="#57606A",
                 transform=offset_copy(fig.transFigure, fig=fig, y=y, units="points"))
        y += 14 * (sub.count("\n") + 1) + 4
    fig.text(0.01, top, head, ha="left", va="bottom", fontsize=12.5, weight="bold", color="#1F2328",
             transform=offset_copy(fig.transFigure, fig=fig, y=y, units="points"))
    fig.savefig(IMG / RENAME[name], dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [img] {RENAME[name]}")


def configure():
    sv.IMG_DIR, sv.OUT_DIR = IMG, OUT
    sv.save_fig, sv.save_csv = _hold, _csv
    sv.FEATURES = list(RAW_INDICATORS)
    sv.K, sv.I1_NAMES, sv.I2_NAMES = 3, I1, I2
    sv.PALETTE = CL_COL + ["#C44E52", "#8172B3"]
    sv.TRAJ_START, sv.TRAJ_END = p2.TRAJ_Y0, p2.TRAJ_Y1
    sv.DTW_BAND = p2.DTW_BAND
    sv.ANOM_YEARS = list(range(2010, 2025))
    sv.REGION_COLORS = dict(REGION_COLORS)
    set_style()
    plt.rcParams.update({"axes.grid": False, "savefig.dpi": 220})


def build_snapshot() -> SimpleNamespace:
    a = pd.read_csv(OUT / "iter1_assignments.csv").set_index("code")
    df = a.rename(columns={"name": "country"})[["country", "region"] + RAW_INDICATORS].copy()
    df["cluster_iter1"] = a["iter1_kmeans"].astype(int)
    X = StandardScaler().fit_transform(df[RAW_INDICATORS])
    return SimpleNamespace(df=df, X=X, year=2024, codes=df.index.values, labels=df["cluster_iter1"].values)


def build_trajectories() -> SimpleNamespace:
    clean = pd.read_parquet(PROC / "panel_iteration2.parquet")
    traj = p2.build_trajectories(clean)                      # identical to V8 (2000-2025, n = 181)
    D = np.load(OUT / "dtw_distance_matrix.npy")
    assert D.shape[0] == len(traj), "cached DTW matrix does not match the trajectory panel"
    wide = clean.pivot_table(index="code", columns="year", values="internet_users_share_pct")
    t50 = pd.Series({c: sv.first_year_crossing(wide.columns.values, wide.loc[c].values) for c in wide.index})
    meta = clean.drop_duplicates("code").set_index("code")[["entity", "region"]].rename(columns={"entity": "country"})
    lab = pd.read_csv(OUT / "iter2_assignments.csv").set_index("code").traj_cluster.reindex(traj.index)
    S = traj.values / 100.0
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    Z = linkage(squareform(D, checks=False), method="average")
    euc = KMeans(3, n_init=50, random_state=SEED).fit_predict(S)
    lab_euc = p2.relabel_by(euc, traj.iloc[:, -1].values)   # same ordering rule as V8: by 2025 level
    return SimpleNamespace(W=traj, S=S, D=D / 100.0, codes=traj.index.values, years=np.array(traj.columns),
                           t50_all=t50, t50=t50.reindex(traj.index).values, meta=meta.reindex(traj.index),
                           Z=Z, lab_dtw=lab.values.astype(int), lab_euc=np.asarray(lab_euc).astype(int))


def main():
    configure()
    np.random.seed(SEED)
    snap = build_snapshot()
    raw = load_panel(verbose=False).rename(columns={"entity": "country"})

    # ---------------- iteration 1: stability of the snapshot clusters
    co, aris = sv.bootstrap_coassignment(snap.X, snap.labels)
    sv.plot_coassignment(snap, co, aris)
    S = sv.SUMMARY
    blk = pd.read_csv(OUT / "p2_v1_block_coassignment.csv", index_col=0)
    _finish("v1_bootstrap_coassignment.png",
            f"Snapshot clusters are only moderately stable: mean bootstrap ARI {S['boot_ari_mean']:.2f}, and "
            f"mixed and legacy countries share a cluster in {blk.iloc[1, 2]:.0%} of resamples")
    sil = sv.plot_silhouette_knife(snap)
    n_border = int((sil["sil_k3"] < 0.10).sum())
    _finish("v2_silhouette_knife_k2_vs_k3.png",
            f"k = 3 has no misassigned country, but {n_border} of {len(sil)} sit on a border (silhouette < 0.10)")

    # ---------------- iteration 1: anomalies over time
    an = sv.yearly_anomalies(raw)
    st = sv.anomaly_stats(an)
    sv.plot_persistence_strip(an, st)
    cat = st.category.value_counts()
    _finish("v6_anomaly_persistence_strip.png",
            f"{cat.get('persistent (structural)', 0)} anomalies persist every year (structural), "
            f"{cat.get('transient (artefact?)', 0)} appear only briefly (artefact candidates, e.g. Fiji)")
    sv.plot_anomaly_why(snap, an, st)
    _finish("v7_anomaly_why_bullets.png",
            "Each top anomaly is unusual on one indicator: Fiji and Hong Kong on mobile subscriptions, Monaco and Andorra on landlines")

    # ---------------- transformation + iteration 2: trajectories under DTW
    tr = build_trajectories()
    sv.plot_dtw_alignment(tr)
    pair = pd.read_csv(OUT / "p2_v4_dtw_pair.csv").iloc[0]
    _finish("v4_dtw_alignment_illustration.png",
            f"Why the transformation uses DTW: {pair.early} and {pair.late} follow one curve "
            f"{pair.lag_years:.0f} years apart, so DTW scores them {pair.euclid_over_dtw:.0f} times closer "
            f"than the same-year (Euclidean) distance", keep_subtitle=False)
    sv.plot_dendrogram(tr)
    _finish("v5_dtw_dendrogram.png",
            f"Late starters split from all others at the root; cophenetic correlation {sv.SUMMARY['cophenetic_corr']:.2f}")
    sv.plot_ridgeline(tr)
    t = pd.read_csv(OUT / "p2_v3_years_to_50pct_by_country.csv").groupby("cluster_dtw").years_to_50pct.median()
    _finish("v3_years_to_50pct_ridgeline.png",
            f"The archetypes are ordered in time (median 50 % crossing {t.iloc[2]:.0f}, {t.iloc[1]:.0f}, "
            f"{t.iloc[0]:.0f}), but DTW groups by shape: the majority spans 17 years, so timing alone "
            f"(Euclidean η² {sv.SUMMARY['eta2_euclid']:.2f} vs DTW {sv.SUMMARY['eta2_dtw']:.2f}) separates dates better")
    sv.plot_embeddings(snap, tr, sil)
    S = sv.SUMMARY
    mv = pd.read_csv(OUT / "p2_v8_cluster_movers.csv").groupby("changed").sil_iter1.median()
    _finish("v8_embedding_iter1_vs_iter2.png",
            f"The feedback loop regroups {S['n_movers']} of {S['n_common']} countries "
            f"(ARI {S['ari_iter1_iter2']:.2f}); movers had weaker iteration-1 membership (median silhouette "
            f"{mv.loc[True]:.2f} vs {mv.loc[False]:.2f})")
    pd.Series(S, name="value").to_frame().to_csv(OUT / "p2_evidence_summary.csv")
    print(pd.Series(S).to_string())


if __name__ == "__main__":
    main()

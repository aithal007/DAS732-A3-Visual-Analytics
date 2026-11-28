"""
p3_varshith_plots.py  -  Varshith M Gowda (figures V7, V10, V11a, V11b, V12)
Reads the outputs of p3_varshith_supervised.py (run that first).
"""
from __future__ import annotations

import json
import warnings

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon

from common import (ACCENT, ACCENT2, BAD, GEO, IMG, INK, MUTED, OUT, PROC, REGION_COLORS,
                    REGION_ORDER, SEED, TARGET, savefig)
from p3_varshith_supervised import ITER2_FEATURES, MICE, add_missing_flags, hgb

warnings.filterwarnings("ignore")
NICE = {"broadband_subs_per100": "Fixed broadband /100", "mobile_subs_per100": "Mobile subs /100",
        "telephone_lines_per100": "Landlines /100", "year": "Year"}


def load():
    cc = pd.read_parquet(PROC / "p3_iter1_oof.parquet")
    full = pd.read_parquet(PROC / "p3_iter2_oof.parquet")
    te = pd.read_parquet(PROC / "p3_forecast_holdout.parquet")
    clean = pd.read_parquet(PROC / "panel_iteration2.parquet")
    res = pd.read_csv(OUT / "p3_cv_results.csv", index_col=0)
    reg = pd.read_csv(OUT / "p3_regional_rmse_same_rows.csv", index_col=0)
    perm = pd.read_csv(OUT / "p3_iter1_permutation_importance_heldout.csv", index_col=0)
    by_year = pd.read_csv(OUT / "p3_forecast_coverage_by_year.csv", index_col=0)
    fm = json.load(open(OUT / "p3_model_metrics.json"))["forecast"]
    return cc, full, te, clean, res, reg, perm, by_year, fm


def fig_v7(cc, res, perm):
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw={"width_ratios": [1, 1.1, 0.9]})
    ax = axs[0]
    for r in REGION_ORDER:
        g = cc[cc.region == r]
        ax.scatter(g[TARGET], g.pred_gbm_i1, s=5, alpha=0.35, color=REGION_COLORS[r], label=r, lw=0)
    ax.plot([0, 100], [0, 100], color=INK, lw=1.2, ls="--")
    m = res.loc["I1 HistGBM (raw, complete case)"]
    ax.set(xlim=(0, 100), ylim=(0, 100), xlabel="actual internet users (%)", ylabel="out-of-fold prediction (%)",
           title=f"(a) HistGBM on raw infrastructure\nRMSE {m.rmse_mean:.1f} pp, R² {m.r2_mean:.3f} (5-fold by country)")
    ax.legend(markerscale=3, fontsize=7.5, loc="upper left")

    ax = axs[1]
    data = [cc.loc[cc.region == r, "residual_i1"].values for r in REGION_ORDER]
    bp = ax.boxplot(data, vert=False, patch_artist=True, showfliers=False, widths=0.6)
    for b, r in zip(bp["boxes"], REGION_ORDER):
        b.set(facecolor=REGION_COLORS[r] + "55", edgecolor=REGION_COLORS[r])
    for med in bp["medians"]:
        med.set(color=INK, lw=1.6)
    ax.set_yticks(range(1, len(REGION_ORDER) + 1), REGION_ORDER)
    ax.axvline(0, color=BAD, ls="--", lw=1.1)
    # error by internet level: where does the baseline fail?
    bins = pd.cut(cc[TARGET], [0, 10, 30, 60, 85, 100.1])
    by_lvl = cc.groupby(bins, observed=True).residual_i1.apply(lambda s: np.sqrt(np.mean(s ** 2)))
    txt = "RMSE by actual level:\n" + "\n".join(f"{str(k):>12}: {v:4.1f}" for k, v in by_lvl.items())
    ax.text(0.98, 0.03, txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=7.4, family="monospace",
            bbox=dict(fc="white", ec=MUTED, lw=0.6))
    ax.set(xlabel="residual = actual - predicted (pp)",
           title="(b) Residuals by region (whiskers 1.5 IQR)\nerror is small below 10% online, 12-14 pp everywhere above")

    ax = axs[2]
    p = perm.sort_values("mean")
    ax.barh([NICE.get(i, i) for i in p.index], p["mean"], xerr=p["std"], color=ACCENT, capsize=3)
    ax.set(xlabel="drop in held-out R² when the feature is shuffled",
           title="(c) Permutation importance on held-out folds\nbroadband dominates; mobile adds little")
    fig.suptitle("V7  Iteration 1 baseline: where a model of raw infrastructure fails", x=0.01, ha="left",
                 fontsize=12.5, weight="bold")
    fig.tight_layout()
    savefig(fig, "V7_baseline_diagnostics.png")
    return by_lvl


def fig_v10(res, reg):
    fig, axs = plt.subplots(1, 2, figsize=(16, 5.4), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axs[0]
    r = reg.drop(index=[i for i in reg.index if i == "Other"]).sort_values("rmse_i1")
    i1 = res.loc["I1 HistGBM (raw, complete case)", "rmse_mean"]
    i2 = res.loc["I2 HistGBM (engineered + MICE), same rows as I1", "rmse_mean"]
    rows = list(r.itertuples()) + [None]
    for y, row in enumerate(rows):
        a, b, lab = (i1, i2, "OVERALL (CV mean)") if row is None else (row.rmse_i1, row.rmse_i2, f"{row.Index} (n={int(row.n)})")
        better = b < a
        ax.plot([a, b], [y, y], color=ACCENT if better else BAD, lw=3 if row is None else 2, alpha=0.7)
        ax.scatter(a, y, color=MUTED, s=60, zorder=3)
        ax.scatter(b, y, color=ACCENT if better else BAD, s=60, zorder=3)
        ax.text(max(a, b) + 0.2, y, f"{(b - a) / a:+.0%}", va="center", fontsize=8.5,
                color=ACCENT if better else BAD, weight="bold")
        ax.text(4.6, y, lab, va="center", ha="right", fontsize=9, weight="bold" if row is None else None)
    ax.set_yticks([])
    ax.set_xlim(4.8, 16)
    ax.scatter([], [], color=MUTED, label="iteration 1 (raw, complete case)")
    ax.scatter([], [], color=ACCENT, label="iteration 2 (transformed), SAME rows")
    ax.legend(loc="lower right")
    ax.set(xlabel="RMSE (percentage points), identical rows and folds",
           title="(a) Better in 4 of 6 regions; flat (+1%) in the Americas")
    ax.spines["left"].set_visible(False)

    ax = axs[1]
    order = ["I1 Ridge (raw, complete case)", "I1 HistGBM (raw, complete case)", "I1 Random Forest (raw, complete case)",
             "I2 HistGBM (engineered + MICE), same rows as I1", "Ablation A: raw features + MICE, all rows",
             "Ablation B: engineered + median imputation, all rows", "I2 HistGBM (engineered + MICE), all rows"]
    t = res.loc[order]
    cols = [MUTED, MUTED, MUTED, ACCENT, ACCENT2, ACCENT2, ACCENT]
    ax.barh(range(len(t)), t.rmse_mean, xerr=t.rmse_std, color=cols, capsize=3)
    for i, (k, row) in enumerate(t.iterrows()):
        ax.text(row.rmse_mean + row.rmse_std + 0.2, i, f"{row.rmse_mean:.2f}  (R² {row.r2_mean:.3f}, n={int(row.n_rows)})",
                va="center", fontsize=8)
    ax.set_yticks(range(len(t)), [o.replace(" (raw, complete case)", "").replace("I2 HistGBM ", "I2 ")
                                  for o in order], fontsize=8.5)
    ax.axhline(3.5, color=MUTED, lw=0.6, ls=":")
    ax.invert_yaxis()
    ax.set(xlim=(0, 17), xlabel="5-fold grouped CV RMSE ± sd (pp)",
           title="(b) Engineered features carry the gain; MICE adds a little")
    fig.suptitle(f"V10  Iteration 1 vs 2: on identical rows RMSE falls {i1:.2f} → {i2:.2f} pp "
                 f"({(i2 - i1) / i1:+.0%}); with imputation more rows become usable",
                 x=0.01, ha="left", fontsize=12.5, weight="bold")
    fig.tight_layout()
    savefig(fig, "V10_iteration_comparison.png")


def fig_v11a_shap(clean):
    import shap
    d = add_missing_flags(clean.copy()).dropna(subset=[TARGET])
    imp = MICE().fit(d[ITER2_FEATURES])
    X = pd.DataFrame(imp.transform(d[ITER2_FEATURES]), columns=ITER2_FEATURES)
    m = hgb().fit(X, d[TARGET])
    S = X.sample(600, random_state=SEED)
    try:
        sv = shap.TreeExplainer(m)(S)
    except Exception:
        sv = shap.Explainer(m.predict, X.sample(100, random_state=SEED))(S)
    imp_rank = pd.Series(np.abs(sv.values).mean(0), index=ITER2_FEATURES).sort_values(ascending=False)
    imp_rank.to_csv(OUT / "p3_shap_mean_abs.csv")
    plt.figure(figsize=(10, 6.5))
    shap.plots.beeswarm(sv, max_display=14, show=False, color_bar_label="feature value")
    plt.title("V11a  SHAP for the iteration-2 model: phone ownership (mostly MICE-imputed) and\n"
              "log broadband lead; landline_peak, the A1 'history' feature, ranks third",
              loc="left", fontsize=11.5, weight="bold")
    fig = plt.gcf()
    savefig(fig, "V11a_shap_beeswarm.png")
    return imp_rank


def fig_v11b_map(full):
    import json as _j
    gj = _j.load(open(GEO))
    res = full[full.year >= 2015].groupby(["code", "entity"]).residual_i2.mean().reset_index()
    rmap = dict(zip(res.code, res.residual_i2))
    norm = mcolors.TwoSlopeNorm(vmin=-15, vcenter=0, vmax=15)
    cmap = plt.get_cmap("RdBu")
    fig, ax = plt.subplots(figsize=(15, 7.3))
    patches, fcs = [], []
    for ft in gj["features"]:
        pr = ft["properties"]
        iso = pr.get("adm0_a3") if pr.get("iso_a3") in (None, "-99") else pr.get("iso_a3")
        v = rmap.get(iso)
        col = cmap(norm(np.clip(v, -15, 15))) if v is not None else "#E6E6E6"
        g = ft["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        for poly in polys:
            patches.append(MplPolygon(np.array(poly[0]), closed=True))
            fcs.append(col)
    ax.add_collection(PatchCollection(patches, facecolors=fcs, edgecolors="#666666", linewidths=0.3))
    ax.set(xlim=(-180, 180), ylim=(-58, 85), aspect="equal")
    ax.axis("off")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02, shrink=0.5)
    cb.set_label("mean residual 2015-2024, actual - predicted (pp);  blue = more online than infrastructure predicts")
    top = res.reindex(res.residual_i2.abs().sort_values(ascending=False).index).head(8)
    ax.text(-178, -50, "largest |residual|:\n" + "\n".join(f"{r.entity}: {r.residual_i2:+.1f}" for r in top.itertuples()),
            fontsize=8, va="bottom", bbox=dict(fc="white", ec=MUTED, lw=0.5))
    ax.set_title("V11b  Where infrastructure under- or over-explains internet use (iteration 2, out-of-fold)",
                 loc="left", fontsize=12, weight="bold")
    savefig(fig, "V11b_residual_choropleth.png")
    try:
        import plotly.express as px
        f = px.choropleth(res, locations="code", color=res.residual_i2.clip(-15, 15), hover_name="entity",
                          hover_data={"residual_i2": ":.1f"}, color_continuous_scale="RdBu",
                          color_continuous_midpoint=0, labels={"color": "residual (pp)"},
                          title="V11b mean residual 2015-2024 (actual - predicted)")
        f.write_html(str(OUT / "V11b_residual_choropleth.html"), include_plotlyjs="cdn")
    except Exception as e:
        print("  plotly html skipped:", e)
    return res


def fig_v12(te, clean, fm, by_year):
    names = ["India", "Kenya", "Brazil", "Indonesia", "Germany", "Nigeria"]
    fig = plt.figure(figsize=(17, 8.6))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.05])
    axs = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(6)]
    for ax, nm in zip(axs, names):
        h = clean[(clean.entity == nm) & (clean.year >= 2000) & (clean.year <= 2020)]
        f = te[te.entity == nm].sort_values("year")
        ax.plot(h.year, h[TARGET], color=INK, lw=2, label="observed ≤ 2020 (training)")
        if len(f):
            ax.fill_between(f.year, f.q10_cqr, f.q90_cqr, color=ACCENT, alpha=0.15, label="80% conformal (CQR)")
            ax.fill_between(f.year, f.q10, f.q90, color=ACCENT, alpha=0.35, label="80% raw quantile")
            ax.plot(f.year, f.q50, "--", color=ACCENT, lw=2, label="median forecast")
            ax.plot(f.year, f[TARGET], "o", color=BAD, ms=5, label="observed 2021-25 (hidden)")
        ax.axvline(2020.5, color=MUTED, lw=0.8)
        ax.set(xlim=(2000, 2025.5), ylim=(0, 100), title=nm)
    axs[0].legend(loc="upper left", fontsize=7.5)
    ax = fig.add_subplot(gs[:, 3])
    x = by_year.index.values
    ax.bar(x - 0.2, by_year.coverage_raw * 100, 0.4, color=ACCENT + "88", label="raw quantile")
    ax.bar(x + 0.2, by_year.coverage_cqr * 100, 0.4, color=ACCENT, label="conformal (CQR)")
    ax.axhline(80, color=BAD, ls="--", lw=1.2)
    ax.text(2020.6, 81.5, "nominal 80%", color=BAD, fontsize=8)
    for xi, n in zip(x, by_year.n):
        ax.text(xi, 2, f"n={n}", ha="center", fontsize=7, color="white" if n > 20 else INK)
    ax.set_xticks(x)
    ax.set(ylim=(0, 100), ylabel="share of 2021-25 outcomes inside the interval (%)",
           title=f"Coverage: raw {fm['coverage_80_quantile']:.0%} → CQR {fm['coverage_80_cqr']:.0%}\n"
                 f"{fm['share_underpredicted']:.0%} of forecasts are too LOW\n(mean error +{fm['mean_signed_error']:.1f} pp)")
    ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("V12  Cone of uncertainty: forecasting 2021-25 from a model trained on ≤ 2020. "
                 "The raw band is too narrow after the 2020 shift (V13); conformal re-calibration restores coverage",
                 x=0.01, ha="left", fontsize=12, weight="bold")
    fig.tight_layout()
    savefig(fig, "V12_forecast_fan_chart.png")


def fig_v12b_pdp_calibration(clean):
    """(a-c) partial dependence + ICE of the iteration-2 model; (d) empirical vs nominal coverage."""
    from sklearn.inspection import partial_dependence
    d = add_missing_flags(clean.copy()).dropna(subset=[TARGET])
    imp = MICE().fit(d[ITER2_FEATURES])
    X = pd.DataFrame(imp.transform(d[ITER2_FEATURES]), columns=ITER2_FEATURES)
    m = hgb().fit(X, d[TARGET])
    feats = [("log1p_broadband", "log(1 + broadband /100)", [0, 1, 3, 10, 30]),
             ("landline_peak", "landline peak /100 (history)", None),
             ("years_since_mobile_gt_landline", "years since mobile overtook landlines", None)]
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.9), gridspec_kw={"width_ratios": [1, 1, 1, 1.15]})
    Xs = X.sample(800, random_state=SEED)
    pd_rows = []
    for ax, (f, lab, ticks) in zip(axs[:3], feats):
        r = partial_dependence(m, Xs, [f], kind="both", grid_resolution=40,
                               percentiles=(0.02, 0.98))
        grid = r["grid_values"][0]
        ice = r["individual"][0]
        for row in ice[np.random.default_rng(SEED).choice(len(ice), 60, replace=False)]:
            ax.plot(grid, row, color=ACCENT, alpha=0.12, lw=0.8)
        ax.plot(grid, r["average"][0], color=INK, lw=2.6, label="partial dependence")
        ax.plot([], [], color=ACCENT, alpha=0.5, label="ICE (60 country-years)")
        if ticks:
            ax.set_xticks(np.log1p(ticks), ticks)
            ax.set_xlim(left=0)
            lab = "fixed broadband /100 (log axis)"
        rng = r["average"][0].max() - r["average"][0].min()
        pd_rows.append({"feature": f, "pd_range_pp": rng})
        ax.set(xlabel=lab, ylim=(0, 100), ylabel="predicted internet users (%)" if f == feats[0][0] else None,
               title=f"PD range {rng:.0f} pp")
        ax.legend(loc="upper left", fontsize=7.5)
    axs[0].set_title(f"(a) Broadband: flat on the floor (<3/100), rising 3→30\nPD range {pd_rows[0]['pd_range_pp']:.0f} pp")
    axs[1].set_title(f"(b) Landline history adds lift\nPD range {pd_rows[1]['pd_range_pp']:.0f} pp")
    axs[2].set_title(f"(c) Time since mobile overtook landlines\nPD range {pd_rows[2]['pd_range_pp']:.0f} pp")
    # (d) calibration of intervals at several nominal levels
    dd = d.copy()
    tr, te = dd[dd.year <= 2020], dd[(dd.year > 2020) & (dd.year <= 2025)]
    tr0, cal = dd[dd.year < 2018], dd[(dd.year >= 2018) & (dd.year <= 2020)]
    imp1 = MICE().fit(tr[ITER2_FEATURES]); imp0 = MICE().fit(tr0[ITER2_FEATURES])
    T = lambda im, z: pd.DataFrame(im.transform(z[ITER2_FEATURES]), columns=ITER2_FEATURES)  # noqa: E731
    Xtr, Xte, Xtr0, Xcal = T(imp1, tr), T(imp1, te), T(imp0, tr0), T(imp0, cal)
    rows = []
    for nom in [0.5, 0.6, 0.7, 0.8, 0.9]:
        a = 1 - nom
        lo = hgb(loss="quantile", quantile=a / 2).fit(Xtr, tr[TARGET]).predict(Xte)
        hi = hgb(loss="quantile", quantile=1 - a / 2).fit(Xtr, tr[TARGET]).predict(Xte)
        lo0 = hgb(loss="quantile", quantile=a / 2).fit(Xtr0, tr0[TARGET]).predict(Xcal)
        hi0 = hgb(loss="quantile", quantile=1 - a / 2).fit(Xtr0, tr0[TARGET]).predict(Xcal)
        sc = np.maximum(lo0 - cal[TARGET].values, cal[TARGET].values - hi0)
        q = np.quantile(sc, min(1, np.ceil((len(sc) + 1) * nom) / len(sc)))
        y = te[TARGET].values
        rows.append({"nominal": nom, "raw": np.mean((y >= lo) & (y <= hi)),
                     "cqr": np.mean((y >= lo - q) & (y <= hi + q)), "margin": q})
    cal_df = pd.DataFrame(rows)
    cal_df.to_csv(OUT / "p3_interval_calibration.csv", index=False)
    pd.DataFrame(pd_rows).to_csv(OUT / "p3_partial_dependence_ranges.csv", index=False)
    ax = axs[3]
    ax.plot([0.45, 0.95], [0.45, 0.95], color=MUTED, ls="--", lw=1, label="perfect calibration")
    ax.plot(cal_df.nominal, cal_df.raw, "o-", color=BAD, lw=2, label="raw quantile")
    ax.plot(cal_df.nominal, cal_df.cqr, "s-", color=ACCENT, lw=2, label="conformal (CQR)")
    for r_ in cal_df.itertuples():
        ax.text(r_.nominal, r_.raw - 0.045, f"{r_.raw:.0%}", ha="center", fontsize=8, color=BAD)
        ax.text(r_.nominal, r_.cqr + 0.02, f"{r_.cqr:.0%}", ha="center", fontsize=8, color=ACCENT)
    ax.set(xlim=(0.45, 0.95), ylim=(0.2, 1.0), xlabel="nominal coverage", ylabel="empirical coverage 2021-25",
           title="(d) Interval calibration at five levels:\nraw is over-confident everywhere; CQR tracks the diagonal")
    ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("V12b  What the model learned (partial dependence) and how far to trust it (calibration curve)",
                 x=0.01, ha="left", fontsize=12.5, weight="bold")
    fig.tight_layout()
    savefig(fig, "V12b_pdp_calibration.png")
    return cal_df, pd.DataFrame(pd_rows)


def main():
    cc, full, te, clean, res, reg, perm, by_year, fm = load()
    print("V7 ..."); lvl = fig_v7(cc, res, perm); print(lvl.round(2).to_string())
    print("V10 ..."); fig_v10(res, reg)
    print("V11a ..."); sh = fig_v11a_shap(clean); print(sh.head(10).round(2).to_string())
    print("V11b ..."); fig_v11b_map(full)
    print("V12 ..."); fig_v12(te, clean, fm, by_year)
    print("V12b ..."); c, r = fig_v12b_pdp_calibration(clean); print(c.round(3).to_string()); print(r.round(1).to_string())


if __name__ == "__main__":
    main()

"""
p1_parikshith_stats.py  -  Parikshith Aithal K V
Statistical analysis + exploratory visualisation (Iteration 1), the feedback-loop
transformation view (Iteration 2) and the drift chart that motivates re-calibration.

Figures
  V1  Diffusion curves: 3-parameter logistic (S-curve) fits per country, take-off year by
      region, Kruskal-Wallis + Dunn-style pairwise Mann-Whitney test.
  V2  Broadband floor effect: segmented (piecewise) regression with an estimated breakpoint
      + LOWESS, and Spearman rho by year with country-bootstrap 95% CIs (mobile decouples).
  V3  Multi-SIM inflation: subscriptions vs owners, the harmonised 1.5 SIM/owner threshold,
      and the Fiji 2023-24 recording artefact found later by anomaly detection.
  V9  Feedback loop made visible: what the transformation changed (artefact repair,
      engineered vs raw feature association with the target, coverage).
  V14 Synthesis of the answer from both iterations.
  (V0 and W1-W3, the workflow diagrams, are TikZ diagrams in report/diagrams/.)
Tables in outputs/p1_*.csv
"""
from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import stats
from scipy.optimize import curve_fit

from common import (ACCENT, ACCENT2, BAD, ENGINEERED, INK, MULTI_SIM_RATIO, MUTED, OUT,
                    REGION_COLORS, REGION_ORDER, SEED, SHORT, TARGET, load_panel, savefig,
                    transform_for_iteration2)

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(SEED)


# ======================================================================================
# V0  -  the workflow diagram (compact Keim diamond + the UNROLLED version)
# ======================================================================================
# ======================================================================================
# V1  -  diffusion (logistic S-curve) statistics
# ======================================================================================
def _logistic(t, L, k, t0):
    return L / (1.0 + np.exp(-k * (t - t0)))


def fit_logistic_curves(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code, g in df[(df.year >= 1990)].dropna(subset=[TARGET]).groupby("code"):
        if len(g) < 15:
            continue
        t, y = g.year.values.astype(float), g[TARGET].values
        try:
            p, _ = curve_fit(_logistic, t, y, p0=[max(y.max(), 50), 0.25, 2012],
                             bounds=([max(y.max() * 0.9, 5), 0.02, 1985], [100, 2.0, 2060]), maxfev=20000)
        except Exception:
            continue
        pred = _logistic(t, *p)
        r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
        rows.append({"code": code, "entity": g.entity.iloc[0], "region": g.region.iloc[0],
                     "n_years": len(g), "L_saturation": p[0], "k_growth": p[1], "t0_takeoff": p[2],
                     "r2": r2, "last_value": y[-1]})
    return pd.DataFrame(rows)


def fig_v1_diffusion(df: pd.DataFrame):
    fits = fit_logistic_curves(df)
    fits.to_csv(OUT / "p1_logistic_fits.csv", index=False)
    good = fits[fits.r2 >= 0.9]
    groups = [good.loc[good.region == r, "t0_takeoff"].values for r in REGION_ORDER]
    H, p_kw = stats.kruskal(*[g for g in groups if len(g) > 2])
    # pairwise Mann-Whitney vs Europe, Holm-corrected
    pw = []
    for r in REGION_ORDER:
        if r == "Europe":
            continue
        a, b = good.loc[good.region == r, "t0_takeoff"], good.loc[good.region == "Europe", "t0_takeoff"]
        if len(a) > 2:
            u, p = stats.mannwhitneyu(a, b)
            pw.append({"region": r, "median_t0": a.median(), "europe_median_t0": b.median(),
                       "lag_years": a.median() - b.median(), "p_raw": p})
    pw = pd.DataFrame(pw).sort_values("p_raw")
    m = len(pw)
    pw["p_holm"] = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * pw.p_raw.values))
    pw.to_csv(OUT / "p1_takeoff_lag_tests.csv", index=False)

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.9), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    ax = axs[0]
    for code, g in df[df.year >= 1990].dropna(subset=[TARGET]).groupby("code"):
        ax.plot(g.year, g[TARGET], color=REGION_COLORS.get(g.region.iloc[0], "#999"), lw=0.5, alpha=0.22)
    tt = np.linspace(1990, 2025, 200)
    for r in REGION_ORDER:
        sub = good[good.region == r]
        if len(sub):
            med = sub[["L_saturation", "k_growth", "t0_takeoff"]].median().values
            ax.plot(tt, _logistic(tt, *med), color=REGION_COLORS[r], lw=2.6, label=f"{r} (n={len(sub)})")
    ax.set(xlim=(1990, 2025), ylim=(0, 100), xlabel="year", ylabel="internet users (% of population)",
           title="Diffusion follows an S-curve; regions are shifted in time")
    ax.legend(loc="upper left", fontsize=8)
    ax.text(0.99, 0.02, "thin: observed country series\nthick: logistic with region-median (L, k, t0)",
            transform=ax.transAxes, ha="right", fontsize=7.5, color=MUTED)

    ax = axs[1]
    for i, r in enumerate(REGION_ORDER):
        v = good.loc[good.region == r, "t0_takeoff"].values
        ax.boxplot(v, positions=[i], widths=0.55, vert=True, patch_artist=True, showfliers=False,
                   boxprops=dict(fc=REGION_COLORS[r] + "55", ec=REGION_COLORS[r]),
                   medianprops=dict(color=INK, lw=1.6), whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
        ax.scatter(i + RNG.uniform(-0.18, 0.18, len(v)), v, s=9, color=REGION_COLORS[r], alpha=0.8, zorder=3)
    ax.set_xticks(range(len(REGION_ORDER)), [r.replace(" ", "\n") for r in REGION_ORDER], fontsize=8)
    ax.set(ylabel="take-off year t0 (50% of saturation)", ylim=(1995, 2035),
           title=f"Take-off year differs by region\nKruskal-Wallis H={H:.1f}, p={p_kw:.1e}")
    afr = pw[pw.region == "Africa"].iloc[0]
    ax.annotate(f"Africa median {afr.median_t0:.0f}\n{afr.lag_years:.0f} yrs after Europe\n(Holm p={afr.p_holm:.1e})",
                xy=(0, afr.median_t0), xytext=(1.6, 2031), fontsize=7.8,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))

    ax = axs[2]
    ax.scatter(good.t0_takeoff, good.k_growth, c=[REGION_COLORS.get(r, "#999") for r in good.region],
               s=16, alpha=0.8)
    rho, p = stats.spearmanr(good.t0_takeoff, good.k_growth)
    ax.set(xlabel="take-off year t0", ylabel="growth rate k (per year)", xlim=(1995, 2035),
           title=f"Late adopters do NOT climb faster\nSpearman rho={rho:.2f}, p={p:.1e}")
    for nm in ["Kiribati", "India", "South Korea", "Nigeria", "Norway"]:
        r_ = good[good.entity == nm]
        if len(r_):
            ax.annotate(nm, (r_.t0_takeoff.iloc[0], r_.k_growth.iloc[0]), fontsize=7.5,
                        xytext=(4, 3), textcoords="offset points")
    fig.suptitle(f"V1  Every region follows the same S-curve, up to 12 years late, and late starters do not catch up "
                 f"({len(good)} of {len(fits)} logistic fits with R² ≥ 0.9)", x=0.01, ha="left", fontsize=12.5,
                 weight="bold")
    fig.tight_layout()
    savefig(fig, "V1_diffusion_scurves.png")
    return fits, pw, (H, p_kw), (rho, p)


# ======================================================================================
# V2  -  broadband floor effect + decoupling of mobile
# ======================================================================================
def _segmented_fit(x, y):
    best = None
    for c in np.quantile(x, np.linspace(0.1, 0.9, 81)):
        X = np.column_stack([np.ones_like(x), x, np.maximum(0, x - c)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        sse = np.sum((y - X @ beta) ** 2)
        if best is None or sse < best[0]:
            best = (sse, c, beta)
    sse, c, beta = best
    X1 = np.column_stack([np.ones_like(x), x])
    b1, *_ = np.linalg.lstsq(X1, y, rcond=None)
    sse1 = np.sum((y - X1 @ b1) ** 2)
    n = len(y)
    F = ((sse1 - sse) / 2) / (sse / (n - 4))   # +1 slope +1 breakpoint
    return c, beta, 1 - sse / np.sum((y - y.mean()) ** 2), F, stats.f.sf(F, 2, n - 4)


def _spearman_ci(df_year, col, B=400):
    d = df_year[[col, TARGET]].dropna()
    if len(d) < 25:
        return np.nan, np.nan, np.nan, len(d)
    r = stats.spearmanr(d[col], d[TARGET])[0]
    idx = RNG.integers(0, len(d), (B, len(d)))
    a, b = d[col].values, d[TARGET].values
    boots = [stats.spearmanr(a[i], b[i])[0] for i in idx]
    return r, np.percentile(boots, 2.5), np.percentile(boots, 97.5), len(d)


def fig_v2_floor_effect(df: pd.DataFrame):
    d = df[(df.year >= 2005) & (df.year <= 2024)].dropna(subset=[TARGET, "broadband_subs_per100"])
    x, y = np.log1p(d.broadband_subs_per100.values), d[TARGET].values
    c, beta, r2, F, pF = _segmented_fit(x, y)
    bp = np.expm1(c)
    floor = d[d.broadband_subs_per100 < 1]
    from statsmodels.nonparametric.smoothers_lowess import lowess
    lw = lowess(y, x, frac=0.25, return_sorted=True)

    rows = []
    for yr in range(2000, 2025):
        s = df[df.year == yr]
        for col in ["broadband_subs_per100", "mobile_subs_per100", "telephone_lines_per100"]:
            r, lo, hi, n = _spearman_ci(s, col)
            rows.append({"year": yr, "indicator": col, "rho": r, "lo": lo, "hi": hi, "n": n})
    sp = pd.DataFrame(rows)
    sp.to_csv(OUT / "p1_spearman_by_year.csv", index=False)
    pd.DataFrame([{"breakpoint_log1p": c, "breakpoint_per100": bp, "slope_below": beta[1],
                   "slope_above": beta[1] + beta[2], "r2": r2, "F_vs_linear": F, "p": pF, "n": len(d),
                   "share_rows_bb_lt1": (d.broadband_subs_per100 < 1).mean(),
                   "internet_median_when_bb_lt1": floor[TARGET].median(),
                   "internet_p90_when_bb_lt1": floor[TARGET].quantile(0.9)}]).to_csv(
        OUT / "p1_segmented_regression.csv", index=False)

    fig, axs = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axs[0]
    ax.scatter(x, y, s=5, alpha=0.25, c=[REGION_COLORS.get(r, "#999") for r in d.region], lw=0)
    xx = np.linspace(x.min(), x.max(), 200)
    ax.plot(xx, beta[0] + beta[1] * xx + beta[2] * np.maximum(0, xx - c), color=INK, lw=2.4,
            label=f"segmented fit (R²={r2:.2f})")
    ax.plot(lw[:, 0], lw[:, 1], color=BAD, lw=1.6, ls="--", label="LOWESS (frac=0.25)")
    ax.axvline(c, color=MUTED, ls=":", lw=1.2)
    ax.axvspan(0, np.log1p(1), color=ACCENT2, alpha=0.12)
    ax.text(np.log1p(1) / 2, 97, f"floor: <1 sub/100\n{len(floor)} rows, internet\nmedian {floor[TARGET].median():.0f}%,"
            f" p90 {floor[TARGET].quantile(.9):.0f}%", fontsize=7.6, va="top", ha="left", color=INK)
    ax.text(np.log1p(2.2), 84, f"segmented regression: breakpoint ≈ {bp:.1f} subs/100\nslope {beta[1]:.1f} → {beta[1]+beta[2]:.1f} pp per log-unit\n"
            f"F={F:.0f} vs linear, p={pF:.0e}", fontsize=7.8, color=INK)
    ticks = [0, 1, 3, 10, 30, 60]
    ax.set_xticks(np.log1p(ticks), ticks)
    ax.set(xlabel="fixed broadband subscriptions per 100 (log1p scale)", ylabel="internet users (%)",
           ylim=(0, 100), title=f"Internet grows with log(broadband), but {len(floor)/len(d):.0%} of rows sit on the\n"
                 "broadband floor (<1 sub/100), where internet still ranges 0-30%+")
    for r in REGION_ORDER:
        ax.scatter([], [], color=REGION_COLORS[r], s=18, label=r)
    ax.legend(loc="lower right", fontsize=7.4, ncol=2)

    ax = axs[1]
    colors = {"broadband_subs_per100": ACCENT, "mobile_subs_per100": ACCENT2, "telephone_lines_per100": MUTED}
    for col, g in sp.groupby("indicator"):
        g = g.dropna()
        ax.plot(g.year, g.rho, color=colors[col], lw=2.2, label=SHORT[col])
        ax.fill_between(g.year, g.lo, g.hi, color=colors[col], alpha=0.18)
    ax.axhline(0, color=MUTED, lw=0.8)
    m24 = sp[(sp.indicator == "mobile_subs_per100") & (sp.year == 2024)].iloc[0]
    b24 = sp[(sp.indicator == "broadband_subs_per100") & (sp.year == 2024)].iloc[0]
    ax.annotate(f"2024: mobile ρ={m24.rho:.2f} [{m24.lo:.2f}, {m24.hi:.2f}]", (2024, m24.rho),
                xytext=(2009, 0.25), fontsize=8, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    ax.annotate(f"broadband ρ={b24.rho:.2f}", (2024, b24.rho), xytext=(2015, 0.97), fontsize=8,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    ax.set(xlabel="year", ylabel="Spearman ρ with internet users % (95% bootstrap CI)", ylim=(-0.1, 1.05),
           title="Mobile subscriptions decouple from internet use as they saturate")
    ax.legend(loc="lower left")
    fig.suptitle("V2  Broadband has a floor and a non-linear effect; mobile stopped separating countries once it "
                 "saturated (2005-2024)",
                 x=0.01, ha="left", fontsize=12.5, weight="bold")
    fig.tight_layout()
    savefig(fig, "V2_floor_effect_decoupling.png")
    return sp, (bp, beta, r2, F, pF, floor)


# ======================================================================================
# V3  -  multi-SIM inflation and the Fiji artefact
# ======================================================================================
def fig_v3_multisim(df: pd.DataFrame):
    d = df.dropna(subset=["mobile_subs_per100", "mobile_phone_ownership_share_pct"])
    d = d[d.mobile_phone_ownership_share_pct > 0].copy()
    d["spo"] = d.mobile_subs_per100 / d.mobile_phone_ownership_share_pct
    latest = d.sort_values("year").groupby("code").tail(1)
    q1, q3 = latest.spo.quantile([0.25, 0.75])
    fence = q3 + 1.5 * (q3 - q1)
    share = (latest.spo >= MULTI_SIM_RATIO).mean()
    rho_sub = stats.spearmanr(latest.mobile_subs_per100, latest[TARGET], nan_policy="omit")[0]
    rho_own = stats.spearmanr(latest.mobile_phone_ownership_share_pct, latest[TARGET], nan_policy="omit")[0]
    pd.DataFrame([{"n_countries": len(latest), "median_spo": latest.spo.median(), "q1": q1, "q3": q3,
                   "tukey_fence": fence, "threshold": MULTI_SIM_RATIO, "share_above_threshold": share,
                   "rho_subs_internet": rho_sub, "rho_owners_internet": rho_own}]).to_csv(
        OUT / "p1_multisim_stats.csv", index=False)

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.9), gridspec_kw={"width_ratios": [1.2, 0.9, 1]})
    ax = axs[0]
    ax.scatter(latest.mobile_phone_ownership_share_pct, latest.mobile_subs_per100, s=18, alpha=0.8,
               c=[REGION_COLORS.get(r, "#999") for r in latest.region])
    xx = np.linspace(0, 100, 50)
    ax.plot(xx, xx, color=INK, lw=1.2, label="1 SIM per owner")
    ax.plot(xx, MULTI_SIM_RATIO * xx, color=BAD, lw=1.4, ls="--", label=f"{MULTI_SIM_RATIO} SIMs per owner (flag)")
    for nm, dx, dy, ha in [("Fiji", -10, -4, "right"), ("Hong Kong", -10, -4, "right"),
                           ("Cote d'Ivoire", -8, 8, "right"), ("Uganda", 6, 8, "left")]:
        r_ = latest[latest.entity == nm]
        if len(r_):
            ax.annotate(nm, (r_.mobile_phone_ownership_share_pct.iloc[0], r_.mobile_subs_per100.iloc[0]),
                        xytext=(dx, dy), textcoords="offset points", fontsize=8.5, ha=ha, color=INK,
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.set(xlabel="adults owning a mobile phone (%)", ylabel="mobile subscriptions per 100 people",
           ylim=(0, 600), xlim=(20, 102), title="Subscriptions count SIMs, not people")
    for r in REGION_ORDER:
        ax.scatter([], [], s=18, color=REGION_COLORS[r], label=r)
    ax.legend(loc="upper left", ncol=2, fontsize=7.8)

    ax = axs[1]
    ax.hist(latest.spo.clip(upper=4), bins=30, color=ACCENT, alpha=0.8)
    ax.axvline(MULTI_SIM_RATIO, color=BAD, ls="--", lw=1.4)
    ax.axvline(fence, color=MUTED, ls=":", lw=1.2)
    ax.text(2.45, ax.get_ylim()[1] * 0.85, f"multi-SIM flag ≥ {MULTI_SIM_RATIO}\n{share:.0%} of countries",
            fontsize=8.5, color=BAD)
    ax.annotate("", (MULTI_SIM_RATIO, ax.get_ylim()[1] * 0.86), xytext=(2.43, ax.get_ylim()[1] * 0.88),
                arrowprops=dict(arrowstyle="->", color=BAD, lw=0.8))
    ax.text(fence + 0.07, ax.get_ylim()[1] * 0.45, f"Tukey fence\n{fence:.2f}", fontsize=8.5, color=MUTED)
    ax.set(xlabel="SIMs per phone owner (clipped at 4)", ylabel="countries (latest year)",
           title=f"Median {latest.spo.median():.2f} SIMs per owner")

    ax = axs[2]
    f = df[(df.entity == "Fiji") & (df.year >= 2010)]
    ax.plot(f.year, f.mobile_subs_per100, "o-", color=BAD, lw=1.8, ms=4, label="mobile subs /100 (raw)")
    ax.plot(f.year, f.mobile_phone_ownership_share_pct, "s-", color=ACCENT, lw=1.6, ms=4, label="phone owners %")
    ax.plot(f.year, f[TARGET], "^-", color=INK, lw=1.4, ms=4, label="internet users %")
    ax.annotate("111 → 361 → 574 in two years\nwhile owners stay ~82%:\nrecording artefact, not adoption",
                (2024, 574), xytext=(2011, 420), fontsize=8, arrowprops=dict(arrowstyle="->", color=BAD))
    ax.set(xlabel="year", ylabel="subscriptions per 100 people  /  % of population",
           title="Fiji: the outlier that drives PC2 (see V4, V6)")
    ax.legend(loc="center left", fontsize=7.6)
    fig.suptitle(f"V3  Multi-SIM inflation: subscriptions correlate ρ={rho_sub:.2f} with internet use, "
                 f"ownership ρ={rho_own:.2f} (latest year per country, n={len(latest)})",
                 x=0.01, ha="left", fontsize=12.5, weight="bold")
    fig.tight_layout()
    savefig(fig, "V3_multisim_inflation.png")
    return latest, fence, share, rho_sub, rho_own


# ======================================================================================
# V9  -  the transformation, visualised (iteration 2 starts here)
# ======================================================================================
def fig_v9_transformation(raw: pd.DataFrame, clean: pd.DataFrame, arts: pd.DataFrame):
    arts.to_csv(OUT / "p1_artifact_log.csv", index=False)
    feats_raw = ["mobile_subs_per100", "telephone_lines_per100", "broadband_subs_per100",
                 "mobile_phone_ownership_share_pct"]
    rec = []
    for c in feats_raw + ENGINEERED:
        dd = clean[[c, TARGET]].dropna()
        r = stats.spearmanr(dd[c], dd[TARGET])[0] if len(dd) > 30 else np.nan
        rec.append({"feature": c, "kind": "raw" if c in feats_raw else "engineered", "rho": r,
                    "abs_rho": abs(r), "coverage": clean[c].notna().mean(), "n": len(dd)})
    cmp = pd.DataFrame(rec).sort_values("abs_rho")
    cmp.to_csv(OUT / "p1_feature_association.csv", index=False)

    fig, axs = plt.subplots(1, 3, figsize=(16, 5.4), gridspec_kw={"width_ratios": [1, 1.25, 0.9]})
    ax = axs[0]
    for nm, col, colr in [("Fiji", "mobile_subs_per100", BAD), ("Saint Kitts and Nevis", "broadband_subs_per100", ACCENT)]:
        r0 = raw[(raw.entity == nm) & (raw.year >= 2008)]
        r1 = clean[(clean.entity == nm) & (clean.year >= 2008)]
        ax.plot(r0.year, r0[col], ":", color=colr, lw=1.4)
        ax.plot(r1.year, r1[col], "o-", color=colr, lw=2, ms=3.5, label=f"{nm}: {SHORT[col]}")
        a = arts[(arts.entity == nm)]
        ax.scatter(a.year, a.value, marker="x", s=60, color=colr, zorder=5)
    ax.set_yscale("log")
    from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
    ax.yaxis.set_major_locator(FixedLocator([20, 30, 50, 100, 200, 300, 600]))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xticks(range(2008, 2025, 4))
    ax.set(xlabel="year", ylabel="value (log scale)",
           title=f"T1  {len(arts)} artefact cells removed\n(dotted = raw, x = removed, solid = kept)")
    ax.legend(loc="upper left", fontsize=7.5)

    ax = axs[1]
    col = [ACCENT if k == "engineered" else MUTED for k in cmp.kind]
    ax.barh(range(len(cmp)), cmp.rho, color=col)
    ax.set_yticks(range(len(cmp)), cmp.feature, fontsize=7.8)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set(xlabel="Spearman ρ with internet users % (all country-years)",
           title="T2  13 engineered features vs the 4 raw predictors")
    ax.scatter([], [], color=ACCENT, marker="s", label="engineered (iteration 2)")
    ax.scatter([], [], color=MUTED, marker="s", label="raw (iteration 1)")
    ax.legend(loc="lower right", fontsize=8)

    ax = axs[2]
    cov = cmp.set_index("feature").loc[feats_raw + ["sim_per_owner", "landline_peak", "leapfrog_index",
                                                    "log1p_broadband", "broadband_3yr"], "coverage"]
    ax.barh(range(len(cov)), cov.values * 100, color=[MUTED] * 4 + [ACCENT] * 5)
    ax.set_yticks(range(len(cov)), cov.index, fontsize=7.8)
    ax.set(xlim=(0, 100), xlabel="% of country-years observed",
           title="T3  coverage: why iteration 2 imputes\n(MICE fitted inside each training fold)")
    for i, v in enumerate(cov.values):
        ax.text(v * 100 + 1, i, f"{v:.0%}", va="center", fontsize=7.5)
    fig.suptitle("V9  The feedback loop changes the data: 10 artefact cells removed, 13 features that carry history "
                 "no single-year value has, and imputation where coverage is low",
                 x=0.01, ha="left", fontsize=12.5, weight="bold")
    fig.tight_layout()
    savefig(fig, "V9_feedback_transformation.png")
    return cmp


def main():
    raw = load_panel()
    print("V1 ..."); fits, pw, kw, rk = fig_v1_diffusion(raw)
    print(f"   logistic fits {len(fits)}, R2>=.9: {(fits.r2>=.9).sum()}, KW H={kw[0]:.1f} p={kw[1]:.2e}, rho(t0,k)={rk[0]:.2f}")
    print(pw.to_string(index=False))
    print("V2 ..."); sp, seg = fig_v2_floor_effect(raw)
    print(f"   breakpoint {seg[0]:.2f}/100, slopes {seg[1][1]:.1f}->{seg[1][1]+seg[1][2]:.1f}, R2 {seg[2]:.3f}, F {seg[3]:.1f}")
    print("V3 ..."); _, fence, share, rs, ro = fig_v3_multisim(raw)
    print(f"   fence {fence:.2f}, share>=1.5 {share:.2f}, rho subs {rs:.2f} owners {ro:.2f}")
    clean, arts = transform_for_iteration2(raw)
    clean.to_parquet(OUT.parent / "data" / "processed" / "panel_iteration2.parquet", index=False)
    print("V9 ..."); fig_v9_transformation(raw, clean, arts)




# ======================================================================================
# V14 -  synthesis: the answer to the central question, from both iterations
# ======================================================================================
def fig_v14_synthesis():
    import json
    from common import PROC
    fits = pd.read_csv(OUT / "p1_logistic_fits.csv")
    fits = fits[fits.r2 >= 0.9]
    clean = pd.read_parquet(PROC / "panel_iteration2.parquet")
    t2 = pd.read_csv(OUT / "iter2_assignments.csv").set_index("code")
    cv = pd.read_csv(OUT / "p3_cv_results.csv", index_col=0)
    cmp_ = pd.read_csv(OUT / "iter2_comparison.csv", index_col=0)
    fm = json.load(open(OUT / "p3_model_metrics.json"))["forecast"]

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    # (a) WHEN: take-off year by region, bootstrap CI of the median
    ax = axs[0, 0]
    eur = fits.loc[fits.region == "Europe", "t0_takeoff"].median()
    rows = []
    for r in REGION_ORDER:
        v = fits.loc[fits.region == r, "t0_takeoff"].values
        b = np.median(RNG.choice(v, (2000, len(v))), axis=1)
        rows.append((r, np.median(v), *np.percentile(b, [2.5, 97.5]), len(v)))
    rows.sort(key=lambda t: t[1])
    for i, (r, m, lo, hi, n) in enumerate(rows):
        ax.plot([lo, hi], [i, i], color=REGION_COLORS[r], lw=5, alpha=0.5, solid_capstyle="round")
        ax.scatter(m, i, s=110, color=REGION_COLORS[r], edgecolor=INK, zorder=3)
        ax.text(hi + 0.4, i, f"{m:.0f}  (+{m - eur:.1f} yr)" if r != "Europe" else f"{m:.0f}  (reference)",
                va="center", fontsize=9)
    ax.set_yticks(range(len(rows)), [f"{r} (n={n})" for r, *_, n in rows])
    ax.set_xticks(range(2004, 2025, 4))
    ax.set(xlim=(2003, 2024), xlabel="median take-off year t0 (95% bootstrap CI)",
           title="(a) WHEN: the divide is a delay on the same S-curve")
    # (b) WHAT discriminates in 2024: Spearman with CI, raw vs transformed
    ax = axs[0, 1]
    s24 = clean[clean.year == 2023]
    items = [("broadband_subs_per100", "Fixed broadband /100"), ("log1p_broadband", "log broadband (T2)"),
             ("telephone_lines_per100", "Landlines /100"), ("landline_peak", "Landline peak (T2)"),
             ("mobile_subs_per100", "Mobile subs /100 (raw)"), ("mobile_capped", "Mobile capped at 100 (T2)"),
             ("sim_per_owner", "SIMs per owner (T2)")]
    vals = [(lab, *_spearman_ci(s24, c)) for c, lab in items]
    for i, (lab, r, lo, hi, n) in enumerate(vals[::-1]):
        col = ACCENT if "(T2)" in lab else MUTED
        ax.barh(i, r, color=col, alpha=0.85)
        ax.plot([lo, hi], [i, i], color=INK, lw=1.4)
        ax.text(max(hi, r) + 0.02, i, f"{r:.2f}  (n={n})", va="center", fontsize=8.5)
    ax.set_yticks(range(len(vals)), [v[0] for v in vals[::-1]])
    ax.axvline(0, color=INK, lw=0.8)
    ax.set(xlim=(-0.3, 1.15), xlabel="Spearman ρ with internet users %, 2023 cross-section (95% CI)",
           title="(b) WHAT separates countries today: broadband and landline history, not mobile")
    # (c) WHO: archetypes share mobile, differ in broadband
    ax = axs[1, 0]
    s = clean[clean.year == 2022].set_index("code").join(t2[["traj_cluster"]], how="inner")
    cols3 = ["#E69F00", "#56B4E9", "#009E73"]
    names = ["T0 late starters", "T1 S-curve majority", "T2 early saturators"]
    for j, (var, lab) in enumerate([("mobile_capped", "mobile (capped)"), ("broadband_subs_per100", "broadband"),
                                    ("telephone_lines_per100", "landlines")]):
        for c in range(3):
            v = s.loc[s.traj_cluster == c, var].dropna()
            pos = j * 4 + c
            bp = ax.boxplot(v, positions=[pos], widths=0.7, patch_artist=True, showfliers=False)
            bp["boxes"][0].set(facecolor=cols3[c] + "88", edgecolor=cols3[c])
            bp["medians"][0].set(color=INK, lw=1.6)
            ax.text(pos, -6, f"{v.median():.0f}", ha="center", fontsize=8)
    ax.set_xticks([1, 5, 9], ["mobile subs /100 (capped)", "fixed broadband /100", "landlines /100"])
    ax.set(ylim=(-10, 105), ylabel="per 100 people (2022)",
           title="(c) WHO: mobile saturates in T1 and T2 alike; fixed broadband and\nlandlines separate all three archetypes (medians below boxes)")
    for c in range(3):
        ax.scatter([], [], marker="s", color=cols3[c], label=names[c])
    ax.legend(loc="upper right", fontsize=8.5)
    # (d) what the feedback loop changed
    ax = axs[1, 1]
    i1 = cv.loc["I1 HistGBM (raw, complete case)", "rmse_mean"]
    i2 = cv.loc["I2 HistGBM (engineered + MICE), same rows as I1", "rmse_mean"]
    metr = [("Supervised RMSE, same rows (pp)\nlower is better", i1, i2, "{:.2f}"),
            ("Cluster silhouette in trajectory space\nhigher is better", cmp_.iloc[0, 1], cmp_.iloc[1, 1], "{:.2f}"),
            ("80% interval coverage 2021-25 (%)\ncloser to 80 is better", fm["coverage_80_quantile"] * 100,
             fm["coverage_80_cqr"] * 100, "{:.0f}")]
    for i, (lab, a_, b_, f) in enumerate(metr):
        y = 2 - i
        mx = max(a_, b_)
        ax.barh(y + 0.18, a_ / mx, 0.34, color=MUTED)
        ax.barh(y - 0.18, b_ / mx, 0.34, color=ACCENT)
        ax.text(a_ / mx + 0.02, y + 0.18, "iteration 1: " + f.format(a_), va="center", fontsize=9)
        ax.text(b_ / mx + 0.02, y - 0.18, "iteration 2: " + f.format(b_), va="center", fontsize=9, color=ACCENT,
                weight="bold")
    ax.set_yticks([2, 1, 0], [m[0] for m in metr], fontsize=9)
    ax.set_xticks([])
    ax.set(xlim=(0, 1.45), title="(d) What the feedback loop changed (same data, same folds)")
    ax.spines["bottom"].set_visible(False)
    fig.suptitle("V14  Answer: mobile leapfrogging happened but did not close the divide. The divide is a delay "
                 "on one S-curve,\nand fixed infrastructure, not mobile, marks where a country is on it",
                 x=0.01, ha="left", fontsize=13, weight="bold")
    fig.tight_layout()
    savefig(fig, "V14_synthesis_answer.png")

if __name__ == "__main__":
    main()
    fig_v14_synthesis()

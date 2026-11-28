"""
p1_parikshith_convergence_quality.py  -  Parikshith Aithal K V
Convergence statistics (text only, no figure): sigma-convergence (dispersion, Gini, regional gap)
     and beta-convergence measured as the share of remaining headroom closed 2014 -> 2024.
V1c  Iteration 2 of workflow W1: each country's curve re-expressed in its own time (years since
     take-off t - t0) and its own scale (share of saturation y/L).  If the divide is a delay on one
     S-curve, the curves must collapse onto one line after this transformation.
Data quality (text only, no figure): coverage by year and region, and a missing-not-at-random test
     that justifies complete cases in iteration 1 and missingness flags + in-fold MICE in iteration 2.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from common import (ACCENT, INK, MUTED, OUT, REGION_COLORS, REGION_ORDER, SHORT, TARGET, load_panel,
                    savefig)


def _gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())) if n and x.sum() > 0 else np.nan


def _tag(ax, text, color):
    ax.text(0.0, 1.0, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5, weight="bold",
            color="white", bbox=dict(boxstyle="round,pad=0.25", fc=color, ec="none"),
            zorder=10).set_y(1.0 + 0.16)


def convergence_stats(df: pd.DataFrame):
    """sigma- and beta-convergence (quoted in the report text and used by V14; no figure)."""
    rows = []
    for yr in range(1995, 2025):
        s = df[df.year == yr].dropna(subset=[TARGET])
        if len(s) < 100:
            continue
        med = s.groupby("region")[TARGET].median()
        rows.append({"year": yr, "n": len(s), "sd": s[TARGET].std(), "gini": _gini(s[TARGET]),
                     "p10": s[TARGET].quantile(.1), "p50": s[TARGET].median(), "p90": s[TARGET].quantile(.9),
                     "gap_eu_af": med.get("Europe", np.nan) - med.get("Africa", np.nan)})
    sig = pd.DataFrame(rows)
    sig.to_csv(OUT / "p1_sigma_convergence.csv", index=False)
    a = df[df.year == 2014].set_index("code")[TARGET]
    b = df[df.year == 2024].set_index("code")[TARGET]
    bc = pd.concat([a.rename("y14"), b.rename("y24")], axis=1).dropna()
    bc = bc[bc.y14 < 95].copy()
    bc["closed"] = (bc.y24 - bc.y14) / (100 - bc.y14)
    bc["region"] = bc.index.map(df.drop_duplicates("code").set_index("code").region)
    rho, p = stats.spearmanr(bc.y14, bc.closed)
    lr = stats.linregress(bc.y14, bc.closed)
    bc.to_csv(OUT / "p1_beta_convergence.csv")
    return sig, bc, rho, p, lr


def missingness_stats(df: pd.DataFrame):
    """Coverage by year and region and the MNAR test (quoted in the report text; no figure)."""
    inds = ["internet_users_share_pct", "mobile_subs_per100", "telephone_lines_per100", "broadband_subs_per100",
            "mobile_phone_ownership_share_pct"]
    yrs = list(range(1990, 2026))
    ncty = df.code.nunique()
    cov = pd.DataFrame({SHORT[c]: [df.loc[df.year == y, c].notna().sum() / ncty * 100 for y in yrs] for c in inds},
                       index=yrs).T
    reg = pd.DataFrame({SHORT[c]: df[df.year >= 2010].groupby("region")[c].apply(lambda s: s.notna().mean() * 100)
                        for c in inds}).reindex(REGION_ORDER).T
    d = df[(df.year >= 2015) & (df.year <= 2024)].dropna(subset=[TARGET])
    tests = []
    for c in ["broadband_subs_per100", "mobile_phone_ownership_share_pct"]:
        o, m = d.loc[d[c].notna(), TARGET], d.loc[d[c].isna(), TARGET]
        tests.append((SHORT[c], o, m, stats.mannwhitneyu(o, m).pvalue))
    pd.DataFrame([{"indicator": t[0], "median_internet_observed": t[1].median(), "median_internet_missing":
                   t[2].median(), "n_obs": len(t[1]), "n_mis": len(t[2]), "p": t[3]} for t in tests]).to_csv(
        OUT / "p1_mnar_tests.csv", index=False)
    cov.to_csv(OUT / "p1_coverage_by_year.csv")

    return cov, reg, tests


def fig_v1c_alignment(clean: pd.DataFrame, fits: pd.DataFrame):
    """W1 iteration 2: does re-indexing every curve to its own take-off collapse the curves?"""
    f = fits[(fits.r2 >= 0.9)].set_index("code")
    d = clean[clean.code.isin(f.index) & clean.year.between(1990, 2025)].dropna(subset=[TARGET]).copy()
    d["t0"] = d.code.map(f.t0_takeoff)
    d["L"] = d.code.map(f.L_saturation)
    d["a"] = (d.year - d.t0).round().astype(int)             # years since own take-off
    d["ys"] = 100 * d[TARGET] / d["L"]                        # % of own saturation level

    def mean_iqr(frame, by, col, lo, hi, nmin=60):
        g = frame[frame[by].between(lo, hi)].groupby(by)[col]
        q = g.quantile(.75) - g.quantile(.25)
        q = q[g.size() >= nmin]
        return float(q.mean()), q
    iqr_cal, q_cal = mean_iqr(d, "year", TARGET, 1995, 2024)
    iqr_shift, q_shift = mean_iqr(d, "a", TARGET, -10, 10)
    iqr_both, q_both = mean_iqr(d, "a", "ys", -10, 10)
    kmed = float(f.k_growth.median())
    pd.DataFrame({"representation": ["calendar year, % of population", "years since take-off, % of population",
                                     "years since take-off, % of own saturation"],
                  "mean_IQR_pp": [iqr_cal, iqr_shift, iqr_both]}).to_csv(OUT / "p1_alignment_spread.csv", index=False)

    fig, axs = plt.subplots(1, 4, figsize=(18, 4.9), gridspec_kw={"width_ratios": [1, 1, 1, 0.72]})
    ax = axs[0]
    for code, g in d.groupby("code"):
        ax.plot(g.year, g[TARGET], color=REGION_COLORS.get(g.region.iloc[0], MUTED), lw=0.6, alpha=0.45)
    ax.set(xlim=(1990, 2025), ylim=(0, 100), xlabel="calendar year", ylabel="internet users (% of population)",
           title=f"(a) Calendar time: {d.code.nunique()} curves spread\nover two decades")
    _tag(ax, "ITERATION 1  raw representation", MUTED)
    ax = axs[1]
    for code, g in d.groupby("code"):
        ax.plot(g.a, g.ys, color=REGION_COLORS.get(g.region.iloc[0], MUTED), lw=0.6, alpha=0.45)
    xx = np.linspace(-15, 15, 200)
    ax.plot(xx, 100 / (1 + np.exp(-kmed * xx)), color=INK, lw=2.4, label=f"one logistic curve (median k = {kmed:.2f})")
    ax.axvline(0, color=MUTED, ls=":", lw=1)
    ax.set(xlim=(-15, 15), ylim=(0, 110), xlabel="years since own take-off (t − t0)",
           ylabel="internet users (% of own saturation L)",
           title="(b) Own time and own scale:\nthe curves collapse onto one S-curve")
    ax.legend(loc="upper left", fontsize=8)
    _tag(ax, "ITERATION 2  transformed", ACCENT)
    ax = axs[2]
    for r in REGION_ORDER:
        g = d[(d.region == r) & d.a.between(-12, 12)].groupby("a").ys
        m = g.median()[g.size() >= 5]
        ax.plot(m.index, m.values, color=REGION_COLORS[r], lw=2.2, label=f"{r} (t0 median {f[f.region == r].t0_takeoff.median():.0f})")
    ax.plot(xx, 100 / (1 + np.exp(-kmed * xx)), color=INK, lw=1.2, ls="--")
    ax.axvline(0, color=MUTED, ls=":", lw=1)
    ax.set(xlim=(-12, 12), ylim=(0, 110), xlabel="years since own take-off (t − t0)",
           ylabel="regional median (% of own saturation)",
           title="(c) Six regions, six take-off years,\none shape once aligned")
    ax.legend(fontsize=7.2, loc="upper left")
    _tag(ax, "ITERATION 2  transformed", ACCENT)
    ax = axs[3]
    vals = [iqr_cal, iqr_shift, iqr_both]
    labs = ["calendar\nyear", "own\ntake-off", "own take-off\n+ own scale"]
    bars = ax.bar(labs, vals, color=[MUTED, "#8FB8E8", ACCENT], width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}", ha="center", fontsize=10, weight="bold")
    ax.set(ylabel="mean cross-country IQR (pp)", ylim=(0, max(vals) * 1.25),
           title=f"(d) Spread between countries\nfalls {iqr_cal / iqr_both:.1f}-fold")
    ax.tick_params(axis="x", labelsize=8.5)
    fig.suptitle(f"V1c  The divide is a delay: re-indexed to its own take-off year, every country follows one S-curve "
                 f"(cross-country spread {iqr_cal:.0f} → {iqr_both:.0f} pp)", x=0.01, ha="left", fontsize=12.5,
                 weight="bold", y=1.02)
    fig.tight_layout()
    savefig(fig, "V1c_takeoff_alignment.png")
    return iqr_cal, iqr_shift, iqr_both, d.code.nunique()


def main():
    df = load_panel(False)
    sig, bc, rho, p, lr = convergence_stats(df)
    print(sig[["year", "sd", "gini", "gap_eu_af"]].iloc[[0, 5, 10, 15, 20, -1]].round(2).to_string())
    print(f"beta: rho={rho:.2f} p={p:.1e} slope={lr.slope:.4f} n={len(bc)}")
    from common import PROC
    clean = pd.read_parquet(PROC / "panel_iteration2.parquet")
    fits = pd.read_csv(OUT / "p1_logistic_fits.csv")
    print("V1c spread (calendar, shifted, shifted+scaled, n):", fig_v1c_alignment(clean, fits))
    cov, reg, tests = missingness_stats(df)
    for t in tests:
        print(t[0], "median obs", round(t[1].median(), 1), "mis", round(t[2].median(), 1), "p", t[3], len(t[1]), len(t[2]))


if __name__ == "__main__":
    main()

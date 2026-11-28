"""
p1_parikshith_convergence_quality.py  -  Parikshith Aithal K V
V1b  Is the divide closing?  sigma-convergence (dispersion, Gini, regional gap) and beta-convergence
     measured as the share of remaining headroom closed 2014 -> 2024.
V1c  Iteration 2 of workflow W1: each country's curve re-expressed in its own time (years since
     take-off t - t0) and its own scale (share of saturation y/L).  If the divide is a delay on one
     S-curve, the curves must collapse onto one line after this transformation.
V3b  Data quality: coverage by year and region, and a missing-not-at-random test that justifies
     complete cases in iteration 1 and missingness flags + in-fold MICE in iteration 2.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from common import (ACCENT, ACCENT2, BAD, INK, MUTED, OUT, REGION_COLORS, REGION_ORDER, SHORT, TARGET, load_panel,
                    savefig)


def _gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())) if n and x.sum() > 0 else np.nan


def _tag(ax, text, color):
    ax.text(0.0, 1.0, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5, weight="bold",
            color="white", bbox=dict(boxstyle="round,pad=0.25", fc=color, ec="none"),
            zorder=10).set_y(1.0 + 0.16)


def fig_v1b_convergence(df: pd.DataFrame):
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
    peak = sig.loc[sig.sd.idxmax()]
    gmax = sig.loc[sig.gap_eu_af.idxmax()]

    fig, axs = plt.subplots(1, 3, figsize=(16, 5.3), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    ax = axs[0]
    ax.fill_between(sig.year, sig.p10, sig.p90, color=ACCENT, alpha=0.15, label="10th-90th percentile")
    ax.plot(sig.year, sig.p50, color=ACCENT, lw=2.4, label="median country")
    ax.plot(sig.year, sig.gap_eu_af, color=BAD, lw=2, ls="--", label="Europe − Africa median gap")
    ax.annotate(f"gap peaks at {gmax.gap_eu_af:.0f} pp ({int(gmax.year)})", (gmax.year, gmax.gap_eu_af),
                xytext=(1996, 82), fontsize=8.5, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    ax.set(ylim=(0, 100), xlabel="year", ylabel="internet users (%)",
           title=f"(a) The Europe-Africa gap peaked in {int(gmax.year)}\nand is still {sig.gap_eu_af.iloc[-1]:.0f} pp in 2024")
    ax.legend(loc="upper left", fontsize=8)
    ax = axs[1]
    ax.plot(sig.year, sig.sd / sig.sd.max() * 100, color=INK, lw=2.4,
            label=f"absolute spread: s.d. (peak {sig.sd.max():.0f} pp = 100)")
    ax.plot(sig.year, sig.gini / sig.gini.max() * 100, color=ACCENT2, lw=2.4, ls="--",
            label=f"relative inequality: Gini (peak {sig.gini.max():.2f} = 100)")
    ax.axvline(peak.year, color=MUTED, ls=":", lw=1)
    ax.annotate(f"s.d. peaks {int(peak.year)}", (peak.year, 100), xytext=(peak.year - 9, 104), fontsize=8.5,
                color=MUTED, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.set(ylim=(0, 112), xlabel="year", ylabel="index (own maximum = 100)")
    ax.legend(loc="lower center", fontsize=8)
    ax.set_title(f"(b) Relative inequality fell steadily (Gini {sig.gini.iloc[0]:.2f} → {sig.gini.iloc[-1]:.2f});\n"
                 f"absolute spread fell only after {int(peak.year)}")
    ax = axs[2]
    for r in REGION_ORDER:
        g = bc[bc.region == r]
        ax.scatter(g.y14, g.closed * 100, s=22, color=REGION_COLORS[r], alpha=0.85, label=r)
    xx = np.linspace(0, 95, 50)
    ax.plot(xx, (lr.intercept + lr.slope * xx) * 100, color=INK, lw=2, label="OLS fit")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set(xlim=(0, 95), ylim=(-20, 100), xlabel="internet users in 2014 (%)",
           ylabel="share of remaining gap closed 2014→2024 (%)",
           title=f"(c) Headroom closed vs starting level: Spearman ρ = {rho:.2f}\n"
                 f"(p = {p:.0e}, {len(bc)} countries)")
    ax.legend(fontsize=7.5, loc="lower right", ncol=2)
    _tag(axs[0], "ITERATION 1  raw levels", MUTED)
    _tag(axs[1], "ITERATION 1  raw levels", MUTED)
    _tag(axs[2], "ITERATION 2  headroom (transformed)", ACCENT)
    direction = "SMALLER" if rho > 0 else "LARGER"
    fig.suptitle(f"V1b  Is the divide closing? Dispersion has fallen since {int(peak.year)}, but low starters "
                 f"closed a {direction} share of their remaining gap", x=0.01, ha="left", fontsize=12.5, weight="bold")
    fig.tight_layout()
    savefig(fig, "V1b_convergence.png")
    return sig, bc, rho, p, lr


def fig_v3b_missingness(df: pd.DataFrame):
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

    fig = plt.figure(figsize=(16, 7.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.7, 1])
    ax = fig.add_subplot(gs[0, :])
    im = ax.imshow(cov.values, aspect="auto", cmap="Blues", vmin=0, vmax=100)
    ax.grid(False)
    ax.set_yticks(range(len(cov)), cov.index)
    ax.set_xticks(range(0, len(yrs), 5), yrs[::5])
    for j, y in enumerate(yrs):
        if y % 5 == 0 or y >= 2023:
            for i in range(len(cov)):
                v = cov.values[i, j]
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5, color="white" if v > 60 else INK)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label=f"% of all {ncty} countries observed")
    ax.set_title("(a) Coverage by year: broadband starts late, ownership exists only in survey years, 2025 is almost "
                 "empty")
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(reg.values, aspect="auto", cmap="Blues", vmin=0, vmax=100)
    ax.grid(False)
    ax.set_yticks(range(len(reg)), reg.index)
    ax.set_xticks(range(len(reg.columns)), [c.replace(" ", "\n") for c in reg.columns])
    for i in range(reg.shape[0]):
        for j in range(reg.shape[1]):
            v = reg.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8.5, color="white" if v > 60 else INK)
    ax.set_title("(b) Coverage by region, 2010-2025 (% of country-years)")
    ax = fig.add_subplot(gs[1, 1])
    for k, (lab, o, m, p) in enumerate(tests):
        for vals, col, off in [(o, ACCENT, -0.2), (m, MUTED, 0.2)]:
            bp = ax.boxplot(vals, positions=[k + off], widths=0.32, patch_artist=True, showfliers=False)
            bp["boxes"][0].set(facecolor=col + "66", edgecolor=col)
            bp["medians"][0].set(color=INK)
            ax.text(k + off, -6, f"{vals.median():.0f}", ha="center", fontsize=8)
        ax.text(k, 104, f"Mann-Whitney p = {p:.0e}", ha="center", fontsize=8.5)
    ax.set_xticks([0, 1], [t[0] for t in tests])
    ax.scatter([], [], marker="s", color=ACCENT, label="indicator observed")
    ax.scatter([], [], marker="s", color=MUTED, label="indicator missing")
    ax.legend(loc="center right", fontsize=8)
    ax.set(ylim=(-10, 112), ylabel="internet users % of the same country-year",
           title="(c) Missing not at random (2015-24): rows\nwith and without the indicator differ")
    fig.suptitle("V3b  Data quality decides the method: complete cases in iteration 1; missingness flags and in-fold "
                 "MICE in iteration 2", x=0.01, ha="left", fontsize=12.5, weight="bold")
    fig.tight_layout()
    savefig(fig, "V3b_data_quality_missingness.png")
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
    sig, bc, rho, p, lr = fig_v1b_convergence(df)
    print(sig[["year", "sd", "gini", "gap_eu_af"]].iloc[[0, 5, 10, 15, 20, -1]].round(2).to_string())
    print(f"beta: rho={rho:.2f} p={p:.1e} slope={lr.slope:.4f} n={len(bc)}")
    from common import PROC
    clean = pd.read_parquet(PROC / "panel_iteration2.parquet")
    fits = pd.read_csv(OUT / "p1_logistic_fits.csv")
    print("V1c spread (calendar, shifted, shifted+scaled, n):", fig_v1c_alignment(clean, fits))
    cov, reg, tests = fig_v3b_missingness(df)
    for t in tests:
        print(t[0], "median obs", round(t[1].median(), 1), "mis", round(t[2].median(), 1), "p", t[3], len(t[1]), len(t[2]))


if __name__ == "__main__":
    main()

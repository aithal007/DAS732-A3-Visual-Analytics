"""
p1_parikshith_evidence.py  -  Parikshith Aithal K V
Three W1 evidence figures that answer the parts of the question the other figures leave open.

V1d  Iteration 1, WHERE: world map of each country's take-off year t0 (logistic fits from V1).
     The regional delay of V1 is a spatial pattern, not a few outliers.
V1e  Iteration 1 -> 2, IS IT CLOSING: raw measures disagree (Gini falls, the absolute Europe-Africa gap
     peaks in 2014), so iteration 2 measures the gap as headroom (T4): the share of the remaining gap
     each country closed 2014 -> 2024.
V1f  Iteration 2, DID LEAPFROGGING CLOSE THE DIVIDE: with the engineered features of T2 (landline peak,
     capped mobile), countries that skipped landlines but saturated on mobile are compared with
     countries that built landlines.

Run after p1_parikshith_stats.py (logistic fits) and p1_parikshith_convergence_quality.py
(convergence tables):
    python src/p1_parikshith_evidence.py
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from scipy import stats

from common import (ACCENT, BAD, GEO, INK, MUTED, OUT, PROC, REGION_COLORS, REGION_ORDER, TARGET, savefig)

LEAP_LANDLINE_MAX = 5.0      # never built landlines: peak below 5 lines per 100
LEAP_MOBILE_MIN = 80.0       # but saturated on mobile: capped subscriptions at least 80 per 100
BUILDER_LANDLINE_MIN = 20.0  # built a landline network: peak at least 20 lines per 100


def _tag(ax, text, color):
    ax.text(0.0, 1.16, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5, weight="bold",
            color="white", bbox=dict(boxstyle="round,pad=0.25", fc=color, ec="none"), zorder=10)


def _polygons(geom):
    rings = []
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    for poly in polys:
        rings.append(np.asarray(poly[0])[:, :2])          # outer ring only; holes are tiny at this scale
    return rings


# ---------------------------------------------------------------- V1d
def fig_v1d_takeoff_map(fits: pd.DataFrame):
    good = fits[fits.r2 >= 0.9].set_index("code")
    gj = json.load(open(GEO, encoding="utf-8"))
    lo, hi = np.floor(good.t0_takeoff.quantile(0.02)), np.ceil(good.t0_takeoff.quantile(0.98))
    cmap = plt.get_cmap("viridis_r")   # no grey mid-tones: cannot be confused with "no fit"
    fig, ax = plt.subplots(figsize=(14, 6.6))
    fill, grey = [], []
    for f in gj["features"]:
        p = f["properties"]
        if p.get("continent") == "Antarctica" or p.get("iso_a3") == "ATA":
            continue
        code = p.get("iso_a3") if p.get("iso_a3") not in (None, "-99") else p.get("adm0_a3")
        rings = _polygons(f["geometry"])
        if code in good.index:
            v = (good.loc[code, "t0_takeoff"] - lo) / (hi - lo)
            fill += [(r, cmap(float(np.clip(v, 0, 1)))) for r in rings]
        else:
            grey += rings
    ax.add_collection(PolyCollection(grey, facecolors="white", edgecolors="#9AA1A9", linewidths=0.4, hatch="////"))
    ax.add_collection(PolyCollection([r for r, _ in fill], facecolors=[c for _, c in fill],
                                     edgecolors="white", linewidths=0.3))
    ax.set(xlim=(-170, 190), ylim=(-58, 84), aspect=1.15)
    ax.axis("off")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(lo, hi))
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02, aspect=50)
    cb.set_label("take-off year t0 (inflection of the fitted logistic S-curve); hatched = no reliable fit (R² < 0.9) or no data")
    med = good.groupby("region").t0_takeoff.median()
    txt = "Regional median take-off\n" + "\n".join(f"{r}: {med[r]:.0f}" for r in REGION_ORDER if r in med)
    ax.text(-168, -52, txt, fontsize=9, va="bottom", ha="left", color=INK,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#D0D7DE"))
    gap = med["Africa"] - med["Europe"]
    fig.suptitle(f"V1d  Take-off is a spatial pattern: Europe took off first (median {med['Europe']:.0f}), the Americas "
                 f"and Asia 6-8 years later, most of Africa {gap:.0f} years later ({len(good)} countries)", x=0.01, ha="left", fontsize=12.5, weight="bold")
    fig.tight_layout()
    savefig(fig, "V1d_takeoff_map.png")
    return len(good), med


# ---------------------------------------------------------------- V1e
def fig_v1e_headroom(sig: pd.DataFrame, bc: pd.DataFrame):
    rho, p = stats.spearmanr(bc.y14, bc.closed)
    lr = stats.linregress(bc.y14, bc.closed)
    gmax = sig.loc[sig.gap_eu_af.idxmax()]
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axs[0]
    ax.plot(sig.year, sig.gini / sig.gini.max() * 100, color=ACCENT, lw=2.6,
            label=f"relative inequality: Gini ({sig.gini.iloc[0]:.2f} → {sig.gini.iloc[-1]:.2f})")
    ax.plot(sig.year, sig.gap_eu_af / sig.gap_eu_af.max() * 100, color=BAD, lw=2.6, ls="--",
            label=f"absolute gap: Europe − Africa median (peak {gmax.gap_eu_af:.0f} pp)")
    ax.axvline(gmax.year, color=MUTED, ls=":", lw=1)
    ax.annotate(f"absolute gap peaks {int(gmax.year)},\nstill {sig.gap_eu_af.iloc[-1]:.0f} pp in 2024",
                (gmax.year, 100), xytext=(2015.3, 52), fontsize=9,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    ax.set(ylim=(0, 108), xlabel="year", ylabel="index (own maximum = 100)",
           title="(a) Raw levels: the two measures of the gap disagree")
    ax.legend(loc="lower left", fontsize=8.5)
    ax = axs[1]
    for r in REGION_ORDER:
        g = bc[bc.region == r]
        ax.scatter(g.y14, g.closed * 100, s=30, color=REGION_COLORS[r], alpha=0.85, label=r,
                   edgecolor="white", linewidth=0.5)
    xx = np.linspace(0, 95, 50)
    ax.plot(xx, (lr.intercept + lr.slope * xx) * 100, color=INK, lw=2, label="OLS fit")
    ax.axhline(0, color=MUTED, lw=0.8)
    med = bc.groupby("region").closed.median() * 100
    ax.text(0.98, 0.04, "median share closed\n" + "\n".join(f"{r}: {med[r]:.0f}%" for r in REGION_ORDER if r in med),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#D0D7DE"))
    ax.set(xlim=(0, 95), ylim=(-20, 100), xlabel="internet users in 2014 (%)",
           ylabel="share of remaining gap closed 2014→2024 (%)",
           title=f"(b) Headroom h (T4): low starters close the smallest share\n"
                 f"(Spearman ρ = {rho:+.2f}, p = {p:.0e}, {len(bc)} countries)")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    _tag(axs[0], "ITERATION 1  raw levels", MUTED)
    _tag(axs[1], "ITERATION 2  headroom (transformed)", ACCENT)
    fig.suptitle("V1e  Is the divide closing? Only in relative terms: the countries furthest behind close the "
                 "smallest share of their remaining gap", x=0.01, ha="left", fontsize=12.5, weight="bold", y=1.0)
    fig.tight_layout()
    savefig(fig, "V1e_headroom_closing.png")
    return rho, p, med


# ---------------------------------------------------------------- V1f
def fig_v1f_leapfrog(clean: pd.DataFrame):
    s = clean[clean.year == 2024].dropna(subset=[TARGET, "landline_peak", "mobile_capped"]).copy()
    s["group"] = np.select(
        [(s.landline_peak < LEAP_LANDLINE_MAX) & (s.mobile_capped >= LEAP_MOBILE_MIN),
         s.landline_peak >= BUILDER_LANDLINE_MIN],
        ["leapfrogged: <5 landlines, ≥80 mobile", "built landlines: peak ≥20"], "in between")
    rho, p = stats.spearmanr(s.landline_peak, s[TARGET])
    rho_m, p_m = stats.spearmanr(s.mobile_capped, s[TARGET])
    af = s[s.region == "Africa"]
    rho_af, p_af = stats.spearmanr(af.landline_peak, af[TARGET])
    lf = s[s.group.str.startswith("leap")]
    bl = s[s.group.str.startswith("built")]
    mw = stats.mannwhitneyu(lf[TARGET], bl[TARGET]).pvalue
    s.to_csv(OUT / "p1_leapfrog_2024.csv", index=False)

    fig, axs = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axs[0]
    for r in REGION_ORDER:
        g = s[s.region == r]
        ax.scatter(g.landline_peak, g[TARGET], s=18 + g.mobile_capped * 0.5, color=REGION_COLORS[r], alpha=0.85,
                   edgecolor="white", linewidth=0.5, label=r)
    ax.set_xscale("log")
    ax.axvspan(0.1, LEAP_LANDLINE_MAX, color=ACCENT, alpha=0.07)
    ax.text(0.13, 3, "never built landlines\n(peak < 5 per 100)", fontsize=8.5, color=ACCENT)
    for name in ["Uganda", "Tanzania", "Kenya", "Japan"]:
        row = s[s.entity == name]
        if len(row):
            ax.annotate(name, (row.landline_peak.iloc[0], row[TARGET].iloc[0]), xytext=(4, 3),
                        textcoords="offset points", fontsize=8, color=INK)
    ax.set(xlim=(0.1, 160), ylim=(0, 105), xlabel="landline peak, lines per 100 (log scale; engineered in T2)",
           ylabel="internet users 2024 (%)",
           title=f"(a) Internet use follows the landline network a country built\n"
                 f"Spearman ρ = {rho:.2f} (all), {rho_af:.2f} within Africa; capped mobile only {rho_m:.2f}")
    ax.legend(fontsize=8, loc="lower right", ncol=2, title="dot size = capped mobile per 100", title_fontsize=8)
    ax = axs[1]
    order = ["leapfrogged: <5 landlines, ≥80 mobile", "built landlines: peak ≥20"]
    cols = [ACCENT, MUTED]
    data = [lf[TARGET].values, bl[TARGET].values]
    bp = ax.boxplot(data, positions=[0, 1], widths=0.45, patch_artist=True, showfliers=False)
    for b, c in zip(bp["boxes"], cols):
        b.set(facecolor=c + "44", edgecolor=c)
    for m in bp["medians"]:
        m.set(color=INK, lw=2)
    rng = np.random.default_rng(42)
    for i, (v, c) in enumerate(zip(data, cols)):
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=16, color=c, alpha=0.8, zorder=3)
        ax.text(i, -9, f"median {np.median(v):.0f}%  (n = {len(v)})", ha="center", fontsize=9, weight="bold")
    ax.set_xticks([0, 1], ["leapfrogged\n<5 landlines, ≥80 mobile", "built landlines\npeak ≥20 per 100"])
    ax.set(ylim=(-14, 105), ylabel="internet users 2024 (%)",
           title=f"(b) Mobile saturation without landlines did not\nbring people online (Mann–Whitney p = {mw:.0e})")
    _tag(axs[0], "ITERATION 2  engineered features (T2)", ACCENT)
    _tag(axs[1], "ITERATION 2  test", ACCENT)
    fig.suptitle("V1f  Leapfrogging did not close the divide: countries that skipped landlines are saturated on "
                 "mobile yet less than half online", x=0.01, ha="left", fontsize=12.5, weight="bold", y=1.0)
    fig.tight_layout()
    savefig(fig, "V1f_leapfrog_test.png")
    res = dict(n=len(s), rho_peak=rho, p_peak=p, rho_mobile=rho_m, p_mobile=p_m, rho_africa=rho_af, p_africa=p_af,
               n_africa=len(af), n_leap=len(lf), med_leap=lf[TARGET].median(), n_built=len(bl),
               med_built=bl[TARGET].median(), p_mw=mw, leap_africa=int((lf.region == "Africa").sum()))
    pd.Series(res).to_csv(OUT / "p1_leapfrog_test.csv", header=["value"])
    return res


def main():
    fits = pd.read_csv(OUT / "p1_logistic_fits.csv")
    n, med = fig_v1d_takeoff_map(fits)
    print(f"V1d: {n} countries; regional median t0:", med.round(1).to_dict())
    sig = pd.read_csv(OUT / "p1_sigma_convergence.csv")
    bc = pd.read_csv(OUT / "p1_beta_convergence.csv", index_col=0)
    rho, p, medc = fig_v1e_headroom(sig, bc)
    print(f"V1e: rho={rho:.2f} p={p:.1e}; median share closed:", medc.round(0).to_dict())
    clean = pd.read_parquet(PROC / "panel_iteration2.parquet")
    res = fig_v1f_leapfrog(clean)
    print("V1f:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()})


if __name__ == "__main__":
    main()

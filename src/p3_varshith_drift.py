"""
p3_varshith_drift.py  -  Varshith M Gowda
V13  Drift chart for the ML workflow: Population Stability Index (PSI) of the target and the main
     features per year against the training-era window (2015-2020), with the usual action
     thresholds 0.1 (evaluate) and 0.2 (retrain / re-calibrate). It explains why the 2021-25
     forecast intervals under-cover and motivates the conformal re-calibration in V12/V12b.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ACCENT, ACCENT2, BAD, INK, MUTED, OUT, PROC, TARGET, savefig


def psi(base: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    base, cur = base[~np.isnan(base)], cur[~np.isnan(cur)]
    if len(base) < 30 or len(cur) < 20:
        return np.nan
    edges = np.unique(np.quantile(base, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    pb = np.histogram(base, edges)[0] / len(base)
    pc = np.histogram(cur, edges)[0] / len(cur)
    pb, pc = np.clip(pb, 1e-4, None), np.clip(pc, 1e-4, None)
    return float(np.sum((pc - pb) * np.log(pc / pb)))


def fig_v13_drift(clean: pd.DataFrame, base_years=(2015, 2020)):
    # sim_per_owner is left out: ownership surveys cover ~10% of country-years, so its PSI is noise
    feats = [TARGET, "log1p_broadband", "mobile_capped", "telephone_lines_per100", "landline_decline"]
    base = clean[(clean.year >= base_years[0]) & (clean.year <= base_years[1])]
    rows = []
    for yr in range(2010, 2025):
        cur = clean[clean.year == yr]
        for f in feats:
            rows.append({"year": yr, "feature": f, "psi": psi(base[f].values, cur[f].values)})
    dr = pd.DataFrame(rows)
    dr.to_csv(OUT / "p3_psi_drift.csv", index=False)
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    ax.axhspan(0, 0.1, color="#2DA44E", alpha=0.08)
    ax.axhspan(0.1, 0.2, color="#E69F00", alpha=0.10)
    ax.axhspan(0.2, 3, color=BAD, alpha=0.06)
    ax.axhline(0.1, color="#E69F00", ls="--", lw=1)
    ax.axhline(0.2, color=BAD, ls="--", lw=1)
    ax.axvspan(base_years[0] - 0.5, base_years[1] + 0.5, color=ACCENT, alpha=0.08)
    ax.text(np.mean(base_years), 1.42, "reference baseline\n(training-era window)", ha="center", fontsize=8,
            color=ACCENT)
    ax.axvline(2020.5, color=INK, lw=1.2)
    ax.text(2020.6, 1.42, "forecast holdout\n2021-2025\n(post-COVID)", fontsize=8, va="top")
    styles = {TARGET: (INK, 2.8, "-"), "log1p_broadband": (ACCENT, 2, "-"), "mobile_capped": (ACCENT2, 2, "-"),
              "telephone_lines_per100": (MUTED, 1.6, "-"), "landline_decline": ("#009E73", 1.6, "--")}
    for f, g in dr.groupby("feature"):
        c, lw, ls = styles[f]
        lab = "TARGET: internet users %" if f == TARGET else f
        ax.plot(g.year, g.psi.clip(upper=1.5), color=c, lw=lw, ls=ls, marker="o", ms=3, label=lab)
    ax.set_xticks(range(2010, 2025, 2))
    ax.set(ylim=(0, 1.5), xlim=(2009.5, 2024.5), xlabel="year",
           ylabel="Population Stability Index vs baseline (clipped at 1.5)")
    ax.text(2009.7, 0.04, "PSI < 0.1 keep model", fontsize=7.8, color="#1A7F37")
    ax.text(2009.7, 0.13, "0.1-0.2 evaluate", fontsize=7.8, color="#9A6700")
    ax.text(2009.7, 0.23, "≥ 0.2 retrain / re-calibrate", fontsize=7.8, color=BAD)
    ax.legend(loc="upper left", bbox_to_anchor=(0.12, 1.0), fontsize=8, ncol=2)
    post = dr[(dr.year >= 2022) & (dr.feature == TARGET)].psi.mean()
    feat_post = dr[(dr.year >= 2021) & (dr.year <= 2022) & (dr.feature != TARGET)].psi.max()
    ax.set_title(f"V13  Drift chart: after 2020 the TARGET drifts past 0.2 (mean PSI {post:.2f} in 2022-24) while the\n"
                 f"inputs stay below 0.2 until 2023 (max {feat_post:.2f} in 2021-22) → the input→output relation shifted",
                 loc="left", fontsize=11.5)
    fig.tight_layout()
    savefig(fig, "V13_psi_drift_chart.png")
    return dr


def main():
    clean = pd.read_parquet(PROC / "panel_iteration2.parquet")
    dr = fig_v13_drift(clean)
    print(dr.pivot(index="year", columns="feature", values="psi").round(2).tail(8).to_string())


if __name__ == "__main__":
    main()

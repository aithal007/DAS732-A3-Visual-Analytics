"""
build_workflow_diagrams.py - writes the TikZ sources of the workflow diagrams (report/diagrams/*.tex).

  V0_workflow_architecture.tex  the main question answered by three unrolled workflows (W1, W2, W3)
  W1_workflow_statistics.tex    Parikshith  - statistical workflow
  W2_workflow_unsupervised.tex  Sindhoor    - unsupervised workflow
  W3_workflow_supervised.tex    Varshith    - supervised workflow

Every per-member diagram is the unrolled Keim loop D -> V -> M -> K for two iterations, with
  * the question and the hypotheses written on the diagram, each with its verdict;
  * the real figures plugged into the V stages (thumbnails from images/), numbered in the order the
    analyst looked at them, so the diagram doubles as the record of the analysis path;
  * one red edge labelled "Feedback loop" carrying the data transformation T between the iterations.
Compile with  python src/build_diagrams.py  (pdflatex + pdftoppm).
    python src/build_workflow_diagrams.py
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "report" / "diagrams"

PREAMBLE = r"""\documentclass[tikz,border=3pt]{standalone}
\usetikzlibrary{positioning,arrows.meta,calc}
\usepackage[scaled=0.95]{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{amssymb,amsmath}
\definecolor{cD}{HTML}{2A78D6}\definecolor{cV}{HTML}{E69F00}\definecolor{cM}{HTML}{009E73}
\definecolor{cK}{HTML}{CC79A7}\definecolor{cL}{HTML}{C0392B}\definecolor{ink}{HTML}{1F2328}
\definecolor{mut}{HTML}{57606A}\definecolor{pan}{HTML}{F3F5F7}\definecolor{ok}{HTML}{1A7F37}
\definecolor{no}{HTML}{C0392B}\definecolor{part}{HTML}{B35900}
\tikzset{
  box/.style={draw=#1, fill=#1!6, rounded corners=3pt, line width=0.8pt},
  ttl/.style={font=\bfseries\scriptsize, text=ink, anchor=north west, inner sep=2pt},
  txt/.style={font=\scriptsize, text=ink, anchor=north west, align=left, inner sep=2pt, text width=#1},
  capt/.style={font=\scriptsize, text=ink, align=center, inner sep=1pt},
  chip/.style={draw=#1, fill=white, rounded corners=2pt, line width=0.5pt, font=\scriptsize, text=ink,
               inner sep=1.5pt, align=left},
  stepn/.style={circle, fill=ink, text=white, font=\bfseries\scriptsize, inner sep=0pt, minimum size=0.36cm},
  badge/.style={rounded corners=2pt, fill=#1, text=white, font=\bfseries\scriptsize, inner sep=2pt},
  arr/.style={-{Stealth[length=2.2mm]}, line width=0.9pt, color=#1},
  verdict/.style={rounded corners=2pt, fill=#1, text=white, font=\bfseries\scriptsize, inner sep=1.6pt},
}
\newcommand{\img}[3]{\includegraphics[width=#2cm,height=#3cm,keepaspectratio]{../../images/#1}}
"""

W = 18.1          # diagram width (cm) = IEEE \textwidth, so the figure is printed at 100 %
VERDICT = {"ok": ("ok", "supported"), "no": ("no", "rejected"), "part": ("part", "refined")}


def esc(s: str) -> str:
    return s  # specs are written in LaTeX already


def thumbs(items, x0, x1, ytop, ybot, cols=None):
    """Grid of figure thumbnails with step badges and captions inside the box [x0,x1] x [ybot,ytop]."""
    n = len(items)
    cols = cols or (1 if n == 1 else 2 if n <= 4 else 3)
    rows = -(-n // cols)
    gap, caph = 0.18, 0.62
    w = (x1 - x0 - (cols + 1) * gap) / cols
    h = (ytop - ybot - rows * (caph + gap)) / rows
    out = []
    for i, (fname, step, caption) in enumerate(items):
        r, c = divmod(i, cols)
        in_row = min(cols, n - r * cols)
        off = (cols - in_row) * (w + gap) / 2
        cx = x0 + gap + off + c * (w + gap) + w / 2
        cy = ytop - r * (h + caph + gap) - h / 2
        out.append(rf"\node[inner sep=0.6pt, draw=mut!45, line width=0.3pt, fill=white] (th{step}) at ({cx:.2f},{cy:.2f}) "
                   rf"{{\img{{{fname}}}{{{w:.2f}}}{{{h:.2f}}}}};")
        out.append(rf"\node[stepn] at ([xshift=1pt,yshift=-1pt]th{step}.north west) {{{step}}};")
        out.append(rf"\node[capt, text width={w:.2f}cm, anchor=north] at ({cx:.2f},{cy - h / 2 - 0.04:.2f}) {{{caption}}};")
    return "\n".join(out)


def row(y0, y1, tag, D, V, M, K, it):
    """One iteration: D -> V -> M -> K boxes between y0 (bottom) and y1 (top)."""
    xs = {"D": (0.0, 2.3), "V": (2.6, 10.55), "M": (10.85, 14.0), "K": (14.3, W)}
    L = []
    L.append(rf"\node[badge=ink, anchor=south west] at (0,{y1 + 0.08:.2f}) {{{tag}}};")
    for key, color, (title, body) in [("D", "cD", D), ("M", "cM", M), ("K", "cK", K)]:
        a, b = xs[key]
        L.append(rf"\draw[box={color}] ({a},{y0}) rectangle ({b},{y1});")
        L.append(rf"\node[badge={color}, anchor=north west] at ({a + 0.08},{y1 - 0.08}) {{{key}{it}}};")
        L.append(rf"\node[ttl, text width={b - a - 0.75:.2f}cm, align=left] at ({a + 0.62},{y1 - 0.05}) {{{title}}};")
        L.append(rf"\node[txt={b - a - 0.25:.2f}cm] at ({a + 0.08},{y1 - 0.62}) {{{body}}};")
    a, b = xs["V"]
    vtitle, vitems, vcols = V
    L.append(rf"\draw[box=cV] ({a},{y0}) rectangle ({b},{y1});")
    L.append(rf"\node[badge=cV, anchor=north west] at ({a + 0.08},{y1 - 0.08}) {{V{it}}};")
    L.append(rf"\node[ttl, text width={b - a - 0.75:.2f}cm, align=left] at ({a + 0.62},{y1 - 0.05}) {{{vtitle}}};")
    L.append(thumbs(vitems, a, b, y1 - 0.55, y0 + 0.05, vcols))
    ym = (y0 + y1) / 2
    for (p, q, c) in [("D", "V", "cD"), ("V", "M", "cV"), ("M", "K", "cM")]:
        L.append(rf"\draw[arr={c}] ({xs[p][1]},{ym:.2f}) -- ({xs[q][0]},{ym:.2f});")
    return "\n".join(L), xs


def workflow(spec) -> str:
    H = 23.4
    L = [PREAMBLE, r"\begin{document}", r"\begin{tikzpicture}[x=1cm,y=1cm]"]
    # ---------------------------------------------------------------- header: question + hypotheses
    L.append(rf"\node[anchor=north west, font=\bfseries\small, text=ink, inner sep=0] at (0,{H}) {{{spec['title']}}};")
    L.append(rf"\node[anchor=north east, font=\scriptsize, text=mut, inner sep=0] at ({W},{H - 0.02}) {{{spec['owner']}}};")
    qy1, qy0 = H - 0.5, H - 1.55
    L.append(rf"\draw[box=ink, fill=pan] (0,{qy0}) rectangle ({W},{qy1});")
    L.append(rf"\node[badge=ink, anchor=north west] at (0.08,{qy1 - 0.08}) {{{spec['qid']}}};")
    L.append(rf"\node[txt={W - 1.3}cm, font=\small] at (0.75,{qy1 - 0.05}) {{\textbf{{Question.}} {spec['question']}}};")
    hy1 = qy0 - 0.15
    nh = len(spec["hyp"])
    hy0 = hy1 - 0.45 - 0.72 * nh
    L.append(rf"\draw[box=ink, fill=white] (0,{hy0:.2f}) rectangle ({W},{hy1:.2f});")
    L.append(rf"\node[ttl] at (0.08,{hy1 - 0.04:.2f}) {{Hypotheses, the test that decides each one, and the verdict}};")
    for i, (hid, text, kind, ev) in enumerate(spec["hyp"]):
        y = hy1 - 0.78 - i * 0.72
        col, word = VERDICT[kind]
        L.append(rf"\node[anchor=west, font=\bfseries\scriptsize, text=ink] at (0.1,{y:.2f}) {{{hid}}};")
        L.append(rf"\node[anchor=west, font=\scriptsize, text=ink, text width=7.0cm] at (0.95,{y:.2f}) {{{text}}};")
        L.append(rf"\node[verdict={col}, anchor=west, minimum width=1.55cm, align=center] at (8.15,{y:.2f}) {{{word}}};")
        L.append(rf"\node[anchor=west, font=\scriptsize, text=ink!80, text width=8.1cm] at (9.85,{y:.2f}) {{{ev}}};")
    # ---------------------------------------------------------------- iteration rows and feedback band
    top1 = hy0 - 0.55
    rowh = spec.get("rowh", 6.55)
    r1, xs = row(top1 - rowh, top1, r"ITERATION 1 \textmd{\, raw data}", *spec["it1"], it=1)
    L.append(r1)
    band1 = top1 - rowh - 0.35
    band0 = band1 - spec.get("bandh", 1.55)
    L.append(rf"\draw[box=cL, fill=cL!5, line width=1.1pt] (0,{band0:.2f}) rectangle ({W},{band1:.2f});")
    L.append(rf"\node[badge=cL, anchor=north west] at (0.08,{band1 - 0.08:.2f}) {{T}};")
    L.append(rf"\node[ttl, text=cL] at (0.5,{band1 - 0.05:.2f}) {{{spec['T'][0]}}};")
    L.append(rf"\node[txt={W - 0.3}cm] at (0.08,{band1 - 0.5:.2f}) {{{spec['T'][1]}}};")
    kx = (xs["K"][0] + xs["K"][1]) / 2
    L.append(rf"\draw[cL, line width=1.6pt, -{{Stealth[length=3mm]}}] ({kx:.2f},{top1 - rowh:.2f}) -- ({kx:.2f},{band1:.2f});")
    L.append(rf"\node[fill=white, draw=cL, line width=0.8pt, rounded corners=2pt, font=\bfseries\scriptsize, text=cL, "
             rf"inner sep=2pt, anchor=east] at ({kx - 0.12:.2f},{(top1 - rowh + band1) / 2:.2f}) {{Feedback loop}};")
    top2 = band0 - 0.75
    r2, _ = row(top2 - rowh, top2, r"ITERATION 2 \textmd{\, transformed data}", *spec["it2"], it=2)
    L.append(r2)
    dx = (xs["D"][0] + xs["D"][1]) / 2
    L.append(rf"\draw[cL, line width=1.6pt, -{{Stealth[length=3mm]}}] ({dx:.2f},{band0:.2f}) -- ({dx:.2f},{top2:.2f});")
    # ---------------------------------------------------------------- analysis path + exit to synthesis
    py = top2 - rowh - 0.25
    L.append(rf"\node[txt={W - 0.1}cm, anchor=north west] at (0,{py:.2f}) {{\textbf{{Analysis path.}} {spec['path']}}};")
    L += [spec.get("extra", ""), r"\end{tikzpicture}", r"\end{document}"]
    return "\n".join(L)


# ======================================================================================== specs
W1 = dict(
    title=r"Workflow W1 \textendash{} Statistical analysis: when did countries get online, and is the divide closing?",
    owner="Parikshith Aithal K V", qid="Q1",
    question=r"When did each country take off, is the gap between countries closing, and do the indicators "
             r"measure what they appear to measure?",
    hyp=[("H1.1", r"The divide is a time delay on one common S-curve, not a different curve", "ok",
          r"V1: take-off year differs by region ($p=4\times10^{-17}$); V1c: aligned curves collapse, spread 32$\rightarrow$10 pp"),
         ("H1.2", r"The divide is closing", "part",
          r"V1b: relative gap closes (Gini 0.81$\rightarrow$0.19); low starters close the \emph{smallest} share of headroom ($\rho=+0.52$)"),
         ("H1.3", r"Mobile subscriptions count connected people", "no",
          r"V3: 1.34 SIMs per owner; $\rho$ 0.52 vs 0.89 for ownership; Fiji 111$\rightarrow$574 = recording artefact"),
         ("H1.4", r"Internet use rises linearly with fixed broadband", "no",
          r"V2: floor below 1 per 100 (29\% of rows); segmented fit, $F=21.4$, $p=6\times10^{-10}$")],
    it1=[(r"Raw OWID panel", r"217 countries, 1990--2025\\complete cases only\\coverage + MNAR test"),
         (r"Visual exploration of raw levels", [
             ("V3b_data_quality_missingness.png", 1, r"\textbf{V3b} coverage; ownership missing not at random"),
             ("V1_diffusion_scurves.png", 2, r"\textbf{V1} same S-curve, take-off 6--12 yr later"),
             ("V2_floor_effect_decoupling.png", 3, r"\textbf{V2} broadband floor; mobile decouples"),
             ("V3_multisim_inflation.png", 4, r"\textbf{V3} SIMs $\neq$ people; Fiji spike")], 2),
         (r"Statistical models", r"logistic fit $y=L/(1+e^{-k(t-t_0)})$\\Kruskal--Wallis + Holm\\segmented regression, $F$-test\\bootstrap Spearman CIs\\Mann--Whitney MNAR test"),
         (r"Inferences $K_1$", r"\textbf{K1.1} Fiji 2023--24 = artefact\\\textbf{K1.2} subscriptions overstate people\\\textbf{K1.3} calendar-time levels mix \emph{when} with \emph{how far}: a delay looks like a gap\\\textbf{K1.4} broadband floor; data MNAR")],
    T=(r"Feedback loop: transformation $T$ (a change of representation, not a filter)",
       r"\textbf{T1} artefact rules applied to every country (10 cells, shared core) \quad "
       r"\textbf{T2} per-owner normalisation: SIMs per owner, mobile capped at 100 \quad "
       r"\textbf{T3} \emph{own time and own scale}: each curve re-indexed to years since its take-off $t-t_0$ and "
       r"divided by its saturation $L$ \quad \textbf{T4} \emph{headroom}: share of the remaining gap closed, "
       r"$h=(y_{2024}-y_{2014})/(100-y_{2014})$ \quad \textbf{T5} $\log(1+\text{broadband})$ and a floor flag"),
    it2=[(r"Transformed panel", r"artefacts removed\\aligned time $t-t_0$\\headroom $h$\\per-owner features"),
         (r"Visual exploration of the transformed data", [
             ("V9_feedback_transformation.png", 5, r"\textbf{V9} what $T$ changed: removed cells, new features"),
             ("V1c_takeoff_alignment.png", 6, r"\textbf{V1c} aligned at own take-off: one S-curve"),
             ("V1b_convergence.png", 7, r"\textbf{V1b} headroom: low starters close less")], 2),
         (r"Statistics on $D_2$", r"cross-country IQR in calendar vs aligned time\\$\sigma$-convergence, Gini\\$\beta$-convergence on headroom (OLS, Spearman)"),
         (r"Knowledge $K_2$", r"\textbf{H1.1} delay of 6--12 years on one curve\\\textbf{H1.2} closing in relative terms only; the furthest behind close least\\$\rightarrow$ feeds the synthesis (V14) and W3's features")],
    path=r"\textcircled{\scriptsize 1} check what is observed (V3b) $\rightarrow$ \textcircled{\scriptsize 2} fit and compare take-off years (V1) "
         r"$\rightarrow$ \textcircled{\scriptsize 3} test the broadband relation and its change over time (V2) $\rightarrow$ "
         r"\textcircled{\scriptsize 4} test what mobile subscriptions count, which exposes Fiji (V3) $\rightarrow$ $K_1$ $\rightarrow$ $T$ "
         r"$\rightarrow$ \textcircled{\scriptsize 5} verify the transformation (V9) $\rightarrow$ \textcircled{\scriptsize 6} test the "
         r"delay hypothesis in aligned time (V1c) $\rightarrow$ \textcircled{\scriptsize 7} test closing in headroom terms (V1b) $\rightarrow$ synthesis V14.",
)

W2 = dict(
    title=r"Workflow W2 \textendash{} Unsupervised mining: which kinds of countries exist, and which fit none?",
    owner="Sindhoor Ganapathi Hegde", qid="Q2",
    question=r"Which kinds of countries exist, how stable are those kinds, and which countries do not fit any of them?",
    hyp=[("H2.1", r"Countries form stable connectivity types in the 2024 snapshot", "part",
          r"V5, V5b: $k=3$, silhouette 0.33, bootstrap ARI 0.64; V5c: 15 of 135 on a border"),
         ("H2.2", r"The top anomalies are genuinely unusual countries", "part",
          r"V6b: 4 persistent every year (structural: Monaco, Hong Kong); 16 transient (artefact candidates: Fiji)"),
         ("H2.3", r"Countries at the same level today share the same history (A1 Fig.~5)", "no",
          r"V8, V8d: 37 of 133 change group, ARI 0.30; movers had weaker iteration-1 silhouettes (median 0.28 vs 0.39)"),
         ("H2.4", r"Grouping by path orders countries by adoption timing", "part",
          r"V8c: medians 2001 / 2014 / 2023, but DTW groups by shape ($\eta^2$ 0.59 vs 0.82)")],
    it1=[(r"2024 snapshot", r"135 complete-case countries\\5 z-scored indicators\\raw data, no imputation"),
         (r"Visual exploration of the snapshot", [
             ("V4_pca_biplot.png", 1, r"\textbf{V4} PCA: 82\,\% on two axes"),
             ("V5_cluster_validation_profiles.png", 2, r"\textbf{V5} $k$ scan: $k=3$"),
             ("V5b_bootstrap_coassignment.png", 3, r"\textbf{V5b} stability, ARI 0.64"),
             ("V5c_silhouette_knife.png", 4, r"\textbf{V5c} 15 border countries"),
             ("V6b_anomaly_persistence.png", 5, r"\textbf{V6b} persistent vs transient"),
             ("V6c_anomaly_why.png", 6, r"\textbf{V6c} why each is unusual")], 3),
         (r"Mining models", r"PCA\\$k$-means, GMM, Ward, $k=2..8$\\silhouette, bootstrap ARI (200)\\Isolation Forest + LOF (V6)\\per-year Isolation Forest"),
         (r"Inferences $K_1$", r"\textbf{K1.1} clusters are moderately stable; mixed vs legacy blur\\\textbf{K1.2} Fiji is a \emph{transient} anomaly = artefact; Monaco, Hong Kong are structural\\\textbf{K1.3} one year cannot show a path: same level, different histories")],
    T=(r"Feedback loop: transformation $T$ (a new data representation and a new distance, not a filter)",
       r"\textbf{T1} transient anomalies become artefact rules applied to every country (Fiji 2023--24 removed) \quad "
       r"\textbf{T2} snapshot $\rightarrow$ trajectory: each country becomes its 26-year internet curve 2000--2025 "
       r"(181 countries, interior gaps interpolated, 225 cells) \quad \textbf{T3} Euclidean $\rightarrow$ dynamic time warping "
       r"distance (Sakoe--Chiba band 5 years), so the same S-curve shifted in time counts as similar \quad "
       r"\textbf{T4} average-linkage hierarchy instead of centroids"),
    it2=[(r"Trajectory panel", r"181 curves, 2000--2025\\artefacts removed\\$181\times181$ DTW matrix"),
         (r"Visual exploration of the trajectories", [
             ("V8a_dtw_alignment.png", 7, r"\textbf{V8a} why DTW: shifted curves"),
             ("V8b_dtw_dendrogram.png", 8, r"\textbf{V8b} hierarchy, $k=3$ cut"),
             ("V8_trajectory_archetypes.png", 9, r"\textbf{V8} three path archetypes"),
             ("V8c_takeoff_ridgeline.png", 10, r"\textbf{V8c} 50\,\% crossing by group"),
             ("V8d_iteration_embedding.png", 11, r"\textbf{V8d} who moved between iterations")], 3),
         (r"Trajectory mining", r"DTW + average linkage\\$k$ by silhouette\\cophenetic correlation\\Hungarian matching, ARI\\$\eta^2$ of take-off year"),
         (r"Knowledge $K_2$", r"\textbf{H2.3} archetypes are paths: late starters, S-curve majority, early saturators\\37 of 133 regrouped, silhouette 0.52 vs 0.35\\$\rightarrow$ feeds the synthesis (V14)")],
    path=r"\textcircled{\scriptsize 1} structure of the snapshot (V4) $\rightarrow$ \textcircled{\scriptsize 2} choose $k$ (V5) $\rightarrow$ "
         r"\textcircled{\scriptsize 3} test stability (V5b) $\rightarrow$ \textcircled{\scriptsize 4} find border countries (V5c) $\rightarrow$ "
         r"\textcircled{\scriptsize 5} anomalies over time (V6, V6b) $\rightarrow$ \textcircled{\scriptsize 6} explain them (V6c) $\rightarrow$ $K_1$ "
         r"$\rightarrow$ $T$ $\rightarrow$ \textcircled{\scriptsize 7} justify DTW (V8a) $\rightarrow$ \textcircled{\scriptsize 8} cut the hierarchy (V8b) "
         r"$\rightarrow$ \textcircled{\scriptsize 9} read the archetypes (V8) $\rightarrow$ \textcircled{\scriptsize 10} check timing (V8c) $\rightarrow$ "
         r"\textcircled{\scriptsize 11} compare iterations (V8d) $\rightarrow$ synthesis V14.",
)

W3 = dict(
    title=r"Workflow W3 \textendash{} Supervised learning: can infrastructure predict and forecast who is online?",
    owner="Varshith M Gowda", qid="Q3",
    question=r"How well does infrastructure predict internet use in countries a model has never seen and in years it "
             r"has not seen, and how far can the forecast be trusted?",
    hyp=[("H3.1", r"Raw infrastructure predicts unseen countries well", "part",
          r"V7: $R^2=0.86$, but broadband carries almost all weight and errors are 12--14 pp above 10\,\% adoption"),
         ("H3.2", r"History features and in-fold imputation lower error on identical rows", "ok",
          r"V10: RMSE 11.77$\rightarrow$10.16 on the same 4{,}089 rows ($-14$\,\%); largest gain in Asia, Africa"),
         ("H3.3", r"A model trained up to 2020 gives trustworthy 2021--25 intervals", "no",
          r"V13: target drifts (PSI 0.44); V12: raw 80\,\% interval covers 50\,\%; conformal restores 83\,\%")],
    it1=[(r"Raw panel", r"4{,}089 complete-case country-years\\4 raw predictors\\country-grouped 5-fold CV"),
         (r"Diagnostics of the baseline", [
             ("V7_baseline_diagnostics.png", 1, r"\textbf{V7} predicted vs actual, residuals by region and level, held-out importance")], 1),
         (r"Baseline models", r"Ridge, Random Forest\\histogram gradient boosting\\fixed country$\rightarrow$fold map\\held-out permutation importance"),
         (r"Inferences $K_1$", r"\textbf{K1.1} broadband alone carries the model; mobile adds 0.02\\\textbf{K1.2} errors largest where histories differ\\\textbf{K1.3} complete cases drop rows that are missing not at random (W1)")],
    T=(r"Feedback loop: transformation $T$ (new predictors and new rows, informed by $K_1$ of W1 and W3)",
       r"\textbf{T1} artefact cells removed (shared core) \quad \textbf{T2} 13 past-only engineered features: landline peak and "
       r"decline, leapfrog index, years since mobile overtook landlines, SIMs per owner, capped mobile, $\log(1+\text{broadband})$, "
       r"floor flag, 1- and 3-year momentum \quad \textbf{T3} multivariate imputation (MICE) fitted inside each training fold, "
       r"with missingness flags: 2{,}134 more usable rows \quad \textbf{T4} inner loop: drift chart above 0.2 $\Rightarrow$ conformal re-calibration"),
    it2=[(r"Transformed panel", r"6{,}223 country-years\\13 engineered features\\same folds as iteration 1"),
         (r"Explaining and stress-testing the new model", [
             ("V10_iteration_comparison.png", 2, r"\textbf{V10} same rows: $-14$\,\% error"),
             ("V11b_residual_choropleth.png", 3, r"\textbf{V11} SHAP + residual map"),
             ("V12b_pdp_calibration.png", 4, r"\textbf{V12b} learned floor; calibration"),
             ("V13_psi_drift_chart.png", 5, r"\textbf{V13} target drifts after 2020"),
             ("V12_forecast_fan_chart.png", 6, r"\textbf{V12} conformal fan chart")], 3),
         (r"Models on $D_2$", r"HistGBM + engineered features\\ablations (MICE vs features)\\SHAP, PDP + ICE\\quantile GBM + CQR; PSI"),
         (r"Knowledge $K_2$", r"\textbf{H3.2} history lowers error by 14\,\% on identical rows\\\textbf{H3.3} post-2020 concept drift; conformal intervals cover 83\,\% at nominal 80\,\%\\$\rightarrow$ feeds the synthesis (V14)")],
    path=r"\textcircled{\scriptsize 1} diagnose the raw baseline (V7) $\rightarrow$ $K_1$ $\rightarrow$ $T$ $\rightarrow$ "
         r"\textcircled{\scriptsize 2} compare iterations on identical rows and folds (V10) $\rightarrow$ \textcircled{\scriptsize 3} explain "
         r"the model and locate its errors (V11) $\rightarrow$ \textcircled{\scriptsize 4} read the learned shapes (V12b a--c) $\rightarrow$ "
         r"\textcircled{\scriptsize 5} monitor drift (V13) $\rightarrow$ \textcircled{\scriptsize 6} forecast with re-calibrated intervals "
         r"(V12, V12b d) $\rightarrow$ synthesis V14.",
)


# ======================================================================================== master V0
def master() -> str:
    L = [PREAMBLE, r"\begin{document}", r"\begin{tikzpicture}[x=1cm,y=1cm]"]
    H = 14.6
    # main question + compact Keim diamond
    L.append(rf"\draw[box=ink, fill=pan] (0,{H - 2.55}) rectangle (13.2,{H});")
    L.append(rf"\node[badge=ink, anchor=north west] at (0.08,{H - 0.08}) {{Main question}};")
    L.append(rf"\node[txt=12.9cm, font=\footnotesize] at (0.08,{H - 0.5}) {{Can infrastructure and history explain and "
             r"forecast which countries get online, and does feeding each iteration's inferences back into the data "
             r"change the answer? \textcolor{mut}{\scriptsize (A1 asked whether mobile let countries skip fixed lines, "
             r"and whether that closed the divide.)}};")
    L.append(rf"\node[txt=12.9cm] at (0.08,{H - 1.45}) {{Answered by three workflows, one per team member and one per "
             r"family of methods. Each is a complete unrolled Keim loop with its own question, hypotheses, feedback loop "
             r"and second iteration (Figs.~W1--W3); their knowledge is combined in one synthesis.};")
    cx, cy = 15.65, H - 1.2
    L.append(rf"\node[font=\bfseries\scriptsize, text=ink, anchor=north] at ({cx},{H}) {{Keim et al., compact}};")
    for nm, x, y, c in [("V", cx, cy + 0.45, "cV"), ("D", cx - 1.05, cy - 0.1, "cD"), ("K", cx + 1.05, cy - 0.1, "cK"), ("M", cx, cy - 0.65, "cM")]:
        L.append(rf"\node[box={c}, font=\bfseries\scriptsize, minimum size=0.42cm, inner sep=1pt] (k{nm}) at ({x:.2f},{y:.2f}) {{{nm}}};")
    for a, b in [("kD", "kV"), ("kD", "kM"), ("kV", "kK"), ("kM", "kK"), ("kV", "kM")]:
        L.append(rf"\draw[arr=ink, line width=0.5pt] ({a}) -- ({b});")
    L.append(rf"\draw[arr=cL, line width=0.9pt] (kK.south) .. controls ({cx + 0.9},{cy - 1.25}) and ({cx - 0.9},{cy - 1.25}) .. (kD.south);")
    L.append(rf"\node[font=\scriptsize, text=cL, anchor=north] at ({cx},{cy - 1.0}) {{feedback loop, unrolled below}};")

    # shared data column
    top, bot = H - 2.9, 0.55
    L.append(rf"\draw[box=cD] (0,{bot}) rectangle (1.75,{top});")
    L.append(rf"\node[badge=cD, anchor=north west] at (0.08,{top - 0.08}) {{D}};")
    L.append(rf"\node[txt=1.6cm] at (0.06,{top - 0.6}) {{\textbf{{Raw OWID panel}} (A1)\\[2pt]217 countries\\1960--2025\\5 indicators\\[4pt]\textbf{{Shared core}}\\one definition of the panel, regions, artefact rules and features}};")
    # synthesis column
    L.append(rf"\draw[box=cK, line width=1.1pt] (16.35,{bot}) rectangle ({W},{top});")
    L.append(rf"\node[badge=cK, anchor=north west] at (16.43,{top - 0.08}) {{K$^\ast$}};")
    L.append(rf"\node[txt=1.6cm] at (16.4,{top - 0.6}) {{\textbf{{Synthesis}}\\[2pt]\img{{V14_synthesis_answer.png}}{{1.6}}{{1.4}}\\[2pt]V14: the divide is a delay on one S-curve; fixed infrastructure, not mobile, marks the position}};")

    lanes = [
        ("W1", "Statistics", "Parikshith", r"When, and is it closing?", "V1_diffusion_scurves.png", "V3b, V1, V2, V3",
         r"delay; SIMs $\neq$ people; Fiji artefact; floor", r"shared T1 (artefacts); own time $t-t_0$; headroom; per owner",
         "V1c_takeoff_alignment.png", "V9, V1c, V1b", r"one curve, 6--12 yr delay; low starters close least"),
        ("W2", "Unsupervised", "Sindhoor", r"Which kinds of countries?", "V5b_bootstrap_coassignment.png", "V4, V5, V5b, V5c, V6, V6b, V6c",
         r"unstable borders; no history; \textbf{Fiji transient} $\Rightarrow$ shared T1", r"artefacts out; snapshot$\rightarrow$curves; DTW",
         "V8_trajectory_archetypes.png", "V8a, V8b, V8, V8c, V8d", r"three path archetypes; 37 of 133 regroup"),
        ("W3", "Supervised", "Varshith", r"Predict and forecast?", "V7_baseline_diagnostics.png", "V7",
         r"broadband only; mid-level errors; MNAR", r"13 history features \textbf{from W1's $K_1$}; in-fold MICE; drift$\Rightarrow$CQR",
         "V12_forecast_fan_chart.png", "V10, V11, V12b, V13, V12", r"$-14$\,\% error; drift; 83\,\% coverage"),
    ]
    lane_h = (top - bot) / 3
    xs = [(2.0, 3.85), (4.1, 6.55), (6.8, 8.75), (9.05, 11.05), (11.35, 13.8), (14.05, 16.1)]
    for i, (wid, fam, who, q, th1, ids1, k1, t, th2, ids2, k2) in enumerate(lanes):
        y1 = top - i * lane_h - 0.08
        y0 = y1 - lane_h + 0.22
        ym = (y0 + y1) / 2
        L.append(rf"\draw[box=ink, fill=white] ({xs[0][0]},{y0:.2f}) rectangle ({xs[0][1]},{y1:.2f});")
        L.append(rf"\node[badge=ink, anchor=north west] at ({xs[0][0] + 0.06},{y1 - 0.06:.2f}) {{{wid}}};")
        L.append(rf"\node[txt=1.75cm] at ({xs[0][0] + 0.04},{y1 - 0.5:.2f}) {{\textbf{{{fam}}}\\{who}\\[3pt]\textit{{{q}}}}};")
        for (a, b), col, lab, th, ids, kk in [(xs[1], "cV", "Iteration 1", th1, ids1, None), (xs[4], "cV", "Iteration 2", th2, ids2, None)]:
            L.append(rf"\draw[box={col}] ({a},{y0:.2f}) rectangle ({b},{y1:.2f});")
            L.append(rf"\node[ttl] at ({a + 0.02},{y1 - 0.02:.2f}) {{{lab}: V + M}};")
            L.append(rf"\node[inner sep=0.5pt, draw=mut!45, line width=0.3pt] at ({(a + b) / 2:.2f},{ym + 0.02:.2f}) {{\img{{{th}}}{{{b - a - 0.2:.2f}}}{{{lane_h - 1.45:.2f}}}}};")
            L.append(rf"\node[capt, text width={b - a - 0.1:.2f}cm, anchor=south] at ({(a + b) / 2:.2f},{y0 + 0.02:.2f}) {{{ids}}};")
        for (a, b), lab, txt in [(xs[2], "K$_1$", k1), (xs[5], "K$_2$", k2)]:
            L.append(rf"\draw[box=cK] ({a},{y0:.2f}) rectangle ({b},{y1:.2f});")
            L.append(rf"\node[ttl] at ({a + 0.02},{y1 - 0.02:.2f}) {{{lab}}};")
            L.append(rf"\node[txt={b - a - 0.1:.2f}cm] at ({a + 0.02},{y1 - 0.42:.2f}) {{{txt}}};")
        a, b = xs[3]
        L.append(rf"\draw[box=cL, fill=cL!6, line width=1pt] ({a},{y0:.2f}) rectangle ({b},{y1:.2f});")
        L.append(rf"\node[ttl, text=cL] at ({a + 0.02},{y1 - 0.02:.2f}) {{T: Feedback loop}};")
        L.append(rf"\node[txt={b - a - 0.1:.2f}cm] at ({a + 0.02},{y1 - 0.42:.2f}) {{{t}}};")
        for (p, q2), col in zip(zip(xs[:-1], xs[1:]), ["ink", "cV", "cL", "cL", "cV"]):
            L.append(rf"\draw[arr={col}{', line width=1.4pt' if col == 'cL' else ''}] ({p[1]},{ym:.2f}) -- ({q2[0]},{ym:.2f});")
        L.append(rf"\draw[arr=cD] (1.75,{ym:.2f}) -- ({xs[0][0]},{ym:.2f});")
        L.append(rf"\draw[arr=cK] ({xs[5][1]},{ym:.2f}) -- (16.35,{ym:.2f});")
        if i == 0:
            w1k = (xs[2], y0, ym)
        if i == 1:
            w2k = (xs[2], y0, y1)
        if i == 2:
            w3t = (xs[3], y1)
    L.append(rf"\node[anchor=north west, font=\scriptsize, text=mut, inner sep=0] at (0,{bot - 0.12}) {{Solid black: workflow order. Red: the "
             r"feedback loop of each workflow (one edge per workflow, no self-loops). Bold notes mark knowledge passed between workflows.};")
    L += [r"\end{tikzpicture}", r"\end{document}"]
    return "\n".join(L)


if __name__ == "__main__":
    for name, text in [("V0_workflow_architecture", master()), ("W1_workflow_statistics", workflow(W1)),
                       ("W2_workflow_unsupervised", workflow(W2)), ("W3_workflow_supervised", workflow(W3))]:
        (DIAG / f"{name}.tex").write_text(text, encoding="utf8")
        print("wrote", name)

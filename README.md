# DAS732 A3: Visual Analytics Workflow

**From Snapshots to Trajectories: a visual analytics workflow on global internet adoption**
Team Plot Twist: Parikshith Aithal K V (BT2024249), Sindhoor Ganapathi Hegde (BT2024014), Varshith M Gowda (BT2024227)

**Live demo (runs in the browser, first load 20-60 s):** https://aithal007.github.io/DAS732-A3-Visual-Analytics/

Core dataset: the OWID "Internet" panel from A1 (217 countries, 1960 to 2025). The A1 report is the appendix of the A3 report.

## What is in this folder

| Path | Contents |
|---|---|
| `report/DAS732_A3_Report.pdf` | IEEE-format report, 17 pages, with the A1 report appended (39 pages in total; A1 is not assessed) |
| `report/DAS732_A3_Report.tex` | LaTeX source. Figures are read from `images/` |
| `images/` | The 20 report figures (V0-V14 plus V0b, V1b, V3b, V8b, V12b) as 21 PNG files (V11 has two panels, V11a and V11b). V0 and V0b are also vector PDFs in `report/diagrams/`, which the report uses. Every figure appears in the full deck |
| `report/diagrams/` | TikZ sources of the two architecture diagrams (V0 workflow, V0b system), vector PDF in the report |
| `slides/DAS732_A3_DataStory_6min.pptx` | **Video deck, 6-minute cut**: 16 story slides + 8 hidden appendix slides for questions |
| `slides/DAS732_A3_DataStory.pptx` | Full data-story deck (24 slides, about 15 min); every report figure; notes hold the narration |
| `video/VIDEO_SCRIPT_6min.md` | Narration for the 6-minute video (5:34): who speaks, what to say, when to demo |
| `video/VIDEO_SCRIPT.md` | Narration for the full 24-slide deck |
| `video/` | Put the recorded demo video here |
| `app.py` | Interactive Streamlit demo, one tab per workflow stage |
| `src/` | All code (see below) |
| `outputs/` | Every table and number quoted in the report (CSV/JSON), plus the run logs |
| `data/raw/` | `master_dataset.csv` (A1 merged panel) and `region_lookup.csv` (A1 continents) |
| `data/processed/` | Generated parquet files (iteration-2 panel, out-of-fold predictions, forecasts) |

## Run it

```bash
pip install -r requirements.txt
python src/run_all.py          # about 4 minutes; regenerates images/ and outputs/
python src/build_diagrams.py   # TikZ architecture diagrams (needs pdflatex)
streamlit run app.py           # interactive demo
python src/build_slides.py     # rebuilds the full deck
python src/build_slides_6min.py # 6-minute video cut + its script
python src/build_pages.py      # rebuilds docs/ (GitHub Pages, stlite in-browser app)
cd report && latexmk -pdf DAS732_A3_Report.tex
```

## Code map (who owns what)

| File | Owner | Workflow stage | Figures |
|---|---|---|---|
| `src/common.py` | Parikshith (shared) | D1 loading; the **feedback-loop transformation T** (artefact rules and engineered features) | - |
| `src/p1_parikshith_stats.py` | Parikshith | V1 statistics; transformation views; synthesis | V1, V2, V3, V9, V14 |
| `report/diagrams/*.tex`, `src/build_diagrams.py` | Parikshith | workflow and system architecture (TikZ) | V0, V0b |
| `src/p1_parikshith_convergence_quality.py` | Parikshith | convergence (sigma, beta, Gini); data quality and missing-not-at-random test | V1b, V3b |
| `src/p2_sindhoor_unsupervised.py` | Sindhoor | M1 PCA, clustering, anomalies; M2 DTW trajectory clustering | tables; quick-look plots in `outputs/diagnostics/` |
| `src/p2_sindhoor_figures.py` | Sindhoor | report figures for the unsupervised part | V4, V5, V6, V8, V8b |
| `src/p3_leakage_guards.py` | Varshith | leakage assertions | - |
| `src/p3_varshith_supervised.py` | Varshith | M1 baseline; M2 engineered model, ablations, forecast with CQR | - |
| `src/p3_varshith_plots.py` | Varshith | V2/V4 model views | V7, V10, V11a/b, V12, V12b |
| `src/p3_varshith_drift.py` | Varshith | PSI drift chart for the forecast | V13 |
| `app.py` | team | live demo, one tab per workflow stage | - |

## Viva preparation: likely questions

- **Where is the feedback loop?** The red edge K1 to D2 in V0, and Table II of the report. Each inference changes the data: T1 cleaning, T2 features, T3 trajectories plus imputation.
- **Why "unrolled"?** A cyclic diagram cannot show which pass produced which inference. Unrolling gives two iterations with no self-loops and one labelled loop edge.
- **Why group the CV folds by country?** Rows of the same country are near-duplicates across years. Random K-fold would leak a country's own history into its test fold. Grouping tests generalisation to an unseen country.
- **Is any feature leaking the target?** No. A guard rejects duplicates, counts and any column named "internet". Features are past-only (the landline peak is an expanding maximum). The trajectory clusters use the target and are deliberately not used as a feature.
- **Why k = 3?** k = 2 has the highest silhouette, but it only separates low from high. Within k = 3 to 6, with no cluster smaller than 4, k = 3 has the best silhouette. It is only moderately stable (bootstrap ARI 0.64), and the report says so.
- **Why DTW?** Countries follow the same S-curve shifted in time (V1). Euclidean distance would treat a late starter as dissimilar. DTW aligns shifts within a 5-year band.
- **What is PSI?** PSI = Σ (cur − base)·ln(cur/base) over the baseline's deciles. Thresholds: below 0.1 keep the model; 0.1 to 0.2 evaluate; above 0.2 retrain or re-calibrate.
- **What is CQR?** Fit the quantile models on years up to 2017 and score 2018-20, using the score max(lo − y, y − hi). Take the ⌈(n+1)(1−α)⌉/n quantile of the scores and widen the 2021-25 band by it.
- **Why does phone ownership top SHAP?** Ownership is observed in only 11% of rows. Elsewhere it is a MICE estimate built from the other indicators, so it acts as a composite index. This is a stated limitation.
- **Causation?** No. There is no income variable, by design (the question is about infrastructure), so all results are associations.

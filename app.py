"""
Interactive demonstrator for the DAS732 A3 visual analytics workflow.
    python src/run_all.py      (once, to create outputs/ and data/processed/)
    streamlit run app.py
Also runs in the browser (GitHub Pages) through stlite: see src/build_pages.py.

Tabs follow the unrolled workflow: Workflow -> Data -> Iteration 1 (explore, mine, model) ->
Feedback loop -> Iteration 2 (archetypes, models) -> Forecast & drift -> Answer
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ROOT = Path(__file__).resolve().parent
PROC, OUT, IMG = ROOT / "data" / "processed", ROOT / "outputs", ROOT / "images"
TARGET = "internet_users_share_pct"

# ---------------------------------------------------------------- design tokens
NAVY, INK, MUTED, LINE, SURF = "#14213D", "#1F2328", "#6E7781", "#D0D7DE", "#F6F8FA"
BLUE, ORANGE, RED, GREEN, PINK = "#2A78D6", "#E69F00", "#C0392B", "#009E73", "#CC79A7"
REGION_COLORS = {"Africa": "#E69F00", "Asia": "#56B4E9", "Europe": "#009E73", "North America": "#CC79A7",
                 "South America": "#D55E00", "Oceania": "#0072B2", "Other": "#999999"}
ARCH = {0: ("T0 late starters", "#E69F00"), 1: ("T1 S-curve majority", "#56B4E9"),
        2: ("T2 early saturators", "#009E73")}
CLUS = {"0": "#E69F00", "1": "#56B4E9", "2": "#009E73"}
STAGE = {"D": BLUE, "V": ORANGE, "M": GREEN, "K": PINK, "LOOP": RED}
TRACK = ["D1", "V1", "M1", "V2", "K1", "LOOP", "D2", "V3", "M2", "V4", "K2"]

pio.templates["a3"] = go.layout.Template(layout=go.Layout(
    font=dict(family="Inter, Segoe UI, sans-serif", size=13, color=INK),
    colorway=["#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#D55E00", "#0072B2"],
    paper_bgcolor="white", plot_bgcolor="white",
    xaxis=dict(gridcolor="#EEF1F4", linecolor=LINE, zeroline=False, ticks="outside", tickcolor=LINE),
    yaxis=dict(gridcolor="#EEF1F4", linecolor=LINE, zeroline=False),
    hoverlabel=dict(bgcolor="white", bordercolor=LINE, font=dict(color=INK, size=12)),
    legend=dict(orientation="h", y=-0.18, x=0, bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=10, r=10, t=10, b=10)))
pio.templates.default = "plotly_white+a3"

st.set_page_config(page_title="From Snapshots to Trajectories", page_icon="🌐", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"], .stMarkdown, .stText, button, input {{ font-family: 'Inter', 'Segoe UI', sans-serif; }}
.block-container {{ padding-top: 1.0rem; padding-bottom: 3rem; max-width: 1400px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
/* sidebar */
section[data-testid="stSidebar"] {{ background: {NAVY}; }}
section[data-testid="stSidebar"] * {{ color: #E6EDF3 !important; }}
section[data-testid="stSidebar"] .sb-q {{ background: rgba(255,255,255,.07); border-left: 3px solid {RED};
   padding: 10px 12px; border-radius: 6px; font-size: .9rem; line-height: 1.45; }}
section[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb] div,
section[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb] span,
section[data-testid="stSidebar"] [data-testid="stSelectbox"] input,
section[data-testid="stSidebar"] [data-baseweb="select"] div,
section[data-testid="stSidebar"] [data-baseweb="select"] input {{ color: {INK} !important;
   -webkit-text-fill-color: {INK} !important; opacity: 1 !important; }}
section[data-testid="stSidebar"] [data-baseweb="select"] > div {{ background: white !important; border-radius: 8px; }}
/* hero */
.hero {{ background: linear-gradient(120deg, {NAVY} 0%, #1F3A68 60%, #2A78D6 140%); border-radius: 16px;
   padding: 26px 30px 22px 30px; color: white; margin-bottom: 18px; position: relative; overflow: hidden; }}
.hero:after {{ content: ""; position: absolute; right: -60px; top: -60px; width: 260px; height: 260px;
   border-radius: 50%; background: radial-gradient(circle, rgba(255,255,255,.12), rgba(255,255,255,0) 70%); }}
.hero .kicker {{ font-size: .78rem; letter-spacing: .12em; text-transform: uppercase; opacity: .75; font-weight: 600; }}
.hero h1 {{ font-size: 2.35rem; font-weight: 800; margin: 4px 0 6px 0; color: white; letter-spacing: -.02em; }}
.hero p {{ font-size: 1.05rem; opacity: .9; margin: 0; max-width: 900px; }}
/* kpi */
.kpis {{ display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 14px; margin: 0 0 18px 0; }}
.kpi {{ background: white; border: 1px solid {LINE}; border-radius: 14px; padding: 14px 16px;
   box-shadow: 0 1px 2px rgba(20,33,61,.06); }}
.kpi .lab {{ font-size: .78rem; color: {MUTED}; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; }}
.kpi .val {{ font-size: 2.0rem; font-weight: 800; color: {INK}; line-height: 1.15; margin: 4px 0; }}
.kpi .pill {{ display: inline-block; font-size: .78rem; font-weight: 600; padding: 2px 9px; border-radius: 999px; }}
.good {{ background: #E6F4EA; color: #1A7F37; }} .bad {{ background: #FDECEA; color: {RED}; }}
.neu {{ background: #EEF4FC; color: {BLUE}; }}
/* tracker */
.track {{ display: flex; gap: 6px; flex-wrap: wrap; margin: 2px 0 14px 0; }}
.track span {{ font-size: .74rem; font-weight: 700; padding: 4px 10px; border-radius: 999px; background: #EDF0F3;
   color: {MUTED}; }}
/* tabs as pills */
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: none; flex-wrap: wrap; }}
.stTabs [data-baseweb="tab"] {{ background: {SURF}; border: 1px solid {LINE}; border-radius: 999px;
   padding: 6px 14px; height: auto; font-weight: 600; }}
.stTabs [aria-selected="true"] {{ background: {NAVY}; color: white !important; border-color: {NAVY}; }}
.stTabs [aria-selected="true"] p {{ color: white !important; }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none; }}
/* cards */
.card-h {{ font-weight: 700; font-size: 1.02rem; color: {INK}; margin: 2px 0 0 0; }}
.card-s {{ font-size: .86rem; color: {MUTED}; margin: 0 0 6px 0; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 14px !important; background: white;
   box-shadow: 0 1px 2px rgba(20,33,61,.05); }}
/* observation / cause / inference */
.oci {{ display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 12px; margin-top: 8px; }}
.oci div {{ background: white; border: 1px solid {LINE}; border-left-width: 5px; border-radius: 10px;
   padding: 10px 14px; font-size: .92rem; line-height: 1.45; }}
.oci b {{ display: block; font-size: .72rem; letter-spacing: .08em; text-transform: uppercase; margin-bottom: 3px; }}
.o {{ border-left-color: {BLUE} !important; }} .o b {{ color: {BLUE}; }}
.c {{ border-left-color: {ORANGE} !important; }} .c b {{ color: #B07800; }}
.i {{ border-left-color: {RED} !important; }} .i b {{ color: {RED}; }}
.foot {{ color: {MUTED}; font-size: .8rem; margin-top: 30px; border-top: 1px solid {LINE}; padding-top: 10px; }}
@media (max-width: 900px) {{ .kpis, .oci {{ grid-template-columns: 1fr; }} .hero h1 {{ font-size: 1.6rem; }} }}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- helpers
def tracker(active):
    html = ""
    for t in TRACK:
        k = "LOOP" if t == "LOOP" else t[0]
        on = t in active
        style = f"background:{STAGE[k]};color:white" if on else ""
        html += f'<span style="{style}">{"feedback loop" if t == "LOOP" else t}</span>'
    st.markdown(f'<div class="track">{html}</div>', unsafe_allow_html=True)


def oci(o, c, i):
    st.markdown(f'<div class="oci"><div class="o"><b>Observation</b>{o}</div><div class="c"><b>Cause</b>{c}</div>'
                f'<div class="i"><b>Inference</b>{i}</div></div>', unsafe_allow_html=True)


def card(title, sub=None):
    box = st.container(border=True)
    box.markdown(f'<p class="card-h">{title}</p>' + (f'<p class="card-s">{sub}</p>' if sub else ""),
                 unsafe_allow_html=True)
    return box


def show(fig, h=420, box=None):
    fig.update_layout(height=h)
    (box or st).plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def _tab(name):
    """parquet locally; CSV in the in-browser (GitHub Pages / stlite) build"""
    p = PROC / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.read_csv(PROC / f"{name}.csv")


@st.cache_data
def load():
    rd = lambda f: pd.read_csv(OUT / f)  # noqa: E731
    return {
        "panel": _tab("panel_iteration2"), "raw": pd.read_csv(ROOT / "data" / "raw" / "master_dataset.csv"),
        "i1": _tab("p3_iter1_oof"), "i2": _tab("p3_iter2_oof"), "fc": _tab("p3_forecast_holdout"),
        "cv": pd.read_csv(OUT / "p3_cv_results.csv", index_col=0),
        "reg": pd.read_csv(OUT / "p3_regional_rmse_same_rows.csv", index_col=0),
        "psi": rd("p3_psi_drift.csv"), "arts": rd("p1_artifact_log.csv"), "it1": rd("iter1_assignments.csv"),
        "it2": rd("iter2_assignments.csv"), "fits": rd("p1_logistic_fits.csv"), "sp": rd("p1_spearman_by_year.csv"),
        "sig": rd("p1_sigma_convergence.csv"), "beta": rd("p1_beta_convergence.csv"),
        "cov": pd.read_csv(OUT / "p1_coverage_by_year.csv", index_col=0), "tsne": rd("iter2_tsne_embedding.csv"),
        "calib": rd("p3_interval_calibration.csv"), "m": json.load(open(OUT / "p3_model_metrics.json")),
    }


D = load()
fm, cv = D["m"]["forecast"], D["cv"]
I1 = cv.loc["I1 HistGBM (raw, complete case)", "rmse_mean"]
I2 = cv.loc["I2 HistGBM (engineered + MICE), same rows as I1", "rmse_mean"]
regions_of = D["panel"].drop_duplicates("code").set_index("code").region

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 🌐 Team Plot Twist")
    st.caption("DAS732 Data Visualization · IIIT Bangalore · Term 1, 2026-27 · Assignment 3")
    st.markdown('<div class="sb-q"><b>A3 question</b><br>Can infrastructure and history explain and forecast which '
                'countries get online, and does feeding iteration-1 inferences back into the data change the '
                'answer?</div>', unsafe_allow_html=True)
    st.write("")
    it = st.radio("Model views show", ["Iteration 1 (raw)", "Iteration 2 (transformed)"], index=1)
    countries_all = sorted(D["fc"].entity.unique())
    focus = st.selectbox("Focus country", countries_all, index=countries_all.index("India"))
    st.write("")
    st.caption("Data: OWID Internet panel (A1), 217 countries, 1960-2025. Every model score is out-of-fold "
               "(countries held out) or out-of-time (2021-25 held out).")
ITER2 = it.startswith("Iteration 2")

# ---------------------------------------------------------------- hero + KPIs
st.markdown("""<div class="hero"><div class="kicker">Visual analytics workflow · one feedback loop</div>
<h1>From Snapshots to Trajectories</h1>
<p>Did mobile leapfrogging close the digital divide? Two Keim iterations on the OWID internet panel:
statistics, clustering and prediction on raw data, a feedback loop that changes the data, and a second
iteration that learns what the first could not.</p></div>""", unsafe_allow_html=True)

st.markdown(f"""<div class="kpis">
<div class="kpi"><div class="lab">Africa's take-off delay</div><div class="val">11.9 yrs</div>
  <span class="pill neu">behind Europe · p = 1e-15</span></div>
<div class="kpi"><div class="lab">Model error, same rows</div><div class="val">{I2:.2f} pp</div>
  <span class="pill good">▼ {I1 - I2:.2f} vs iteration 1 ({(I2 - I1) / I1:+.0%})</span></div>
<div class="kpi"><div class="lab">Archetype separation</div><div class="val">0.52</div>
  <span class="pill good">▲ silhouette vs 0.35 snapshot</span></div>
<div class="kpi"><div class="lab">80% interval coverage</div><div class="val">{fm['coverage_80_cqr']:.0%}</div>
  <span class="pill bad">raw was {fm['coverage_80_quantile']:.0%}</span> <span class="pill good">conformal fixes it</span></div>
</div>""", unsafe_allow_html=True)

tabs = st.tabs(["🧭 Workflow", "🗂️ Data", "📈 Explore", "🧩 Mine", "🎯 Baseline model", "🔁 Feedback loop",
                "🧬 Archetypes", "⚙️ Iteration 2 models", "🔮 Forecast & drift", "✅ Answer"])

# ================================================================ 0 workflow
with tabs[0]:
    tracker(TRACK)
    b = card("The workflow, unrolled", "Two complete Keim iterations joined by one labelled feedback edge")
    b.image(str(IMG / "V0_workflow_architecture.png"), width="stretch")
    b = card("System architecture (V0b)", "one shared core, three analysis modules, delivery reads artefacts only")
    b.image(str(IMG / "V0b_system_architecture.png"), width="stretch")
    oci("Keim's diamond drawn as a cycle has self-loops and cannot show which pass produced which inference.",
        "Iteration 2 is only meaningful if it is visibly driven by what iteration 1 found.",
        "Unroll it: D1 → V1 → M1 → V2 → K1, one red feedback edge K1 → D2, then D2 → V3 → M2 → V4 → K2.")

# ================================================================ 1 data
with tabs[1]:
    tracker(["D1"])
    c1, c2 = st.columns([1.5, 1])
    with c1:
        b = card("Coverage by year (V3b)", "% of all 217 countries observed")
        cov = D["cov"]
        fig = go.Figure(go.Heatmap(z=cov.values, x=cov.columns.astype(int), y=cov.index, colorscale="Blues",
                                   zmin=0, zmax=100, hovertemplate="%{y}<br>%{x}: %{z:.0f}%<extra></extra>",
                                   colorbar=dict(title="%", thickness=10)))
        show(fig, 330, b)
    with c2:
        b = card("Artefacts removed by the feedback loop", "10 cells, two transparent rules")
        b.dataframe(D["arts"][["entity", "year", "variable", "value"]].round(1), hide_index=True, height=300)
    oci("Broadband starts around 2000, ownership exists only in survey years, and 2025 is nearly empty. Rows without "
        "an ownership survey have a median 50% online against 75% with one.",
        "Surveys and broadband statistics are collected less often in less-connected countries.",
        "Data are missing not at random: complete cases in iteration 1, in-fold MICE plus missingness flags in "
        "iteration 2.")

# ================================================================ 2 explore
with tabs[2]:
    tracker(["D1", "V1"])
    c1, c2 = st.columns([1.15, 1])
    with c1:
        b = card("Diffusion S-curves (V1)", "observed series with fitted logistic curves")
        sel = b.multiselect("Countries", sorted(D["panel"].entity.unique()),
                            default=list(dict.fromkeys(["India", "Kenya", "Germany", "Brazil", focus])))
        sub = D["panel"][D["panel"].entity.isin(sel) & (D["panel"].year >= 1990)]
        fig = px.line(sub, x="year", y=TARGET, color="entity", markers=True, labels={TARGET: "internet users (%)"})
        fits = D["fits"].set_index("entity")
        tt = np.linspace(1990, 2030, 200)
        for c in sel:
            if c in fits.index:
                L, k, t0 = fits.loc[c, ["L_saturation", "k_growth", "t0_takeoff"]]
                fig.add_scatter(x=tt, y=L / (1 + np.exp(-k * (tt - t0))), mode="lines", showlegend=False,
                                line=dict(dash="dot", width=1.4, color=MUTED),
                                hovertemplate=f"{c} logistic · take-off {t0:.0f} · k {k:.2f}<extra></extra>")
        fig.update_yaxes(range=[0, 100])
        show(fig, 390, b)
    with c2:
        b = card("Is the divide closing? (V1b)", "absolute spread and relative inequality, each indexed to its peak")
        sg = D["sig"]
        fig = go.Figure([
            go.Scatter(x=sg.year, y=sg.sd / sg.sd.max() * 100, name="s.d. (absolute spread)",
                       line=dict(color=INK, width=3)),
            go.Scatter(x=sg.year, y=sg.gini / sg.gini.max() * 100, name="Gini (relative inequality)",
                       line=dict(color=ORANGE, width=3, dash="dash")),
            go.Scatter(x=sg.year, y=sg.gap_eu_af, name="Europe − Africa gap (pp)", line=dict(color=RED, width=2))])
        fig.update_yaxes(title="index / pp", range=[0, 110])
        show(fig, 445, b)
    c1, c2 = st.columns(2)
    with c1:
        b = card("What correlates with internet use, by year (V2)", "Spearman ρ with 95% country-bootstrap bands")
        sp = D["sp"].dropna()
        fig = go.Figure()
        for ind, col, name in [("broadband_subs_per100", BLUE, "Fixed broadband"),
                               ("mobile_subs_per100", ORANGE, "Mobile subscriptions"),
                               ("telephone_lines_per100", MUTED, "Landlines")]:
            g = sp[sp.indicator == ind]
            fig.add_scatter(x=list(g.year) + list(g.year[::-1]), y=list(g.hi) + list(g.lo[::-1]), fill="toself",
                            fillcolor=col, opacity=0.15, line=dict(width=0), showlegend=False, hoverinfo="skip")
            fig.add_scatter(x=g.year, y=g.rho, name=name, line=dict(color=col, width=3))
        fig.update_yaxes(range=[0, 1.02], title="Spearman ρ")
        show(fig, 360, b)
    with c2:
        b = card("Headroom closed 2014→2024 (V1b)", "each dot a country; hover for names")
        bt = D["beta"].copy()
        bt["entity"] = bt.code.map(D["panel"].drop_duplicates("code").set_index("code").entity)
        bt["closed_pct"] = bt.closed * 100
        fig = px.scatter(bt, x="y14", y="closed_pct", color="region", hover_name="entity",
                         color_discrete_map=REGION_COLORS,
                         labels={"y14": "internet users 2014 (%)", "closed_pct": "share of remaining gap closed (%)"})
        fig.update_traces(marker=dict(size=9, line=dict(width=1, color="white")))
        show(fig, 360, b)
    oci("Same S-curve everywhere, Africa 11.9 years behind Europe. Gini fell 0.81 → 0.19, but the absolute gap peaked "
        "in 2014 (64 pp) and countries that started low closed less of their headroom (ρ = +0.52).",
        "Regions entered one diffusion process at different times; mobile saturated everywhere, so it stopped "
        "varying (ρ 0.88 → 0.42) while broadband still does (0.80).",
        "The divide is a delay that is narrowing only in relative terms; late starters are not catching up.")

# ================================================================ 3 mine
with tabs[3]:
    tracker(["M1", "V2"])
    it1 = D["it1"].copy()
    c1, c2 = st.columns([1.35, 1])
    with c1:
        b = card("2024 snapshot clusters (V5) and anomalies (V6)", "bubble size = anomaly score")
        method = b.radio("Clustering method", ["iter1_kmeans", "iter1_gmm", "iter1_ward"], horizontal=True,
                         format_func=lambda m: m.replace("iter1_", "").upper())
        it1["cluster"] = it1[method].astype(int).astype(str)
        fig = px.scatter(it1, x="broadband_subs_per100", y=TARGET, color="cluster", color_discrete_map=CLUS,
                         size=(it1.anomaly_combined.clip(0.05) ** 3) * 40 + 4, hover_name="name",
                         hover_data={"reason": True, "anomaly_rank": True, "mobile_subs_per100": ":.0f"},
                         labels={"broadband_subs_per100": "fixed broadband /100", TARGET: "internet users (%)"})
        show(fig, 430, b)
    with c2:
        b = card("Top anomalies", "Isolation Forest + LOF, rank-averaged")
        b.dataframe(it1.sort_values("anomaly_rank").head(12)[["anomaly_rank", "name", "reason"]]
                    .rename(columns={"anomaly_rank": "rank"}), hide_index=True, height=440)
    oci("k = 3 archetypes ordered by connectivity (broadband medians 1 / 20 / 39 per 100), bootstrap ARI 0.64. Fiji "
        "is the top anomaly for both detectors; Ward's third cluster is only Fiji and Hong Kong.",
        "Countries differ mostly in fixed infrastructure; one recording error distorts a single-year snapshot.",
        "Snapshot archetypes reproduce A1 but are fragile, and Fiji 2023-24 is an error to remove (K1.1).")

# ================================================================ 4 baseline
with tabs[4]:
    tracker(["M1", "V2", "K1"])
    c1, c2 = st.columns([1, 1])
    i1 = D["i1"]
    with c1:
        b = card("Out-of-fold predictions, raw infrastructure (V7)", f"HistGBM RMSE {I1:.2f} pp, countries held out")
        fig = px.scatter(i1.sample(1500, random_state=1), x=TARGET, y="pred_gbm_i1", color="region",
                         color_discrete_map=REGION_COLORS, opacity=0.6, hover_name="entity",
                         labels={TARGET: "actual (%)", "pred_gbm_i1": "predicted (%)"})
        fig.add_shape(type="line", x0=0, y0=0, x1=100, y1=100, line=dict(color=MUTED, dash="dash"))
        show(fig, 400, b)
    with c2:
        b = card("Residuals by region", "actual − predicted, percentage points")
        fig = px.box(i1, x="residual_i1", y="region", color="region", color_discrete_map=REGION_COLORS,
                     points=False, labels={"residual_i1": "residual (pp)", "region": ""})
        fig.update_layout(showlegend=False)
        show(fig, 400, b)
    oci("Raw infrastructure explains 86% of variance for unseen countries; error is 12-14 pp above 10% adoption. "
        "Shuffling mobile costs 0.02 R², broadband about 1.0.",
        "Raw inputs carry no history and inflated mobile counts.",
        "The model needs cleaned mobile, a broadband floor and history features (K1.2-K1.4).")

# ================================================================ 5 loop
with tabs[5]:
    tracker(["K1", "LOOP", "D2"])
    arts = D["arts"]
    c1, c2 = st.columns([1.2, 1])
    with c1:
        b = card("Inspect a removed cell", "raw (iteration 1) against cleaned (iteration 2)")
        c = b.selectbox("Country", sorted(arts.entity.unique()), index=sorted(arts.entity.unique()).index("Fiji"))
        v = arts[arts.entity == c].variable.iloc[0]
        raw = D["raw"][(D["raw"].entity == c) & (D["raw"].year >= 2005)]
        cl = D["panel"][(D["panel"].entity == c) & (D["panel"].year >= 2005)]
        fig = go.Figure([go.Scatter(x=raw.year, y=raw[v], name="raw", line=dict(dash="dot", color=RED, width=2)),
                         go.Scatter(x=cl.year, y=cl[v], name="cleaned", mode="lines+markers",
                                    line=dict(color=BLUE, width=3))])
        fig.update_yaxes(title=v)
        show(fig, 360, b)
    with c2:
        b = card("Inference → transformation", "the feedback loop in one table")
        b.markdown("""
| K1 inference | Transformation T |
|---|---|
| Fiji 2023-24 is an artefact | T1: two rules, 10 cells removed |
| SIMs ≠ people | T2: SIMs per owner, mobile capped at 100 |
| History matters | T2: landline peak, decline, years since mobile overtook landlines · T3: 2000-25 trajectories |
| Broadband floor, MNAR gaps | T2: log broadband, zero flag · T3: in-fold MICE + flags |
""")
    oci("Ten cells are removed; history features correlate 0.74-0.75 with internet use at near-full coverage; "
        "SIMs per owner ≈ 0.", "Each K1 inference names what was wrong or missing in the raw data.",
        "Every engineered feature is past-only and never uses the target, so iteration 2 is leakage-free.")

# ================================================================ 6 archetypes
with tabs[6]:
    tracker(["D2", "M2", "V4"])
    c1, c2 = st.columns([1, 1])
    with c1:
        b = card("Trajectory map (V8b)", "t-SNE of DTW distances · size = 2025 internet share")
        ts = D["tsne"].copy()
        ts["archetype"] = ts.traj_cluster.map(lambda k: ARCH[int(k)][0])
        fig = px.scatter(ts, x="tsne_x", y="tsne_y", color="archetype", size="internet_2025", size_max=16,
                         hover_name="entity", color_discrete_map={v[0]: v[1] for v in ARCH.values()},
                         hover_data={"tsne_x": False, "tsne_y": False, "internet_2025": ":.0f"})
        fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
        show(fig, 430, b)
    with c2:
        b = card("Curves of one archetype", "pick an archetype; the focus country is highlighted")
        arch = b.radio("Archetype", [0, 1, 2], format_func=lambda k: ARCH[k][0], horizontal=True)
        codes = D["it2"].loc[D["it2"].traj_cluster == arch, "code"]
        sub = D["panel"][D["panel"].code.isin(codes) & D["panel"].year.between(2000, 2025)]
        fig = go.Figure()
        for code_, g in sub.groupby("code"):
            fig.add_scatter(x=g.year, y=g[TARGET], mode="lines", line=dict(color=ARCH[arch][1], width=1),
                            opacity=0.35, name=g.entity.iloc[0], showlegend=False,
                            hovertemplate=f"{g.entity.iloc[0]}: %{{y:.0f}}%<extra></extra>")
        f = D["panel"][(D["panel"].entity == focus) & D["panel"].year.between(2000, 2025)]
        fig.add_scatter(x=f.year, y=f[TARGET], mode="lines", line=dict(color=INK, width=4), name=focus)
        fig.update_yaxes(range=[0, 100], title="internet users (%)")
        show(fig, 400, b)
    b = card("Snapshot vs trajectory (V8)", "static figure from the report")
    b.image(str(IMG / "V8_trajectory_archetypes.png"), width="stretch")
    oci("Three archetypes (43 / 113 / 25 countries), silhouette 0.52 against 0.35 for snapshot labels; 37 of 133 "
        "countries change group; two thirds of Africa are late starters.",
        "Countries at the same 2024 level got there along different paths.",
        "Archetypes are paths, not positions, and they form one continuum ordered by timing.")

# ================================================================ 7 models
with tabs[7]:
    tracker(["M2", "V4"])
    c1, c2 = st.columns([1.35, 1])
    with c1:
        b = card(f"Out-of-fold residual map · {it}", "blue = more online than infrastructure predicts")
        yr = b.slider("Year", 2005, 2024, 2020)
        if ITER2:
            r = D["i2"][D["i2"].year == yr][["code", "entity", "residual_i2"]].rename(columns={"residual_i2": "res"})
        else:
            r = D["i1"][D["i1"].year == yr][["code", "entity", "residual_i1"]].rename(columns={"residual_i1": "res"})
        fig = px.choropleth(r, locations="code", color=r.res.clip(-20, 20), hover_name="entity",
                            color_continuous_scale="RdBu", range_color=(-20, 20),
                            labels={"color": "actual − predicted (pp)"})
        fig.update_geos(showframe=False, projection_type="natural earth", showcoastlines=False,
                        landcolor="#EEF1F4", showcountries=True, countrycolor="white")
        fig.update_layout(coloraxis_colorbar=dict(thickness=10, len=0.6))
        show(fig, 400, b)
        b.markdown(f"**RMSE in {yr}: {np.sqrt(np.mean(r.res ** 2)):.2f} pp**")
    with c2:
        b = card("Regional RMSE on identical rows (V10)", "grey = iteration 1, blue = iteration 2")
        reg = D["reg"].drop(index="Other", errors="ignore").reset_index().melt(
            id_vars="region", value_vars=["rmse_i1", "rmse_i2"], var_name="iteration", value_name="RMSE")
        reg["iteration"] = reg.iteration.map({"rmse_i1": "Iteration 1", "rmse_i2": "Iteration 2"})
        fig = px.bar(reg, y="region", x="RMSE", color="iteration", barmode="group", orientation="h",
                     color_discrete_map={"Iteration 1": "#9AA4AE", "Iteration 2": BLUE}, labels={"region": ""})
        show(fig, 455, b)
    oci(f"On identical rows RMSE falls {I1:.2f} → {I2:.2f} pp (−14%), most in Asia (−21%) and Africa (−18%); "
        "partial dependence shows a flat broadband floor below 3 per 100 and a 12 pp landline-history lift.",
        "History features help where paths differ most; ablations show the features, not imputation, carry the gain.",
        "Feeding K1 back improves prediction for unseen countries, measured fairly.")

# ================================================================ 8 forecast
with tabs[8]:
    tracker(["D2", "V3", "M2", "V4"])
    c1, c2 = st.columns([1.2, 1])
    with c1:
        b = card(f"{focus}: cone of uncertainty (V12)", "trained on ≤ 2020, red dots were hidden from the model")
        band = b.radio("Interval", ["raw quantile", "conformal (CQR)"], horizontal=True, index=1)
        h = D["panel"][(D["panel"].entity == focus) & (D["panel"].year.between(2000, 2020))]
        f = D["fc"][D["fc"].entity == focus].sort_values("year")
        lo, hi = ("q10", "q90") if band == "raw quantile" else ("q10_cqr", "q90_cqr")
        fig = go.Figure([
            go.Scatter(x=h.year, y=h[TARGET], name="observed ≤ 2020", line=dict(color=INK, width=3)),
            go.Scatter(x=list(f.year) + list(f.year[::-1]), y=list(f[hi]) + list(f[lo][::-1]), fill="toself",
                       fillcolor="rgba(42,120,214,0.22)", line=dict(width=0), name=f"80% {band}", mode="lines"),
            go.Scatter(x=f.year, y=f.q50, name="median forecast", mode="lines",
                       line=dict(dash="dash", color=BLUE, width=3)),
            go.Scatter(x=f.year, y=f[TARGET], name="observed 2021-25", mode="markers",
                       marker=dict(color=RED, size=11, line=dict(width=1.5, color="white")))])
        fig.add_vline(x=2020.5, line_color=MUTED, line_dash="dot")
        fig.update_yaxes(range=[0, 100], title="internet users (%)")
        show(fig, 400, b)
    with c2:
        b = card("Calibration curve (V12b)", "empirical coverage 2021-25 against nominal")
        cb = D["calib"]
        fig = go.Figure([
            go.Scatter(x=[0.45, 0.95], y=[0.45, 0.95], mode="lines", name="perfect",
                       line=dict(color=MUTED, dash="dash")),
            go.Scatter(x=cb.nominal, y=cb.raw, name="raw quantile", mode="lines+markers",
                       line=dict(color=RED, width=3), marker=dict(size=9)),
            go.Scatter(x=cb.nominal, y=cb.cqr, name="conformal (CQR)", mode="lines+markers",
                       line=dict(color=BLUE, width=3), marker=dict(size=9, symbol="square"))])
        fig.update_xaxes(title="nominal", tickformat=".0%"); fig.update_yaxes(title="empirical", tickformat=".0%")
        show(fig, 455, b)
    b = card("Drift chart (V13)", "Population Stability Index against 2015-20 · red band = retrain / re-calibrate")
    psi = D["psi"]
    fig = px.line(psi, x="year", y="psi", color="feature", markers=True)
    fig.add_hrect(y0=0.2, y1=1.5, fillcolor=RED, opacity=0.06, line_width=0)
    fig.add_hline(y=0.1, line_dash="dash", line_color=ORANGE)
    fig.add_hline(y=0.2, line_dash="dash", line_color=RED)
    fig.add_vline(x=2020.5, line_color=MUTED, line_dash="dot")
    fig.update_yaxes(range=[0, 1.5], title="PSI")
    show(fig, 360, b)
    oci(f"Target PSI passes 0.2 after 2020 while inputs stay below it until 2023. {fm['share_underpredicted']:.0%} "
        f"of forecasts are too low; raw 80% covers {fm['coverage_80_quantile']:.0%}, conformal "
        f"{fm['coverage_80_cqr']:.0%}, and CQR tracks the diagonal from 50% to 90%.",
        "Post-2020 acceleration: the same infrastructure now coincides with more people online (concept drift).",
        "Re-calibrate uncertainty whenever the drift chart crosses 0.2.")

# ================================================================ 9 answer
with tabs[9]:
    tracker(["K2"])
    b = card("The answer (V14)", "when · what · who · what the loop changed")
    b.image(str(IMG / "V14_synthesis_answer.png"), width="stretch")
    oci("Take-off 6-12 years behind Europe; in 2023 broadband ρ 0.79 and landline peak 0.81 against mobile 0.52; "
        "archetype broadband medians 0 / 17 / 40 per 100.",
        "Mobile saturated everywhere; fixed infrastructure still records when and how far a country moved.",
        "Mobile leapfrogging happened but did not close the divide. The divide is a delay on one S-curve, narrowing "
        "only in relative terms, and fixed infrastructure marks where a country stands. Association, not cause.")

st.markdown('<div class="foot">Team Plot Twist · Parikshith Aithal K V · Sindhoor Ganapathi Hegde · Varshith M Gowda · '
            'DAS732 Data Visualization, IIIT Bangalore · Source: Our World in Data, Internet collection</div>',
            unsafe_allow_html=True)

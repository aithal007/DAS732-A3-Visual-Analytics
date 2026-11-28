"""Data-story deck for the video demo / live viva: slides/DAS732_A3_DataStory.pptx

Design rules (from A1/A2 feedback): every report figure appears; every slide title is the finding;
every figure slide carries an Observation -> Cause -> Inference strip; a workflow tracker shows where
in the unrolled diamond the story is. Speaker notes = narration script (presenter, time, words, demo cues).
"""
import json
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
IMG, OUT = ROOT / "images", ROOT / "outputs"
fm = json.load(open(OUT / "p3_model_metrics.json"))["forecast"]


def rgb(h):
    return RGBColor.from_string(h.lstrip("#"))


NAVY, INK, MUTED, BG, WHITE = rgb("#14213D"), rgb("#1F2328"), rgb("#6E7781"), rgb("#F6F8FA"), rgb("#FFFFFF")
STAGE = {"D": rgb("#2A78D6"), "V": rgb("#E69F00"), "M": rgb("#009E73"), "K": rgb("#CC79A7"),
         "LOOP": rgb("#C0392B"), "Q": rgb("#14213D")}
CARD = {"Observation": rgb("#2A78D6"), "Cause": rgb("#E69F00"), "Inference": rgb("#C0392B")}
TRACK = ["D1", "V1", "M1", "V2", "K1", "LOOP", "D2", "V3", "M2", "V4", "K2"]

# V11 on one slide: SHAP beeswarm + residual map side by side
_a, _b = Image.open(IMG / "V11a_shap_beeswarm.png"), Image.open(IMG / "V11b_residual_choropleth.png")
_h = 1400
_a = _a.resize((int(_a.width * _h / _a.height), _h)); _b = _b.resize((int(_b.width * _h / _b.height), _h))
_c = Image.new("RGB", (_a.width + _b.width + 40, _h), "white"); _c.paste(_a, (0, 0)); _c.paste(_b, (_a.width + 40, 0))
(OUT / "slide_assets").mkdir(exist_ok=True); _c.save(OUT / "slide_assets" / "V11_combined.png")

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
W, H = prs.slide_width, prs.slide_height
BLANK = prs.slide_layouts[6]
FONT = "Segoe UI"


def box(s, x, y, w, h, fill=None, line=None, shape=MSO_SHAPE.RECTANGLE):
    sh = s.shapes.add_shape(shape, x, y, w, h)
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid(); sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line; sh.line.width = Pt(1)
    sh.shadow.inherit = False
    return sh


def txt(s, x, y, w, h, runs, size=16, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    """runs: str or list of paragraphs; each paragraph is str or list of (text, bold, color)."""
    tb = s.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    paras = runs if isinstance(runs, list) else [runs]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(3)
        parts = para if isinstance(para, list) else [(para, bold, color)]
        for t, b, c in parts:
            r = p.add_run()
            r.text = t
            r.font.size, r.font.bold, r.font.name = Pt(size), b, FONT
            r.font.color.rgb = c
    return tb


def picture(s, path, x, y, maxw, maxh):
    im = Image.open(path)
    r = min(maxw / im.width, maxh / im.height)
    w, h = int(im.width * r), int(im.height * r)
    s.shapes.add_picture(str(path), x + (maxw - w) // 2, y + (maxh - h) // 2, w, h)


def tracker(s, here):
    x0, y = Inches(0.4), H - Inches(0.42)
    wseg = Inches(0.62)
    for i, t in enumerate(TRACK):
        k = "LOOP" if t == "LOOP" else t[0]
        on = t in here
        ww = Inches(1.05) if t == "LOOP" else wseg
        b = box(s, x0, y, ww - Inches(0.05), Inches(0.26), STAGE[k] if on else rgb("#E7EBEF"),
                shape=MSO_SHAPE.ROUNDED_RECTANGLE)
        tf = b.text_frame
        tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = "feedback loop" if t == "LOOP" else t
        r.font.size, r.font.bold, r.font.name = Pt(9), True, FONT
        r.font.color.rgb = WHITE if on else MUTED
        x0 += ww


def header(s, title, chip, chip_kind, presenter):
    box(s, 0, 0, W, Inches(1.0), NAVY)
    c = box(s, Inches(0.4), Inches(0.28), Inches(1.45), Inches(0.42), STAGE[chip_kind],
            shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    tf = c.text_frame; tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = chip; r.font.size, r.font.bold, r.font.name = Pt(12), True, FONT
    r.font.color.rgb = WHITE
    txt(s, Inches(2.0), Inches(0.12), Inches(10.9), Inches(0.8), title, 22, WHITE, True, anchor=MSO_ANCHOR.MIDDLE)
    txt(s, W - Inches(3.4), H - Inches(0.47), Inches(3.0), Inches(0.35), presenter, 11, MUTED,
        align=PP_ALIGN.RIGHT)


def oci_cards(s, x, y, w, h, oci, horizontal):
    keys = ["Observation", "Cause", "Inference"]
    n = 3
    gap = Inches(0.12)
    for i, k in enumerate(keys):
        if horizontal:
            cw = (w - gap * (n - 1)) // n
            cx, cy, ch = x + i * (cw + gap), y, h
        else:
            ch = (h - gap * (n - 1)) // n
            cx, cy, cw = x, y + i * (ch + gap), w
        box(s, cx, cy, cw, ch, WHITE, rgb("#D0D7DE"))
        box(s, cx, cy, Inches(0.09), ch, CARD[k])
        txt(s, cx + Inches(0.15), cy + Inches(0.05), cw - Inches(0.2), Inches(0.3), k.upper(), 10.5, CARD[k], True)
        txt(s, cx + Inches(0.15), cy + Inches(0.32), cw - Inches(0.22), ch - Inches(0.36), oci[k], 12.5, INK)


def fig_slide(title, img, chip, kind, here, presenter, oci, notes):
    s = prs.slides.add_slide(BLANK)
    box(s, 0, 0, W, H, BG)
    header(s, title, chip, kind, presenter)
    im = Image.open(IMG / img)
    aspect = im.width / im.height
    top, bottom = Inches(1.12), H - Inches(0.55)
    if aspect > 1.9:   # wide figure: figure on top, cards underneath
        ch = Inches(1.55)
        picture(s, IMG / img, Inches(0.3), top, W - Inches(0.6), bottom - top - ch - Inches(0.12))
        oci_cards(s, Inches(0.4), bottom - ch, W - Inches(0.8), ch, oci, horizontal=True)
    else:              # compact figure: figure left, cards right
        cw = Inches(3.9)
        picture(s, IMG / img, Inches(0.3), top, W - cw - Inches(0.8), bottom - top)
        oci_cards(s, W - cw - Inches(0.4), top + Inches(0.05), cw, bottom - top - Inches(0.1), oci, horizontal=False)
    tracker(s, here)
    s.notes_slide.notes_text_frame.text = notes
    return s


def text_slide(title, chip, kind, here, presenter, paras, notes, size=19, img=None, split=None):
    s = prs.slides.add_slide(BLANK)
    box(s, 0, 0, W, H, BG)
    header(s, title, chip, kind, presenter)
    if img:   # two text columns on top, supporting figure full width underneath
        half = (len(paras) + 1) // 2 if split is None else split
        cw = (W - Inches(1.2)) // 2
        txt(s, Inches(0.5), Inches(1.15), cw, Inches(2.4), paras[:half], size, INK)
        txt(s, Inches(0.5) + cw + Inches(0.2), Inches(1.15), cw, Inches(2.4), paras[half:], size, INK)
        picture(s, IMG / img, Inches(0.4), Inches(3.55), W - Inches(0.8), H - Inches(3.55) - Inches(0.55))
    else:
        txt(s, Inches(0.7), Inches(1.35), W - Inches(1.4), H - Inches(2.2), paras, size, INK)
    tracker(s, here)
    s.notes_slide.notes_text_frame.text = notes
    return s


P, S_, V_ = "Presenter: Parikshith", "Presenter: Sindhoor", "Presenter: Varshith"
B = lambda t: (t, True, INK)          # noqa: E731
N = lambda t: (t, False, INK)         # noqa: E731
M = lambda t: (t, False, MUTED)       # noqa: E731

# ------------------------------------------------------------------ 1 title
s = prs.slides.add_slide(BLANK)
box(s, 0, 0, W, H, NAVY)
box(s, Inches(0.8), Inches(1.5), Inches(0.12), Inches(2.3), STAGE["LOOP"])
txt(s, Inches(1.1), Inches(1.35), Inches(11), Inches(1.2), "From Snapshots to Trajectories", 46, WHITE, True)
txt(s, Inches(1.1), Inches(2.45), Inches(11), Inches(1.4),
    "A visual analytics workflow on global internet adoption:\ndid mobile leapfrogging close the digital divide?",
    24, rgb("#C9D1D9"))
txt(s, Inches(1.1), Inches(4.6), Inches(11), Inches(1.2),
    ["Team Plot Twist  ·  Parikshith Aithal K V  ·  Sindhoor Ganapathi Hegde  ·  Varshith M Gowda",
     "DAS732 Data Visualization  ·  Programming Assignment 3  ·  IIIT Bangalore  ·  Term 1, 2026-27"],
    15, rgb("#C9D1D9"))
s.notes_slide.notes_text_frame.text = (
    "[Parikshith, 0:00-0:20] Good morning. We are Team Plot Twist. In A1 we looked at the Our World in Data "
    "internet data and asked whether mobile leapfrogging closed the digital divide. In A3 we test what we saw, "
    "with statistics, clustering and prediction, inside a visual analytics workflow with one feedback loop.")

# ------------------------------------------------------------------ 2 question
text_slide("One question, four sub-questions, one feedback loop", "QUESTION", "Q", [], V_, [
    [B("A3 question:  "), N("can infrastructure and history explain and forecast which countries get online, "
                             "and does feeding iteration-1 inferences back into the data change the answer?")],
    [M(" ")],
    [B("RQ1 Statistics  "), N("when and how fast did countries adopt; is the divide closing; what does each "
                               "indicator measure?  "), M("V1, V1b, V2, V3, V3b")],
    [B("RQ2 Unsupervised  "), N("which kinds of countries exist, and which do not fit?  "), M("V4-V6, V8, V8b")],
    [B("RQ3 Supervised  "), N("can infrastructure predict an unseen country and forecast unseen years?  "),
     M("V7, V10-V13")],
    [B("RQ4 Workflow  "), N("which inferences change the data; what does iteration 2 learn?  "),
     M("V0, V0b, V9, V14")],
    [M(" ")],
    [M("Data: A1's OWID panel, 217 countries, 1960-2025, five indicators. Target for prediction: internet users %.")]],
    "[Varshith, 0:20-0:50] Our question extends A1. Four sub-questions, one per family of methods the brief "
    "asks for: statistics, unsupervised mining, supervised learning, and the workflow itself. Every figure in the "
    "report answers one of them, and every figure is in this video.")

# ------------------------------------------------------------------ 3 workflow
fig_slide("The workflow, unrolled: two full iterations joined by one labelled feedback loop",
          "V0_workflow_architecture.png", "WORKFLOW", "Q", TRACK, P, {
              "Observation": "Keim's diamond drawn as a cycle has self-loops: it cannot show which pass produced "
                             "which inference.",
              "Cause": "Iteration 2 only makes sense if it is visibly driven by iteration 1's findings.",
              "Inference": "Unroll it: D1-V1-M1-V2-K1, then ONE red feedback edge K1 to D2, then D2-V3-M2-V4-K2."},
          "[Parikshith, 0:50-1:35] On the left, Keim's diamond: data, visualization, models, knowledge. As a cycle "
          "it cannot tell us which pass produced which finding, so on the right we unroll it. Iteration 1 runs on the "
          "raw A1 data: exploration, models, diagnostics, and four inferences K1. The red edge is the single feedback "
          "loop: K1 becomes a data transformation. Iteration 2 repeats the full diamond on the transformed data. "
          "The tracker at the bottom of every slide shows where we are.")

fig_slide("System architecture: one shared core, three analysis modules, every number read from artefacts",
          "V0b_system_architecture.png", "SYSTEM", "Q", [], P, {
              "Observation": "common.py holds the panel, artefact rules and engineered features; three modules "
                             "write tables and figures; report, deck and app only read them.",
              "Cause": "The feedback loop must be one definition shared by every analysis, or iterations drift apart.",
              "Inference": "One command (run_all.py) reproduces every figure and number with fixed seeds."},
          "[Parikshith, 0:00-0:30] How it is built: one shared core module holds the data loading, the artefact "
          "rules and the engineered features, so the feedback loop is a single definition. Three analysis modules "
          "write tables and figures, and the report, the deck and the browser app only read those artefacts. One "
          "command reproduces everything.")

# ------------------------------------------------------------------ 4-6 Parikshith iteration 1
fig_slide("V1  Every region follows the same S-curve, up to 12 years late, and late starters do not catch up",
          "V1_diffusion_scurves.png", "ITER 1 · V", "V", ["D1", "V1"], P, {
              "Observation": "Logistic fits 201 of 203 countries (median R² 0.99). Africa takes off in 2017, "
                             "11.9 yr after Europe (Kruskal-Wallis p = 4e-17).",
              "Cause": "Adoption is one diffusion process (Rogers); regions entered it at different times.",
              "Inference": "The divide is a DELAY on a shared curve, and late starters climb no faster "
                           "(rho = -0.21), so it is not closing by itself."},
          "[Parikshith, 1:35-2:20] I fitted a three-parameter logistic S-curve to every country; 201 of 203 fit with "
          "R-squared above 0.9. The take-off year, half of saturation, differs by region with p of 4 times 10 to the "
          "minus 17. Africa takes off in 2017, almost 12 years after Europe, confirmed by Holm-corrected Mann-Whitney. "
          "And on the right: later take-off does not mean faster growth. So the divide is a delay that is not "
          "closing on its own.")

fig_slide("V1b  The divide narrowed in relative terms only; low starters close the smallest share of their gap",
          "V1b_convergence.png", "ITER 1 · V", "V", ["D1", "V1"], P, {
              "Observation": "Gini 0.81 → 0.19 (1995-2024), but the Europe-Africa gap peaked at 64 pp in 2014 and "
                             "is still 52. Headroom closed rises with the 2014 level (rho = +0.52).",
              "Cause": "Relative measures shrink as leaders saturate near 100%; laggards are still on the flat "
                       "start of their S-curve.",
              "Inference": "No catch-up (no beta-convergence in headroom): the absolute divide is shrinking slowly "
                           "and only since 2014."},
          "[Parikshith, 0:00-0:40] Is the divide closing? It depends on how it is measured. Relative inequality, "
          "the Gini, fell from 0.81 to 0.19. But the absolute gap between the median European and African country "
          "peaked at 64 points in 2014 and is still 52. And on the right: countries that started low closed a "
          "SMALLER share of their remaining gap. So there is no catch-up; the divide shrinks only because leaders "
          "hit the ceiling.")

fig_slide("V2  Broadband has a floor and a non-linear effect; mobile stopped separating countries after 2005",
          "V2_floor_effect_decoupling.png", "ITER 1 · V", "V", ["D1", "V1"], P, {
              "Observation": "29% of country-years have <1 broadband sub/100 yet up to 30% online; a break near "
                             "23/100. Mobile rho fell 0.88 → 0.42, broadband stays 0.80.",
              "Cause": "Mobile saturated everywhere, so it no longer varies between countries; broadband still does.",
              "Inference": "Use log broadband with a floor flag; A1's single pooled correlation hid a trend over "
                           "time."},
          "[Parikshith, 2:20-3:05] Left: almost a third of country-years have no broadband, yet internet use there "
          "still reaches 30 percent: that is the floor. Above it internet grows with log broadband, and segmented "
          "regression finds a break near 23 per 100. Right: correlation with internet use per year, with bootstrap "
          "intervals. In 2005 all three were near 0.9; by 2024 mobile is 0.42 and broadband 0.80. Mobile saturated, "
          "so it stopped telling countries apart.")

fig_slide("V3  Subscriptions count SIMs, not people, and Fiji's 2023-24 values are a recording artefact",
          "V3_multisim_inflation.png", "ITER 1 · V", "V", ["D1", "V1"], P, {
              "Observation": "Median 1.34 SIMs per owner. Ownership rho 0.89 vs subscriptions 0.52. Fiji: "
                             "111 → 361 → 574 per 100 while ownership stays 82%.",
              "Cause": "Prepaid multi-SIM markets inflate counts; a two-year 5x jump with flat ownership is not "
                       "adoption.",
              "Inference": "Normalise mobile by people (one team threshold: 1.5 SIMs/owner) and treat Fiji 2023-24 "
                           "as an error."},
          "[Parikshith, 3:05-3:45] The median country has 1.34 SIMs per phone owner, so subscriptions overstate "
          "people. Ownership correlates 0.89 with internet use, subscriptions only 0.52. We set one threshold for the "
          "whole team, 1.5 SIMs per owner. And Fiji jumps from 111 to 574 subscriptions in two years while ownership "
          "stays flat: that is a recording artefact. Sindhoor's anomaly detector finds it independently.")

fig_slide("V3b  Data are missing not at random, so the method changes between iterations",
          "V3b_data_quality_missingness.png", "DATA · V", "D", ["D1", "V1"], P, {
              "Observation": "Broadband starts ~2000, ownership only in survey years, 2025 nearly empty. Without an "
                             "ownership survey a country-year is 50% online vs 75% with one (p = 1e-41).",
              "Cause": "Statistics are collected less often where connectivity is low.",
              "Inference": "Iteration 1: complete cases only. Iteration 2: in-fold MICE plus missingness flags, never "
                           "a global mean."},
          "[Parikshith, 0:00-0:40] Before modelling, data quality. Broadband exists only from about 2000, phone "
          "ownership only in survey years, and 2025 is almost empty. Crucially, the gaps are not random: country-years "
          "without an ownership survey are far less online, 50 against 75 percent. Filling them with an average would "
          "bias poor countries upwards, so iteration 1 uses complete cases and iteration 2 uses imputation fitted "
          "inside each training fold, plus flags that tell the model a value was missing.")

# ------------------------------------------------------------------ 7-9 Sindhoor iteration 1
fig_slide("V4  Two axes describe 82% of 2024 connectivity; the mobile axis is stretched by two countries",
          "V4_pca_biplot.png", "ITER 1 · M", "M", ["D1", "M1", "V2"], S_, {
              "Observation": "PC1 (64%) = general connectivity; PC2 (18%) loads 0.96 on mobile alone. "
                             "Fiji and Hong Kong sit far above everyone.",
              "Cause": "Mobile subscriptions vary independently of the other four indicators, and two extreme "
                       "values dominate that variance.",
              "Inference": "One component is real connectivity; the other is partly an outlier effect to check."},
          "[Sindhoor, 3:45-4:20] On the 2024 snapshot of 135 complete countries, PCA puts 64 percent of variance on "
          "one connectivity axis, where all five indicators load positively. The second axis is mobile alone, and "
          "Fiji and Hong Kong stretch it by themselves.")

fig_slide("V5  Three archetypes ordered by connectivity, separated by landlines and broadband, only moderately stable",
          "V5_cluster_validation_profiles.png", "ITER 1 · M", "M", ["M1", "V2"], S_, {
              "Observation": "k = 3 best silhouette in k = 3-6 (0.33). Medians: broadband 1 / 20 / 39 per 100. "
                             "Bootstrap ARI 0.64.",
              "Cause": "Countries differ mostly in fixed infrastructure; a single year's snapshot is noisy.",
              "Inference": "The archetypes reproduce A1 Fig. 23, but Ward's 'best' result is just Fiji + Hong "
                           "Kong: snapshots are fragile."},
          "[Sindhoor, 4:20-5:00] I scanned k from 2 to 8 for k-means, GMM and Ward. k equal to 3 gives three "
          "archetypes ordered by connectivity, mobile-first, mixed and fixed-line legacy, matching A1's four kinds of "
          "country. They differ most on landlines and broadband, least on mobile. But bootstrap ARI is only 0.64, and "
          "Ward's high silhouette is fake: one of its clusters is just Fiji and Hong Kong.")

fig_slide("V6  Both detectors rank Fiji first; most anomalies are multi-SIM markets",
          "V6_anomalies.png", "ITER 1 · M", "M", ["M1", "V2", "K1"], S_, {
              "Observation": "Isolation Forest and LOF share 7 of their top 15 (Spearman 0.77). Fiji ranks first "
                             "in both; Monaco for landlines.",
              "Cause": "Multi-SIM inflation and one recording error create the most isolated points.",
              "Inference": "The model independently confirms V3: Fiji 2023-24 is an error to remove, not a "
                           "country to describe."},
          "[Sindhoor, 5:00-5:35] Isolation Forest and LOF, rank-averaged. Both put Fiji first. Most of the others "
          "are multi-SIM markets; Monaco is there for 95 landlines per 100. This is the model confirming what "
          "Parikshith's V3 showed, and it becomes inference K1.1. [DEMO, app tab 1: switch the clustering method; "
          "hover Fiji.]")

# ------------------------------------------------------------------ 10 Varshith iteration 1
fig_slide("V7  A raw-infrastructure model explains 86% of variance but leans on broadband alone",
          "V7_baseline_diagnostics.png", "ITER 1 · M", "M", ["M1", "V2"], V_, {
              "Observation": "Grouped 5-fold CV: HistGBM RMSE 11.8 pp, R² 0.856. Error 12-14 pp above 10% online. "
                             "Shuffling mobile costs 0.02 R².",
              "Cause": "Raw inputs carry no history and inflated mobile counts.",
              "Inference": "Prediction fails where histories differ most; the model needs history and cleaned "
                           "mobile."},
          "[Varshith, 5:35-6:15] We predict internet share from raw broadband, mobile, landlines and year. Folds are "
          "grouped by country, so the model is always tested on countries it has never seen. Random forest gets 11.5 "
          "points, gradient boosting 11.8. Held-out permutation importance: broadband dominates, mobile adds almost "
          "nothing. The errors are large in mid-adoption countries.")

# ------------------------------------------------------------------ 11-12 Parikshith feedback loop
text_slide("K1 → FEEDBACK LOOP: each inference becomes a change to the data", "FEEDBACK", "LOOP",
           ["K1", "LOOP", "D2"], S_, [
               [B("K1.1  Fiji 2023-24 is a recording artefact  "), M("(V3-V6)")],
               [N("   → T1: two artefact rules on every country; 10 cells removed")],
               [B("K1.2  Subscriptions overstate people  "), M("(V3, V7)")],
               [N("   → T2: SIMs per owner, flag at 1.5, mobile capped at 100")],
               [B("K1.3  History matters  "), M("(V1, V1b, V5, A1 Fig. 5)")],
               [N("   → T2: landline peak, decline, years since mobile overtook landlines")],
               [N("   → T3: reshape to 2000-2025 trajectories")],
               [B("K1.4  Broadband floor; data missing not at random  "), M("(V2, V3b)")],
               [N("   → T2: log broadband, zero flag;  T3: in-fold MICE + flags")],
               [M("Below, V9: the transformed data where iteration 2 starts. Every feature is past-only and "
                  "never uses the target.")]],
           "[Sindhoor, 6:15-7:00] This is the feedback loop. Four inferences leave iteration 1 and each one "
           "changes the data. Fiji is an error, so two artefact rules, a level shift and a one-year spike, remove 10 "
           "cells across all countries. Subscriptions are not people, so we normalise by owners and cap mobile at 100. "
           "History matters, so we add the landline peak as an expanding maximum, past-only, and reshape to "
           "trajectories. And broadband is non-linear, so log transform, a floor flag, and imputation inside each "
           "training fold.", size=14, img="V9_feedback_transformation.png", split=4)

fig_slide("V9  The transformation removes ten artefact cells and adds history no single-year value contains",
          "V9_feedback_transformation.png", "LOOP · V", "LOOP", ["LOOP", "D2", "V3"], P, {
              "Observation": "T1: Fiji and Saint Kitts spikes removed. T2: history features rho 0.74-0.75 with "
                             "near-full coverage; SIMs per owner rho ≈ 0.",
              "Cause": "Pooled correlations share a time trend, so the gain is new information, not bigger "
                       "numbers.",
              "Inference": "Iteration 2 starts from cleaner data with the history dimension iteration 1 lacked."},
          "[Parikshith, 7:00-7:35] This is the first view of iteration 2: what the transformation changed. On the "
          "left, the removed cells. In the middle, the engineered features against the raw ones: the history features "
          "carry information no single-year value has, and SIMs per owner is noise, so we neutralise it. On the "
          "right, coverage, which is why we impute. [DEMO, app tab 2: Fiji raw versus cleaned.]")

# ------------------------------------------------------------------ 13 Sindhoor iteration 2
fig_slide("V8  Grouped by PATH, countries form three S-curve archetypes that cut across the 2024 snapshot",
          "V8_trajectory_archetypes.png", "ITER 2 · M", "M", ["D2", "M2", "V4"], S_, {
              "Observation": "DTW on 181 curves: late starters (43), S-curve majority (113), early saturators (25). "
                             "Silhouette 0.52 vs 0.35; 37 of 133 change group.",
              "Cause": "Countries at the same 2024 level reached it along different paths.",
              "Inference": "Archetypes are paths, not positions: A1's 'same place, opposite histories' found by "
                           "a model."},
          "[Sindhoor, 7:35-8:15] In iteration 2 I clustered each country's whole curve, 2000 to 2025, with dynamic "
          "time warping. Three archetypes: late starters like Tanzania, the S-curve majority like the Maldives, early "
          "saturators like South Korea. Two thirds of Africa are late starters. These clusters separate far better, "
          "silhouette 0.52, and 37 of 133 countries change group compared with the 2024 snapshot.")

fig_slide("V8b  The three archetypes are the natural cut of the DTW tree and lie on one timing continuum",
          "V8b_dendrogram_tsne.png", "ITER 2 · M", "M", ["M2", "V4"], S_, {
              "Observation": "Late starters split from everyone else at the root of the dendrogram; the t-SNE map "
                             "runs from Ethiopia and Tanzania to South Korea and Germany.",
              "Cause": "DTW aligns curves that have the same shape but different timing.",
              "Inference": "k = 3 is the structure in the data, not an arbitrary choice; India, Indonesia and "
                           "Nigeria sit on the boundary."},
          "[Sindhoor, 0:00-0:35] Is k equal to 3 justified? The dendrogram of DTW distances says yes: late starters "
          "split off at the root and early saturators next. The t-SNE map of the same distances lays the countries on "
          "one continuum ordered by timing, from Ethiopia to South Korea, with India, Indonesia and Nigeria on the "
          "boundary.")

# ------------------------------------------------------------------ 14-15 Varshith iteration 2
fig_slide("V10  On identical rows and folds, the transformed data lower error by 14%, most in Asia and Africa",
          "V10_iteration_comparison.png", "ITER 2 · M", "M", ["M2", "V4"], V_, {
              "Observation": "RMSE 11.77 → 10.16 on the same 4,089 rows; 8.84 on all 6,223. Asia -21%, Africa "
                             "-18%, Americas +1%.",
              "Cause": "History features help where paths differ most; ablations show features, not imputation, "
                       "carry the gain.",
              "Inference": "Feeding K1 back improves prediction for unseen countries, fairly measured."},
          "[Varshith, 8:15-8:55] Same countries, same folds, same rows. Error falls from 11.77 to 10.16 points, minus "
          "14 percent, most in Asia and Africa, nothing in the Americas. With imputation, 2,134 more rows become "
          "usable and error reaches 8.84. The ablations show the engineered features carry most of the gain.")

fig_slide("V11  Ownership, broadband and the landline peak drive the model; its errors are geographic",
          "../outputs/slide_assets/V11_combined.png", "ITER 2 · V", "V", ["M2", "V4"], V_, {
              "Observation": "SHAP: phone ownership, log broadband, landline peak. Blue (more online than predicted): "
                             "Gulf, Cuba. Red: Mongolia, China, Egypt.",
              "Cause": "Ownership is mostly imputed (a composite index); residual geography reflects policy and "
                       "price outside the data.",
              "Inference": "Infrastructure explains most of adoption; the rest is regional and needs other data."},
          "[Varshith, 8:55-9:30] On the left, SHAP ranks phone ownership, log broadband and the landline "
          "peak on top; ownership is mostly imputed, so it acts as a composite index. The map shows where "
          "infrastructure under- or over-explains adoption. Kosovo is a data gap, not a finding. [DEMO, app tab 3: "
          "toggle iteration 1 and 2.]")

# ------------------------------------------------------------------ 16 Parikshith drift
fig_slide("V13  After 2020 the target drifts past the retraining threshold while the inputs do not",
          "V13_psi_drift_chart.png", "ITER 2 · V", "V", ["D2", "V3"], V_, {
              "Observation": "PSI vs 2015-20: target 0.44 in 2022-24 (> 0.2); every input < 0.2 until 2023.",
              "Cause": "Post-2020 acceleration: the same infrastructure now coincides with more people online.",
              "Inference": "The input → output relation shifted; a model trained to 2020 will forecast too low and "
                           "too confidently."},
          "[Varshith, 9:30-10:10] This is the drift chart from the lecture on ML workflows: Population Stability "
          "Index per year against a 2015 to 2020 baseline, with thresholds 0.1 and 0.2. After 2020 the target crosses "
          "0.2 while every input stays below until 2023. Same infrastructure, more people online. So we predict, "
          "before looking, that the forecast will be too low and too confident.")

# ------------------------------------------------------------------ 17 Varshith forecast
fig_slide("V12  Forecasts made in 2020 run low; conformal re-calibration restores the cone of uncertainty",
          "V12_forecast_fan_chart.png", "ITER 2 · M", "M", ["M2", "V4"], V_, {
              "Observation": f"Holdout RMSE {fm['holdout_rmse']:.1f} pp; {fm['share_underpredicted']:.0%} too low; "
                             f"80% band covers {fm['coverage_80_quantile']:.0%}.",
              "Cause": "Exactly the drift V13 flagged: the post-2020 shift is absent from the training years.",
              "Inference": f"CQR (+{fm['cqr_margin_pp']:.1f} pp) restores {fm['coverage_80_cqr']:.0%} coverage: "
                           "re-calibrate whenever drift crosses 0.2."},
          f"[Varshith, 10:10-10:45] Quantile gradient boosting trained up to 2020, tested on 2021 to 2025. As the "
          f"drift chart predicted, {fm['share_underpredicted']:.0%} of forecasts are too low and the 80 percent band "
          f"covers only {fm['coverage_80_quantile']:.0%}. Conformalised quantile regression widens it by "
          f"{fm['cqr_margin_pp']:.1f} points and coverage returns to {fm['coverage_80_cqr']:.0%}. [DEMO, app tab 4: "
          "India, raw versus conformal.]")

fig_slide("V12b  The model learned the broadband floor and a history effect; only conformal intervals are calibrated",
          "V12b_pdp_calibration.png", "ITER 2 · M", "M", ["M2", "V4"], V_, {
              "Observation": "Partial dependence: flat below ~3 broadband per 100, rising to 30 (14 pp); landline peak "
                             "+12 pp. Raw coverage 27-68% at nominal 50-90%; CQR 50-94%.",
              "Cause": "Gradient boosting recovers the V2 floor; post-2020 drift breaks raw quantiles at every "
                       "level.",
              "Inference": "The model encodes the A1 story explicitly, and its uncertainty is trustworthy only after "
                           "conformal re-calibration."},
          "[Varshith, 0:00-0:40] Two checks on the model. Partial dependence: predictions stay flat while broadband "
          "is on the floor and rise between 3 and 30 per 100; the landline history adds 12 points. And the "
          "calibration curve tests the intervals at five levels, not one: raw intervals are over-confident everywhere, "
          "the conformal ones sit on the diagonal.")

# ------------------------------------------------------------------ 18 Parikshith synthesis
fig_slide("V14  Answer: the divide is a delay on one S-curve, and fixed infrastructure, not mobile, marks the position",
          "V14_synthesis_answer.png", "KNOWLEDGE", "K", ["K2"], P, {
              "Observation": "Take-off 6-12 yr behind Europe; 2023 rho: broadband 0.79, landline peak 0.81, "
                             "mobile 0.52; broadband 0 / 17 / 40 by archetype.",
              "Cause": "Mobile saturated everywhere; fixed infrastructure still records when and how far a country "
                       "moved.",
              "Inference": "Mobile leapfrogging happened but did not close the divide. The feedback loop improved "
                           "every check."},
          "[Parikshith, 10:45-11:30] Putting both iterations together. When: every region is on the same S-curve, "
          "6 to 12 years behind Europe. What: today broadband and the landline peak separate countries, mobile does "
          "not. Who: the archetypes differ in fixed infrastructure. And the feedback loop improved every check: error "
          "on the same rows, cluster separation, and forecast coverage.")

# ------------------------------------------------------------------ 19 conclusion
text_slide("What we know now, and what we cannot claim", "KNOWLEDGE", "K", ["K2"], V_, [
    [B("Answer.  "), N("Mobile leapfrogging happened but did not close the divide: a 6-12 year delay on one S-curve, "
                       "with no faster catch-up; fixed infrastructure, not mobile, marks a country's position.")],
    [B("Iteration 2 learned  "), N("archetypes are paths (0.52 vs 0.35); history cuts error 14% on identical rows; "
                                   "post-2020 drift requires re-calibrated forecasts (50% → 83% coverage).")],
    [M(" ")],
    [B("Limits.  "), N("Association, not cause (no income variable, by design). Ownership mostly imputed. Snapshot "
                       "clusters moderately stable (ARI 0.64). Conformal coverage assumes no further drift.")],
    [M(" ")],
    [M("Reproduce: python src/run_all.py  ·  Demo: streamlit run app.py  ·  Report: IEEE format, A1 report appended")]],
    "[Varshith, 11:30-12:00] To close: mobile leapfrogging happened but did not close the divide. We cannot "
    "claim causation, ownership is mostly imputed, and the forecasts assume no further drift. Everything is "
    "reproducible with one command. Thank you; we are happy to take questions.", size=18)

# ------------------------------------------------------------------ re-time the narration
import re  # noqa: E402

t = 0
share = {}
for sl in prs.slides:
    tf = sl.notes_slide.notes_text_frame
    m = re.match(r"\[(\w+), (\d+):(\d+)-(\d+):(\d+)\]", tf.text)
    if not m:
        continue
    dur = (int(m[4]) * 60 + int(m[5])) - (int(m[2]) * 60 + int(m[3]))
    share[m[1]] = share.get(m[1], 0) + dur
    new = f"[{m[1]}, {t // 60}:{t % 60:02d}-{(t + dur) // 60}:{(t + dur) % 60:02d}]"
    tf.text = new + tf.text[m.end():]
    t += dur
print("total", f"{t // 60}:{t % 60:02d}", {k: f"{v // 60}:{v % 60:02d} ({v / t:.0%})" for k, v in share.items()})

out = ROOT / "slides" / "DAS732_A3_DataStory.pptx"
prs.save(out)
print("saved", out, len(prs.slides), "slides")

# video script = the speaker notes, in order
lines = ["# A3 video script (generated from the slide notes by src/build_slides.py)", "",
         f"Length {t // 60}:{t % 60:02d}. Record the deck in slideshow mode. At each [DEMO ...] cue switch to the "
         "app (https://aithal007.github.io/DAS732-A3-Visual-Analytics/ or `streamlit run app.py`) for 10-15 seconds, "
         "then return. Every report figure appears on a slide.", ""]
for i, sl in enumerate(prs.slides, 1):
    title = next((sh.text_frame.text.splitlines()[0] for sh in sl.shapes
                  if sh.has_text_frame and len(sh.text_frame.text) > 20), "")
    lines += [f"## Slide {i}: {title}", "", sl.notes_slide.notes_text_frame.text, ""]
(ROOT / "video" / "VIDEO_SCRIPT.md").write_text("\n".join(lines), encoding="utf8")

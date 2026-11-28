"""Data-story deck for the video demo / live viva: slides/DAS732_A3_DataStory.pptx (32 slides)

Structure: main question and V0 -> workflow W1 (Parikshith) -> W2 (Sindhoor) -> W3 (Varshith) -> synthesis.
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
TRACK = ["D1", "V1", "M1", "K1", "LOOP", "D2", "V2", "M2", "K2"]   # one workflow, unrolled

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


def split_workflow(name):
    """Portrait workflow diagram -> two halves side by side (header + iteration 1 | loop + iteration 2),
    cut just above the red feedback band, so the diagram is readable on a 16:9 slide."""
    import numpy as np
    im = Image.open(IMG / name).convert("RGB")
    a = np.asarray(im).astype(int)
    red = (abs(a[..., 0] - 192) < 30) & (abs(a[..., 1] - 57) < 30) & (abs(a[..., 2] - 43) < 30)
    rows = red.mean(axis=1)
    h = a.shape[0]
    cut = next(r for r in range(int(h * 0.42), int(h * 0.8)) if rows[r] > 0.5) - 12
    top, bot = im.crop((0, 0, im.width, cut)), im.crop((0, cut, im.width, h))
    gap = 60
    out = Image.new("RGB", (im.width * 2 + gap, max(top.height, bot.height)), "white")
    out.paste(top, (0, 0)); out.paste(bot, (im.width + gap, 0))
    p = OUT / "slide_assets" / name.replace(".png", "_split.png")
    out.save(p)
    return "../outputs/slide_assets/" + p.name


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
    "Three visual analytics workflows on global internet adoption:\ndid mobile leapfrogging close the digital divide?",
    24, rgb("#C9D1D9"))
txt(s, Inches(1.1), Inches(4.6), Inches(11), Inches(1.2),
    ["Team Plot Twist  ·  Parikshith Aithal K V  ·  Sindhoor Ganapathi Hegde  ·  Varshith M Gowda",
     "DAS732 Data Visualization  ·  Programming Assignment 3  ·  IIIT Bangalore  ·  Term 1, 2026-27"],
    15, rgb("#C9D1D9"))
s.notes_slide.notes_text_frame.text = (
    "[Parikshith, 0:00-0:20] We are Team Plot Twist. In A1 we asked whether mobile leapfrogging closed the "
    "digital divide. In A3 we test what we saw with three visual analytics workflows, one per team member, each "
    "with its own hypotheses and its own feedback loop.")

# ------------------------------------------------------------------ 2 question
text_slide("One main question, three workflows, one synthesis", "QUESTION", "Q", [], V_, [
    [B("Main question:  "), N("can infrastructure and history explain and forecast which countries get online, "
                              "and does feeding each iteration's inferences back into the data change the answer?")],
    [M(" ")],
    [B("W1 Statistics (Parikshith)  "), N("when did countries take off, is the gap closing, what do the indicators "
                                           "measure?  "), M("V1, V1d, V2, V3 → V9, V1c, V1e, V1f")],
    [B("W2 Unsupervised (Sindhoor)  "), N("which kinds of countries exist, how stable, which fit none?  "),
     M("V4-V6c → V8a, V8, V8d")],
    [B("W3 Supervised (Varshith)  "), N("can infrastructure predict unseen countries and years; can we trust it?  "),
     M("V7 → V10-V13")],
    [M(" ")],
    [N("Each workflow: question and hypotheses written on it, figures plugged into the stage that produced them, a "
       "numbered analysis path, and one labelled feedback loop that transforms the data.")],
    [M("Data: A1's OWID panel, 217 countries, 1960-2025, five indicators. Target for prediction: internet users %.")]],
    "[Varshith, 0:20-0:45] Our main question extends A1. We answer it with three workflows, one per family of methods "
    "the brief asks for, and one per team member: statistics, unsupervised mining and supervised learning. Each has "
    "its own hypotheses and its own feedback loop, and a synthesis combines them.", size=17)

# ------------------------------------------------------------------ 3-4 master workflow + system
fig_slide("V0  Three unrolled workflows answer one question; each has exactly one labelled feedback loop",
          "V0_workflow_architecture.png", "WORKFLOW", "Q", TRACK, P, {
              "Observation": "Keim's diamond as a cycle has self-loops: it cannot show which pass produced which "
                             "inference.",
              "Cause": "A second iteration is only meaningful if the first iteration's knowledge visibly changes "
                       "the data.",
              "Inference": "Unroll each workflow: iteration 1 → K1 → red feedback loop T → iteration 2 → K2; all "
                           "three feed the synthesis V14."},
          "[Parikshith, 0:45-1:20] This is the main workflow. On the top right, Keim's diamond; as a cycle it cannot "
          "show which pass produced which finding, so we unroll it. Each lane is one workflow: iteration 1 on raw data, "
          "its inferences, the red feedback loop that transforms the data, iteration 2 and its knowledge. Bold notes "
          "mark knowledge passed between workflows, and all three meet in the synthesis.")

# ------------------------------------------------------------------ W1 Parikshith
fig_slide("W1  Statistical workflow: hypotheses, plugged-in evidence, a numbered path and one feedback loop",
          split_workflow("W1_workflow_statistics.png"), "W1", "Q", TRACK, P, {
              "Observation": "H1.1 delay on one curve: supported. H1.2 closing: refined. H1.3 subscriptions = "
                             "people: rejected. H1.4 linear broadband: rejected. H1.5 leapfroggers caught up: rejected.",
              "Cause": "Raw calendar-time levels mix when a country started with how far it has come.",
              "Inference": "T re-expresses every curve in its own time and scale, and the gap in headroom: a new "
                           "coordinate system, not a filter."},
          "[Parikshith, 0:00-0:40] My workflow, W1, statistics. The question and four hypotheses are written at the "
          "top, each with its verdict. Iteration 1 uses four figures, numbered in the order I read them. The red band "
          "is the feedback loop: it re-expresses each country's curve in its own time and scale. Iteration 2 tests "
          "the hypotheses again on the transformed data.")

fig_slide("V1  Every region follows an S-curve, up to 12 years late, and late starters do not catch up",
          "V1_diffusion_scurves.png", "W1 · ITER 1", "V", ["D1", "V1", "M1"], P, {
              "Observation": "Logistic fits 201 of 203 countries (median R² 0.99). Africa takes off in 2017, "
                             "11.9 yr after Europe (Kruskal-Wallis p = 4e-17).",
              "Cause": "Adoption is one diffusion process (Rogers); regions entered it at different times.",
              "Inference": "H1.1 is raised: is the divide only a DELAY? Calendar-time plots cannot tell, so the "
                           "loop will re-index time."},
          "[Parikshith, 0:00-0:40] Step 1. I fitted a logistic S-curve to every country; 201 of 203 fit with R-squared "
          "above 0.9. The take-off year differs by region with p of 4 times 10 to the minus 17: Africa takes off in "
          "2017, almost 12 years after Europe, and later starters do not grow faster.")

fig_slide("V1d  Take-off is a spatial pattern: most of Africa took off 12 years after Europe",
          "V1d_takeoff_map.png", "W1 · ITER 1", "V", ["D1", "V1"], P, {
              "Observation": "Take-off year on the map: Europe median 2006, the Americas 2012, Asia 2013, Africa and "
                             "Oceania 2017. Contiguous areas, not outliers.",
              "Cause": "Diffusion spread outward from early-networked economies.",
              "Inference": "The delay is geographic; calendar-time comparisons mix WHERE with WHEN, so T3 "
                           "re-indexes time."},
          "[Parikshith, 0:00-0:30] Step 2, where. The same take-off year on the map. North America, Western Europe, "
          "Japan and Australia took off first; most of Africa and South Asia about twelve years later. The delay "
          "is geographic, not a few outliers.")

fig_slide("V2  Broadband has a floor and a non-linear effect; mobile stopped separating countries after 2005",
          "V2_floor_effect_decoupling.png", "W1 · ITER 1", "V", ["D1", "V1", "M1"], P, {
              "Observation": "29% of country-years have <1 broadband sub/100 yet up to 30% online; a break near "
                             "23/100. Mobile rho fell 0.88 → 0.42, broadband stays 0.80.",
              "Cause": "Mobile saturated everywhere, so it no longer varies between countries; broadband still does.",
              "Inference": "H1.4 rejected: use log broadband with a floor flag; A1's pooled correlation hid a trend."},
          "[Parikshith, 0:00-0:40] Step 3. Almost a third of country-years have no broadband, yet internet there still "
          "reaches 30 percent: a floor. Segmented regression finds a break near 23 per 100. On the right, mobile's "
          "correlation with internet use fell from 0.88 to 0.42 while broadband stays at 0.80.")

fig_slide("V3  Subscriptions count SIMs, not people, and Fiji's 2023-24 values are a recording artefact",
          "V3_multisim_inflation.png", "W1 · ITER 1", "V", ["D1", "V1", "M1", "K1"], P, {
              "Observation": "Median 1.34 SIMs per owner. Ownership rho 0.89 vs subscriptions 0.52. Fiji: "
                             "111 → 361 → 574 per 100 while ownership stays 82%.",
              "Cause": "Prepaid multi-SIM markets inflate counts; a 5x jump in two years with flat ownership is not "
                       "adoption.",
              "Inference": "H1.3 rejected: normalise mobile by people (1.5 SIMs/owner) and treat Fiji 2023-24 as an "
                           "error."},
          "[Parikshith, 0:00-0:35] Step 4. The median country has 1.34 SIMs per phone owner, so subscriptions "
          "overstate people. And Fiji jumps from 111 to 574 subscriptions in two years while ownership stays flat: a "
          "recording artefact. Sindhoor's workflow finds it independently.")

text_slide("W1 feedback loop: four inferences become a new coordinate system for the data", "W1 · LOOP", "LOOP",
           ["K1", "LOOP", "D2"], P, [
               [B("K1.1  Fiji 2023-24 is an artefact  "), M("(V3; W2's V6b)")],
               [N("   → T1: two rules on every country; 10 cells removed")],
               [B("K1.2  Subscriptions overstate people  "), M("(V3)")],
               [N("   → T2: SIMs per owner, mobile capped at 100")],
               [B("K1.3  Calendar levels mix WHEN with HOW FAR  "), M("(V1)")],
               [N("   → T3: own time t - t0 and own scale y/L;  T4: headroom")],
               [B("K1.4  Broadband floor; data MNAR  "), M("(V2; MNAR test)")],
               [N("   → T5: log broadband, floor and missingness flags (used by W3)")]],
           "[Parikshith, 0:00-0:40] The feedback loop of W1. Fiji becomes a rule applied to every country. Subscriptions "
           "are normalised by owners. And the key step: each curve is re-indexed to its own take-off year and divided "
           "by its own saturation, and the gap is measured as headroom. V9 underneath shows what changed.",
           size=14, img="V9_feedback_transformation.png", split=4)

fig_slide("V1c  Re-indexed to its own take-off, every country follows one S-curve: the divide is a delay",
          "V1c_takeoff_alignment.png", "W1 · ITER 2", "V", ["D2", "V2", "M2"], P, {
              "Observation": "Cross-country spread 32 pp in calendar time → 13.6 in own time → 9.7 in own time and "
                             "scale. Six regional medians lie on one curve.",
              "Cause": "Regions differ in WHEN they took off (2006 to 2017), not in the shape of adoption.",
              "Inference": "H1.1 supported: the divide is a 6-12 year delay on one common S-curve."},
          "[Parikshith, 0:00-0:40] Iteration 2, step 6. On the left, 201 curves in calendar time. In the middle, the "
          "same curves in each country's own time and scale: they collapse onto one S-curve, and the spread between "
          "countries falls from 32 to under 10 points. The six regional medians sit on top of each other. The divide "
          "is a delay, not a different curve.")

fig_slide("V1e  Closing only in relative terms: the countries furthest behind close the smallest share of their gap",
          "V1e_headroom_closing.png", "W1 · ITER 2", "V", ["D2", "V2", "M2", "K2"], P, {
              "Observation": "Raw: Gini 0.81 → 0.19, but the Europe-Africa gap peaked at 64 pp in 2014, still 52. "
                             "Headroom: rho = +0.52; Africa closed a median 32% of its gap, Europe 67%.",
              "Cause": "Levels are bounded at 100%: leaders saturate, so relative measures shrink on their own.",
              "Inference": "H1.2 refined: closing in relative terms only; the furthest behind close least."},
          "[Parikshith, 0:00-0:35] Step 7. Is the divide closing? On raw levels the two measures disagree. Measured "
          "as headroom, the share of the remaining gap each country closed, Africa closed a third, Europe two "
          "thirds. It is closing in relative terms only.")

fig_slide("V1f  Leapfrogging did not close the divide: skipped landlines, saturated on mobile, under half online",
          "V1f_leapfrog_test.png", "W1 · ITER 2", "V", ["D2", "V2", "M2", "K2"], P, {
              "Observation": "22 countries never built landlines but are saturated on mobile: median 44% online vs "
                             "93% for landline builders (p = 1e-11). Landline peak rho 0.82, mobile 0.47.",
              "Cause": "The landline network stands for the investment that also built broadband (association, "
                       "not cause).",
              "Inference": "H1.5 rejected: leapfrogging happened, but it did not bring people online."},
          "[Parikshith, 0:00-0:40] Step 8 tests A1's own conclusion with the engineered features. Twenty-two "
          "countries, most of them African, skipped landlines and are saturated on mobile, yet only 44 percent "
          "are online, against 93 percent where landlines were built. Leapfrogging happened; it did not close the "
          "divide.")

# ------------------------------------------------------------------ W2 Sindhoor
fig_slide("W2  Unsupervised workflow: from an unstable snapshot to path archetypes under DTW",
          split_workflow("W2_workflow_unsupervised.png"), "W2", "Q", TRACK, S_, {
              "Observation": "H2.1 stable types: refined. H2.2 anomalies are real: refined. H2.3 same level = same "
                             "history: rejected. H2.4 paths order timing: refined.",
              "Cause": "One year cannot show a path, and cannot tell a recording error from a real market.",
              "Inference": "T turns snapshots into 26-year curves compared by dynamic time warping: a new "
                           "representation and a new distance."},
          "[Sindhoor, 0:00-0:40] My workflow, W2, unsupervised mining. Question and hypotheses at the top. Iteration 1 "
          "clusters the 2024 snapshot and tests how stable it is. The feedback loop turns each country into its "
          "whole curve and changes the distance to dynamic time warping. Iteration 2 clusters paths.")

fig_slide("V4  Two axes describe 82% of 2024 connectivity; the mobile axis is stretched by two countries",
          "V4_pca_biplot.png", "W2 · ITER 1", "M", ["D1", "V1", "M1"], S_, {
              "Observation": "PC1 (64%) = general connectivity; PC2 (18%) loads 0.96 on mobile alone. "
                             "Fiji and Hong Kong sit far above everyone.",
              "Cause": "Mobile subscriptions vary independently of the other four indicators, and two extreme "
                       "values dominate that variance.",
              "Inference": "One component is real connectivity; the other is partly an outlier effect to check."},
          "[Sindhoor, 0:00-0:30] Step 1. On the 2024 snapshot of 135 countries, PCA puts 64 percent of variance on one "
          "connectivity axis. The second axis is mobile alone, and Fiji and Hong Kong stretch it by themselves.")

fig_slide("V5  Three archetypes ordered by connectivity, separated by landlines and broadband",
          "V5_cluster_validation_profiles.png", "W2 · ITER 1", "M", ["M1", "V1"], S_, {
              "Observation": "k = 3 best silhouette in k = 3-6 (0.33). Medians: broadband 1 / 20 / 39 per 100. "
                             "Ward's 'best' cluster is just Fiji + Hong Kong.",
              "Cause": "Countries differ mostly in fixed infrastructure.",
              "Inference": "Three types (mobile-first, mixed, fixed-line legacy) reproduce A1 Fig. 23; how stable "
                           "are they?"},
          "[Sindhoor, 0:00-0:35] Step 2. I scanned k from 2 to 8 for k-means, GMM and Ward. k equal to 3 gives three "
          "archetypes ordered by connectivity: mobile-first, mixed and fixed-line legacy. Ward's high silhouette is "
          "fake: one cluster is just Fiji and Hong Kong.")

fig_slide("V5b  Snapshot clusters are only moderately stable: mixed and legacy share a cluster in 46% of resamples",
          "V5b_bootstrap_coassignment.png", "W2 · ITER 1", "M", ["M1", "V1"], S_, {
              "Observation": "200 bootstrap refits: mean ARI 0.64 ± 0.20. Mobile-first block stable (0.92); "
                             "mixed-legacy off-diagonal 0.46; 45% of pairs ambiguous.",
              "Cause": "In one year many countries sit between 'mixed' and 'legacy'.",
              "Inference": "H2.1 refined: three types exist, but the mixed-legacy boundary is not stable."},
          "[Sindhoor, 0:00-0:35] Step 3, stability. Each cell is how often two countries land in the same cluster over "
          "200 bootstrap refits. The mobile-first block is solid. But mixed and legacy countries share a cluster in 46 "
          "percent of refits: the boundary moves.")

fig_slide("V6  Both detectors rank Fiji first; most anomalies are multi-SIM markets",
          "V6_anomalies.png", "W2 · ITER 1", "M", ["M1", "V1"], S_, {
              "Observation": "Isolation Forest and LOF share 7 of their top 15 (Spearman 0.77). Fiji ranks first "
                             "in both; Monaco for landlines.",
              "Cause": "Multi-SIM inflation and one recording error create the most isolated points.",
              "Inference": "Is each anomaly a country or an error? One year cannot tell."},
          "[Sindhoor, 0:00-0:30] Step 4. Isolation Forest and LOF, rank-averaged. Both put Fiji first; most others are "
          "multi-SIM markets, and Monaco has 95 landlines per 100.")

fig_slide("V6b  Four anomalies persist every year (structural); Fiji appears only in 2023-24 (artefact)",
          "V6b_anomaly_persistence.png", "W2 · ITER 1", "M", ["M1", "V1", "K1"], S_, {
              "Observation": "Isolation Forest refitted each year 2019-24: Monaco, Hong Kong, Andorra, Burundi flagged "
                             "every year; 16 transient, Fiji only 2023-24; 164 never.",
              "Cause": "A real market is unusual every year; a recording error appears and disappears.",
              "Inference": "H2.2 refined: keep structural anomalies, remove transient ones by rule (shared T1)."},
          "[Sindhoor, 0:00-0:35] Step 4 over time. I refitted the detector every year. Monaco, Hong Kong, Andorra and "
          "Burundi are flagged every year: real markets. Fiji appears only in 2023 and 2024: a recording error. That "
          "becomes the shared artefact rule.")

fig_slide("V6c  Each top anomaly is unusual on one indicator",
          "V6c_anomaly_why.png", "W2 · ITER 1", "M", ["M1", "V1", "K1"], S_, {
              "Observation": "Fiji +8.1 SD and Hong Kong +4.1 on mobile; Monaco +4.0 and Andorra +2.1 on landlines; "
                             "Uganda -1.6 on internet and ownership.",
              "Cause": "Multi-SIM markets and legacy landlines, plus one recording error.",
              "Inference": "The anomaly explanations match W1's V3: subscriptions are not people."},
          "[Sindhoor, 0:00-0:30] Step 5, why. Each bar is how far a country is from its own cluster. Fiji and Hong Kong "
          "on mobile, Monaco and Andorra on landlines.")

text_slide("W2 feedback loop: snapshots become trajectories under a time-shift-invariant distance", "W2 · LOOP",
           "LOOP", ["K1", "LOOP", "D2"], S_, [
               [B("K1.1  Snapshot clusters only moderately stable  "), M("(V5b)")],
               [N("   → T2: each country becomes its 2000-2025 curve (181 countries)")],
               [B("K1.2  Fiji transient = artefact; Monaco structural  "), M("(V6b, V6c)")],
               [N("   → T1: shared artefact rule; structural anomalies kept")],
               [B("K1.3  One year cannot show a path  "), M("(A1 Fig. 5)")],
               [N("   → T3: Euclidean → dynamic time warping (band 5 yr)")],
               [N("   → T4: average-linkage hierarchy")],
               [M("Below, V8a: why DTW. Brazil and Bolivia follow one curve 6.6 years apart; DTW scores them 5x closer.")]],
           "[Sindhoor, 0:00-0:40] My feedback loop. Each country becomes its whole curve from 2000 to 2025, artefacts "
           "removed. And the distance changes to dynamic time warping: below, Brazil and Bolivia follow the same curve "
           "almost seven years apart, and DTW scores them five times closer than a same-year comparison.",
           size=14, img="V8a_dtw_alignment.png", split=4)

fig_slide("V8  Grouped by PATH, countries form three S-curve archetypes that cut across the 2024 snapshot",
          "V8_trajectory_archetypes.png", "W2 · ITER 2", "M", ["D2", "M2", "V2"], S_, {
              "Observation": "Late starters (43), S-curve majority (113), early saturators (25). Silhouette 0.52 vs "
                             "0.35; 33 of 51 African countries are late starters.",
              "Cause": "Countries at the same 2024 level reached it along different paths.",
              "Inference": "Archetypes are paths, not positions: A1's 'same place, opposite histories' found by a "
                           "model."},
          "[Sindhoor, 0:00-0:40] Step 7. Three path archetypes: late starters like Tanzania, the S-curve majority like "
          "the Maldives, early saturators like South Korea. They separate far better than the snapshot clusters, "
          "silhouette 0.52 against 0.35.")

fig_slide("V8d  The feedback loop regroups 37 of 133 countries",
          "V8d_iteration_embedding.png", "W2 · ITER 2", "M", ["M2", "V2", "K2"], S_, {
              "Observation": "ARI 0.30 between iterations. India, Indonesia: low → majority; Italy, Spain: legacy → "
                             "majority; Sweden, Norway: mixed → early saturators.",
              "Cause": "A single year hides when and how fast a country got where it is.",
              "Inference": "H2.3 rejected: same level today does not mean same history."},
          "[Sindhoor, 0:00-0:35] Step 8, what the loop changed. The same map, coloured by iteration 1 and iteration 2. "
          "37 of 133 countries change group: India and Indonesia move to the majority, Sweden and Norway become early "
          "saturators.")

# ------------------------------------------------------------------ W3 Varshith
fig_slide("W3  Supervised workflow: history features, fair comparison, drift and re-calibrated forecasts",
          split_workflow("W3_workflow_supervised.png"), "W3", "Q", TRACK, V_, {
              "Observation": "H3.1 raw infrastructure predicts well: refined. H3.2 history lowers error: supported. "
                             "H3.3 2020 model gives trustworthy intervals: rejected.",
              "Cause": "Raw inputs carry no history, and the world after 2020 differs from the training years.",
              "Inference": "T adds 13 history features defined by W1 and in-fold imputation; an inner loop "
                           "re-calibrates when drift crosses 0.2."},
          "[Varshith, 0:00-0:40] My workflow, W3, supervised learning. Three hypotheses at the top. Iteration 1 is a "
          "raw baseline. The feedback loop adds history features, defined from W1's inferences, and imputation inside "
          "each training fold. Iteration 2 compares fairly, explains the model, checks drift and forecasts.")

fig_slide("V7  A raw-infrastructure model explains 86% of variance but leans on broadband alone",
          "V7_baseline_diagnostics.png", "W3 · ITER 1", "M", ["D1", "M1", "V1"], V_, {
              "Observation": "Grouped 5-fold CV: HistGBM RMSE 11.8 pp, R² 0.856. Error 12-14 pp above 10% online. "
                             "Shuffling mobile costs 0.02 R².",
              "Cause": "Raw inputs carry no history and inflated mobile counts.",
              "Inference": "H3.1 refined: prediction fails where histories differ most."},
          "[Varshith, 0:00-0:40] Step 1. We predict internet share from raw broadband, mobile, landlines and year. "
          "Folds hold out whole countries. Gradient boosting gets 11.8 points; broadband dominates, mobile adds almost "
          "nothing, and errors are large in mid-adoption countries.")

text_slide("W3 feedback loop: new predictors and new rows, then a drift-triggered re-calibration", "W3 · LOOP",
           "LOOP", ["K1", "LOOP", "D2"], V_, [
               [B("K1.1  Broadband alone carries the model  "), M("(V7)")],
               [N("   → T2: 13 past-only features from W1's K1 (landline peak, leapfrog, log broadband ...)")],
               [B("K1.2  Errors largest where histories differ  "), M("(V7)")],
               [N("   → (same features)")],
               [B("K1.3  Complete cases drop MNAR rows  "), M("(W1 MNAR test)")],
               [N("   → T3: MICE inside each training fold + flags: 2,134 more rows")],
               [B("Inner loop  "), N("drift chart > 0.2 → conformal re-calibration")]],
           "[Varshith, 0:00-0:35] My feedback loop adds thirteen past-only features, defined from W1's inferences, "
           "and imputation fitted inside each training fold, which adds 2,134 rows. And an inner loop: when the drift "
           "chart crosses 0.2, the forecast intervals are re-calibrated.",
           size=14, img="V9_feedback_transformation.png", split=4)

fig_slide("V10  On identical rows and folds, the transformed data lower error by 14%, most in Asia and Africa",
          "V10_iteration_comparison.png", "W3 · ITER 2", "M", ["D2", "M2", "V2"], V_, {
              "Observation": "RMSE 11.77 → 10.16 on the same 4,089 rows; 8.84 on all 6,223. Asia -21%, Africa "
                             "-18%, Americas +1%.",
              "Cause": "History features help where paths differ most; ablations show features, not imputation, "
                       "carry the gain.",
              "Inference": "H3.2 supported, fairly measured: same countries, same folds, same rows."},
          "[Varshith, 0:00-0:40] Iteration 2, step 2. Same countries, same folds, same rows. Error falls from 11.77 to "
          "10.16 points, minus 14 percent, most in Asia and Africa. With imputation, error reaches 8.84.")

fig_slide("V11  Ownership, broadband and the landline peak drive the model; its errors are geographic",
          "../outputs/slide_assets/V11_combined.png", "W3 · ITER 2", "V", ["M2", "V2"], V_, {
              "Observation": "SHAP: phone ownership, log broadband, landline peak. Blue (more online than predicted): "
                             "Gulf, Cuba. Red: Mongolia, China, Egypt.",
              "Cause": "Ownership is mostly imputed (a composite index); residual geography reflects policy and "
                       "price outside the data.",
              "Inference": "Infrastructure explains most of adoption; the rest is regional and needs other data."},
          "[Varshith, 0:00-0:35] Step 3. SHAP ranks phone ownership, log broadband and the landline peak on top; "
          "ownership is mostly imputed, so it acts as a composite index. The map shows where infrastructure under- or "
          "over-explains adoption.")

fig_slide("V12b  The model learned W1's broadband floor and a history effect; only conformal intervals are calibrated",
          "V12b_pdp_calibration.png", "W3 · ITER 2", "M", ["M2", "V2"], V_, {
              "Observation": "Partial dependence: flat below ~3 broadband per 100, rising to 30 (14 pp); landline peak "
                             "+12 pp. Raw coverage 27-68% at nominal 50-90%; CQR 50-94%.",
              "Cause": "Gradient boosting recovers the V2 floor; post-2020 drift breaks raw quantiles at every "
                       "level.",
              "Inference": "The model encodes W1's findings explicitly; its uncertainty is trustworthy only after "
                           "re-calibration."},
          "[Varshith, 0:00-0:35] Step 4. Partial dependence: predictions stay flat while broadband is on the floor, "
          "exactly W1's finding. And the calibration curve: raw intervals are over-confident at every level, the "
          "conformal ones sit on the diagonal.")

fig_slide("V13  After 2020 the target drifts past the retraining threshold while the inputs do not",
          "V13_psi_drift_chart.png", "W3 · ITER 2", "V", ["D2", "V2"], V_, {
              "Observation": "PSI vs 2015-20: target 0.44 in 2022-24 (> 0.2); every input < 0.2 until 2023.",
              "Cause": "Post-2020 acceleration: the same infrastructure now coincides with more people online.",
              "Inference": "Concept drift: a model trained to 2020 will forecast too low and too confidently."},
          "[Varshith, 0:00-0:35] Step 5, the drift chart. Population Stability Index against a 2015 to 2020 baseline. "
          "After 2020 the target crosses 0.2 while the inputs stay below: same infrastructure, more people online.")

fig_slide("V12  Forecasts made in 2020 run low; conformal re-calibration restores the cone of uncertainty",
          "V12_forecast_fan_chart.png", "W3 · ITER 2", "M", ["M2", "V2", "K2"], V_, {
              "Observation": f"Holdout RMSE {fm['holdout_rmse']:.1f} pp; {fm['share_underpredicted']:.0%} too low; "
                             f"80% band covers {fm['coverage_80_quantile']:.0%}.",
              "Cause": "Exactly the drift V13 flagged: the post-2020 shift is absent from the training years.",
              "Inference": f"H3.3 rejected for raw intervals; CQR (+{fm['cqr_margin_pp']:.1f} pp) restores "
                           f"{fm['coverage_80_cqr']:.0%}."},
          f"[Varshith, 0:00-0:40] Step 6. Trained up to 2020, tested on 2021 to 2025: {fm['share_underpredicted']:.0%} "
          f"of forecasts are too low and the 80 percent band covers only {fm['coverage_80_quantile']:.0%}. Conformal "
          f"re-calibration restores {fm['coverage_80_cqr']:.0%}.")

# ------------------------------------------------------------------ synthesis
fig_slide("V14  Answer: the divide is a delay on one S-curve, and fixed infrastructure, not mobile, marks the position",
          "V14_synthesis_answer.png", "SYNTHESIS", "K", ["K2"], P, {
              "Observation": "W1: take-off 6-12 yr behind Europe. W1+W3: 2023 rho broadband 0.79, landline peak "
                             "0.81, mobile 0.52. W2: broadband 0 / 17 / 40 by archetype.",
              "Cause": "Mobile saturated everywhere; fixed infrastructure still records when and how far a country "
                       "moved.",
              "Inference": "Mobile leapfrogging happened but did not close the divide. Every feedback loop improved "
                           "its check."},
          "[Parikshith, 0:00-0:45] Putting the three workflows together. When, from W1: every region is on the same "
          "S-curve, 6 to 12 years behind Europe. What: broadband and the landline peak separate countries today, "
          "mobile does not. Who, from W2: the archetypes differ in fixed infrastructure. And every feedback loop "
          "improved its check: cluster separation, error on the same rows, and forecast coverage.")

text_slide("What we know now, and what we cannot claim", "KNOWLEDGE", "K", ["K2"], V_, [
    [B("Answer.  "), N("Mobile leapfrogging happened but did not close the divide: a 6-12 year delay on one S-curve, "
                       "closing only in relative terms; fixed infrastructure, not mobile, marks a country's position.")],
    [B("Second iterations learned  "), N("W1: one curve once aligned (spread 32 → 10 pp). W2: archetypes are paths "
                                         "(37 of 133 regrouped). W3: history cuts error 14%; drift needs "
                                         "re-calibration (50% → 83% coverage).")],
    [M(" ")],
    [B("Limits.  "), N("Association, not cause (no income variable, by design). Ownership mostly imputed. Snapshot "
                       "clusters moderately stable (ARI 0.64). Conformal coverage assumes no further drift.")],
    [M(" ")],
    [M("Reproduce: python src/run_all.py && python src/build_diagrams.py  ·  Demo: streamlit run app.py")]],
    "[Varshith, 0:00-0:30] To close: mobile leapfrogging happened but did not close the divide. We cannot claim "
    "causation, ownership is mostly imputed, and the forecasts assume no further drift. Everything is reproducible. "
    "Thank you.", size=18)

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
         f"Length {t // 60}:{t % 60:02d}. Full deck (for the viva). The 6-minute video uses DAS732_A3_DataStory_6min.pptx. "
         "Optional: switch to the app (https://aithal007.github.io/DAS732-A3-Visual-Analytics/ or `streamlit run app.py`) for 10-15 seconds, "
         "then return. Every report figure appears on a slide.", ""]
for i, sl in enumerate(prs.slides, 1):
    title = next((sh.text_frame.text.splitlines()[0] for sh in sl.shapes
                  if sh.has_text_frame and len(sh.text_frame.text) > 20), "")
    lines += [f"## Slide {i}: {title}", "", sl.notes_slide.notes_text_frame.text, ""]
(ROOT / "video" / "VIDEO_SCRIPT.md").write_text("\n".join(lines), encoding="utf8")

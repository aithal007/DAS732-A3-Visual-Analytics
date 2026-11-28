"""6-minute video cut: deck + shot-by-shot recording script (slides AND the live app on localhost).

Reads the full deck built by build_slides.py, keeps 16 story slides in order, moves 8 supporting slides
to the end as hidden appendix slides, writes per-slide speaker notes, and writes
video/VIDEO_SCRIPT_6min.md: every shot has a time, speaker, screen (slide or app tab), action and words.

Timing model: speech at 150 words/min (2.5 words/s), +1 s per slide change, +3 s per switch between
PowerPoint and the browser, plus any silent click time given per shot.

    python src/build_slides.py && python src/build_slides_6min.py
"""
import math
import re
from pathlib import Path

from pptx import Presentation

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "slides" / "DAS732_A3_DataStory.pptx"
OUT = ROOT / "slides" / "DAS732_A3_DataStory_6min.pptx"
WPS = 2.5
LOCAL = "http://localhost:8501"
LIVE = "https://aithal007.github.io/DAS732-A3-Visual-Analytics/"

# shot = (full-deck slide no. or app tab, speaker, action, words, silent seconds)
# slide shots show that slide; "APP:<tab>" shots show the browser on that tab of the app.
SHOTS = [
    (1, "Parikshith", "Slideshow on slide 1 (title).",
     "We are Team Plot Twist. Our question from A1: did mobile leapfrogging close the digital divide? In A3 we "
     "test what we saw with statistics, clustering and prediction, inside a visual analytics workflow with one "
     "feedback loop.", 0),
    (2, "Varshith", "Next slide (the question and four sub-questions).",
     "The brief asks for statistics, unsupervised mining and supervised learning inside a workflow with a feedback "
     "loop. So we split the question into sub-questions, one per method family, plus the workflow itself. The data "
     "is A1's OWID panel: 217 countries, 1960 to 2025.", 0),
    ("APP:Workflow", "Parikshith", "Alt+Tab to the browser (app already open on the Workflow tab). Move the mouse "
     "along row 1, rest it on the red FEEDBACK LOOP edge, then along row 2. Scroll down to the system "
     "architecture diagram.",
     "This is our live app, running locally. The workflow is Keim's diamond unrolled into two rows. Iteration 1 "
     "runs on raw data and ends with four inferences, K1. The red edge is the one feedback loop: K1 becomes a data "
     "transformation, and iteration 2 repeats the whole diamond on the transformed data. Below, one shared core "
     "module implements that loop, so one command reproduces everything.", 0),
    (5, "Parikshith", "Alt+Tab back to PowerPoint; advance to V1 (slide 4).",
     "A logistic S-curve fits 201 of 203 countries. Africa takes off in 2017, almost 12 years after Europe, p of 4 "
     "times 10 to the minus 17, and later starters do not grow faster. The divide is a delay on one curve.", 0),
    (6, "Parikshith", "Next slide (V1b).",
     "Is it closing? Relative inequality, the Gini, fell from 0.81 to 0.19, but the Europe-Africa gap peaked at 64 "
     "points in 2014 and is still 52. Low starters close the smallest share of their gap: no catch-up.", 0),
    (7, "Parikshith", "Next slide (V2).",
     "Nearly 30 percent of country-years have under one broadband line per 100, yet internet reaches 30 percent: a "
     "floor. Mobile's correlation with internet use fell from 0.88 to 0.42, while broadband stays at 0.80. Mobile "
     "saturated.", 0),
    (8, "Parikshith", "Next slide (V3). Point at Fiji's red line on the right panel.",
     "And mobile counts SIMs, not people: 1.34 per owner. Fiji jumps from 111 to 574 subscriptions in two years "
     "with ownership flat at 82 percent: a recording error.", 0),
    (11, "Sindhoor", "Next slide (V5).",
     "Next, unsupervised mining of the 2024 snapshot. k-means gives three archetypes ordered by connectivity, "
     "separated by landlines and broadband, not mobile. Bootstrap ARI is only 0.64, and Ward's best silhouette is "
     "fake: one cluster is just Fiji and Hong Kong.", 0),
    ("APP:Mine", "Sindhoor", "Alt+Tab to the browser, click the Mine tab. Click WARD under 'Clustering method': the "
     "tiny cluster is the two light-blue bubbles. Hover the left one at about 75% (Fiji), then point at row 1 of "
     "the 'Top anomalies' table (Fiji).",
     "In the app, on the Mine tab, I switch from k-means to Ward, and the tiny cluster appears: Fiji and "
     "Hong Kong. And Fiji is the top anomaly for both Isolation Forest and LOF.", 3),
    (13, "Varshith", "Alt+Tab back to PowerPoint; advance to V7 (slide 10).",
     "Then supervised learning: predict internet share from raw infrastructure. Folds hold out whole countries, so "
     "the model is always tested on a country it has never seen. Gradient boosting reaches 11.8 points error, "
     "R-squared 0.86. Broadband dominates, mobile adds almost nothing.", 0),
    (14, "Sindhoor", "Next slide (feedback loop, V9 underneath). Point at each K1 line as you say it.",
     "This is the feedback loop. Four inferences leave iteration 1, and each one changes the data. Fiji is an "
     "error: two artefact rules remove 10 cells. Subscriptions are not people: normalise by owners and cap mobile "
     "at 100. History matters: add the landline peak and reshape to trajectories. Broadband has a floor and the "
     "gaps are not random: log transform, a floor flag, and in-fold imputation.", 0),
    ("APP:Feedback loop", "Sindhoor", "Alt+Tab to the browser, click the Feedback loop tab (Fiji is preselected).",
     "In the app, Fiji's raw series in red against the cleaned one in blue: the two impossible years are gone, "
     "the rest of the series is untouched.", 2),
    (16, "Sindhoor", "Alt+Tab back to PowerPoint; advance to V8 (slide 12).",
     "Iteration 2 clusters each country's whole curve with dynamic time warping: late starters like Tanzania, the "
     "S-curve majority, and early saturators like South Korea. Two thirds of African countries are late starters. Silhouette rises to 0.52 against 0.35, and 37 of "
     "133 countries change group.", 0),
    ("APP:Archetypes", "Sindhoor", "Click the Archetypes tab. Hover the orange end of the t-SNE map (late "
     "starters), then hover India near the orange/blue boundary. Right panel already shows T0 late starters with "
     "India in black; no click needed.",
     "And the trajectory map: late starters sit at one end of a single continuum. India is on the boundary; "
     "against the late starters, its curve has just broken away.", 3),
    (18, "Varshith", "Alt+Tab back to PowerPoint; advance to V10 (slide 13).",
     "Same countries, same folds, same rows: error falls from 11.77 to 10.16 points, minus 14 percent, most in "
     "Asia and Africa. Imputation adds 2,134 usable rows and error reaches 8.84.", 0),
    (21, "Varshith", "Next slide (V12).",
     "Finally, forecasting. Trained up to 2020 and tested on 2021 to 2025, 71 percent of forecasts run low and the "
     "80 percent band covers only 50 percent, because the target drifted after 2020 while the inputs did not. "
     "Conformal re-calibration restores 83 percent.", 0),
    ("APP:Forecast & drift", "Varshith", "Alt+Tab to the browser, click Forecast & drift (focus country India). "
     "Click 'raw quantile', then 'conformal (CQR)'.",
     "In the app: India's hidden points fall above the raw band; with conformal re-calibration they fall inside "
     "it.", 3),
    (23, "Parikshith", "Alt+Tab back to PowerPoint; advance to V14 (slide 15).",
     "Putting it together. When: every region is on the same S-curve, 6 to 12 years behind Europe. What: broadband "
     "and the landline peak separate countries today, mobile does not. Who: the archetypes differ in fixed "
     "infrastructure. And the feedback loop improved every check: error, cluster separation and forecast "
     "coverage.", 0),
    (24, "Varshith", "Next slide (conclusion). Stop recording after 'Thank you'.",
     "So mobile leapfrogging happened, but it did not close the divide. We cannot claim causation, ownership is "
     "mostly imputed, and forecasts assume no further drift. Everything is reproducible with one command. Thank "
     "you.", 0),
]
STORY = [1, 2, 3, 5, 6, 7, 8, 11, 12, 13, 14, 16, 18, 21, 23, 24]   # full-deck numbers kept in the video
APPENDIX = [4, 9, 10, 15, 17, 19, 20, 22]                          # V0b, V3b, V4, V9, V8b, V11, V13, V12b

prs = Presentation(FULL)
full_notes = {i: s.notes_slide.notes_text_frame.text for i, s in enumerate(prs.slides, 1)}
titles = {i: next((sh.text_frame.text.splitlines()[0] for sh in s.shapes
                   if sh.has_text_frame and len(sh.text_frame.text) > 20), "")
          for i, s in enumerate(prs.slides, 1)}
lst = prs.slides._sldIdLst
ids = list(lst)
order = STORY + APPENDIX
assert sorted(order) == list(range(1, len(ids) + 1)), "deck layout changed; update STORY/APPENDIX"
for el in ids:
    lst.remove(el)
for n in order:
    lst.append(ids[n - 1])
pos_of = {n: i for i, n in enumerate(order, 1)}

# ---- timing
t, share, rows, on_app, notes = 0, {}, [], False, {}
last_slide = None
for target, who, action, words, silent in SHOTS:
    app = isinstance(target, str)
    switch = 3 if app != on_app else (1 if not app else 0)
    dur = math.ceil(len(words.split()) / WPS) + switch + silent
    start, end = t, t + dur
    stamp = f"{start // 60}:{start % 60:02d}-{end // 60}:{end % 60:02d}"
    screen = (f"WEBSITE, {target[4:]} tab" if app else f"SLIDE {pos_of[target]}: {titles[target]}")
    rows.append((stamp, who, screen, action, words))
    share[who] = share.get(who, 0) + dur
    if app:
        notes.setdefault(last_slide, []).append(f"[{who}, {stamp}] >>> SWITCH TO APP, {target[4:]} tab: "
                                                f"{action} SAY: {words}")
    else:
        last_slide = target
        notes.setdefault(target, []).append(f"[{who}, {stamp}] {words}")
    t, on_app = end, app

for pos, n in enumerate(order, 1):
    slide = prs.slides[pos - 1]
    tf = slide.notes_slide.notes_text_frame
    if n in STORY:
        tf.text = "\n\n".join(notes.get(n, []))
    else:
        slide._element.set("show", "0")
        tf.text = "[APPENDIX, hidden slide, not in the video. Use for questions.] " + \
                  re.sub(r"^\[[^\]]*\]\s*", "", full_notes[n])
prs.save(OUT)

# ---- script
L = ["# A3 video: 6-minute shot-by-shot script", "",
     f"Total **{t // 60}:{t % 60:02d}** (speech at 150 words per minute, plus slide changes, app switches and "
     "clicks). Generated by `src/build_slides_6min.py`; the same text is in the speaker notes of "
     f"`slides/{OUT.name}`.", "",
     "## Before recording (5 minutes)", "",
     "1. Start the app locally, in a terminal:",
     "   ```", "   cd ass3/A3_submission", "   streamlit run app.py", "   ```",
     f"   It opens **{LOCAL}**. Backup if it fails: the live copy at {LIVE} (first load 20-60 s).",
     "2. In the browser: click the **Workflow** tab, scroll to the top, sidebar *Focus country* = **India**, "
     "*Model views show* = **Iteration 2**. Zoom the browser to 90% so whole charts fit.",
     "3. Open `slides/DAS732_A3_DataStory_6min.pptx` in **Slide Show** (F5) on slide 1. Hidden slides 17-24 are "
     "skipped automatically; keep them for questions.",
     "4. Only two windows open: PowerPoint and the browser. **Alt+Tab** switches between them.",
     "5. Record the whole screen with audio: OBS Studio (Display Capture + microphone), or a Teams/Zoom meeting "
     "with screen share and Record. One person shares and clicks; each speaker unmutes for their part.",
     "6. Rehearse once with a timer. If you run long, drop a click, not a sentence.", "",
     "## Shot list", "",
     "| # | Time | Speaker | Screen |", "|---|---|---|---|"]
for i, (stamp, who, screen, action, words) in enumerate(rows, 1):
    L.append(f"| {i} | {stamp} | {who} | {screen} |")
L += ["", "## Shots in detail", ""]
for i, (stamp, who, screen, action, words) in enumerate(rows, 1):
    L += [f"### {i}. {stamp} · {who} · {screen}", "", f"**Do:** {action}", "", f"**Say:** \"{words}\"", ""]
L += ["## Hidden appendix slides (not in the video, for questions)", ""]
for pos, n in enumerate(order, 1):
    if n in APPENDIX:
        L.append(f"- Slide {pos}: {titles[n]}")
(ROOT / "video" / "VIDEO_SCRIPT_6min.md").write_text("\n".join(L) + "\n", encoding="utf8")
print(f"saved {OUT.name}; {len(rows)} shots; total {t // 60}:{t % 60:02d}",
      {k: f"{v // 60}:{v % 60:02d}" for k, v in share.items()},
      "app shots:", sum(1 for r in rows if r[2].startswith("WEBSITE")))

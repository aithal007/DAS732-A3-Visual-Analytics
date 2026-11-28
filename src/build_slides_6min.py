"""6-minute video cut: deck + shot-by-shot recording script (slides AND the live app on localhost).

Reads the full deck built by build_slides.py, keeps 20 story slides in order, moves 15 supporting slides
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
     "We are Team Plot Twist. Our question from A1: did mobile leapfrogging close the digital divide? In A3 we test "
     "it with three visual analytics workflows, one per team member.", 0),
    (2, "Varshith", "Next slide (main question, three workflows).",
     "Each workflow uses one family of methods the brief asks for: statistics, unsupervised mining and supervised "
     "learning. Each one has its question and hypotheses written on it, its figures plugged in, a numbered analysis "
     "path, and one feedback loop. A synthesis combines them.", 0),
    (3, "Parikshith", "Next slide (V0, main workflow). Trace one lane left to right with the mouse.",
     "This is the main workflow: Keim's diamond, unrolled. Each lane is one workflow. Iteration 1 on raw data, its "
     "inferences, then the red feedback loop that transforms the data, then iteration 2 and its knowledge. All three "
     "lanes feed one synthesis.", 0),
    ("APP:Workflow", "Parikshith", "Alt+Tab to the browser (app on the Workflow tab). Choose 'W1 Statistics' in the "
     "diagram selector. Point at the hypotheses, then at thumbnails 1 to 4, then at the red band.",
     "In our app, this is my workflow, W1. The question and four hypotheses are written at the top, each with its "
     "verdict, and the figures are plugged in, numbered in the order I read them.", 3),
    (7, "Parikshith", "Alt+Tab back to PowerPoint; advance to V1.",
     "A logistic S-curve fits 201 of 203 countries. Africa takes off in 2017, almost 12 years after Europe, "
     "and the test is highly significant. Is the divide only a delay? In calendar time we cannot tell.", 0),
    (9, "Parikshith", "Next slide (V3). Point at Fiji's red line.",
     "Mobile counts SIMs, not people: 1.34 per owner. And Fiji jumps from 111 to 574 subscriptions in two years "
     "with ownership flat: a recording error.", 0),
    (10, "Parikshith", "Next slide (W1 feedback loop, V9 underneath).",
     "So my feedback loop changes the data, not just filters it. Artefacts are removed by rule, mobile is "
     "normalised per owner, and every curve is re-indexed to its own take-off year and its own saturation level.", 0),
    (11, "Parikshith", "Next slide (V1c). Point at panel b, then panel d.",
     "Iteration 2. In their own time, 201 curves collapse onto one S-curve, and the spread between countries falls "
     "from 32 to under 10 points. The divide is a delay.", 0),
    (12, "Parikshith", "Next slide (V1b). Point at panel c.",
     "And in headroom terms, the countries furthest behind close the smallest share of their gap: it is closing in "
     "relative terms only.", 0),
    (13, "Sindhoor", "Next slide (W2 workflow).",
     "My workflow, W2, clusters countries. Its question: which kinds of countries exist, how stable are those kinds, "
     "and which countries fit none of them? Here are my hypotheses, my numbered path and my feedback loop.", 0),
    (16, "Sindhoor", "Next slide (V5b). Point at the mixed-legacy off-diagonal block.",
     "The 2024 snapshot gives three clusters, but they are only moderately stable: over 200 bootstrap refits, mixed "
     "and legacy countries share a cluster 46 percent of the time, and 15 countries sit right on a cluster border.", 0),
    (19, "Sindhoor", "Next slide (V6b). Point at the red rows, then at Fiji.",
     "For anomalies, I refitted the detector every year. Monaco and Hong Kong are flagged every year: real markets. "
     "Fiji is flagged only in 2023 and 2024: a recording error, which becomes the shared artefact rule.", 0),
    ("APP:Mine", "Sindhoor", "Alt+Tab to the browser, click the Mine tab. Click WARD under 'Clustering method'; "
     "hover the two light-blue bubbles (Fiji, Hong Kong); point at row 1 of 'Top anomalies'.",
     "In the app, switching to Ward shows a tiny cluster of just Fiji and Hong Kong, and Fiji is the top anomaly "
     "for both detectors.", 3),
    (21, "Sindhoor", "Alt+Tab back to PowerPoint; advance to the W2 feedback-loop slide (V8a underneath).",
     "My feedback loop turns each country into its whole curve, from 2000 to 2025, and compares curves with dynamic "
     "time warping. Brazil and Bolivia follow one curve seven years apart, and DTW scores them five times closer.", 0),
    (23, "Sindhoor", "Next slide (V8).",
     "In iteration 2, late starters split off at the root of the tree, and we get three path archetypes: late "
     "starters, the S-curve majority and early saturators. They "
     "separate far better than the snapshot clusters, silhouette 0.52 against 0.35.", 0),
    (25, "Sindhoor", "Next slide (V8d). Point at the ringed countries.",
     "And the loop changed the answer: 37 of 133 countries regroup. India and Indonesia join the majority, and "
     "Sweden and Norway become early saturators.", 0),
    ("APP:Archetypes", "Sindhoor", "Alt+Tab to the browser, click the Archetypes tab. Hover India near the "
     "orange/blue boundary of the t-SNE map.",
     "On the trajectory map, India sits right on the boundary between the late starters and the majority.", 3),
    (26, "Varshith", "Alt+Tab back to PowerPoint; advance to W3 workflow.",
     "My workflow, W3, predicts internet use. The raw baseline explains 86 percent of variance for unseen "
     "countries, but leans on broadband alone. My feedback loop adds 13 history features defined by W1, and "
     "imputation inside each training fold.", 0),
    (29, "Varshith", "Next slide (V10).",
     "Same countries, same folds, same rows: error falls from 11.77 to 10.16 points, minus 14 percent, most in Asia "
     "and Africa.", 0),
    (32, "Varshith", "Next slide (V13).",
     "Then the drift chart: after 2020 the target shifts past the 0.2 threshold while the inputs do not, so a model "
     "trained to 2020 will forecast too low: the same infrastructure now means more people online.", 0),
    (33, "Varshith", "Next slide (V12).",
     "And it does: the raw 80 percent band covers only 50 percent of outcomes. Conformal re-calibration restores 83 "
     "percent.", 0),
    ("APP:Forecast & drift", "Varshith", "Alt+Tab to the browser, click Forecast & drift (India). Click 'raw "
     "quantile', then 'conformal (CQR)'.",
     "In the app, India's hidden points fall above the raw band, and inside the conformal one.", 3),
    (34, "Parikshith", "Alt+Tab back to PowerPoint; advance to V14.",
     "Putting the three workflows together: every region is on one S-curve, 6 to 12 years behind Europe; "
     "broadband and the landline peak, not mobile, separate countries today; and every feedback loop improved its "
     "check.", 0),
    (35, "Varshith", "Next slide (conclusion). Stop recording after 'Thank you'.",
     "So mobile leapfrogging happened, but it did not close the divide. We cannot claim causation, and the "
     "forecasts assume no further drift. Thank you.", 0),
]
STORY = [1, 2, 3, 7, 9, 10, 11, 12, 13, 16, 19, 21, 23, 25, 26, 29, 32, 33, 34, 35]   # full-deck numbers in the video
APPENDIX = [4, 5, 6, 8, 14, 15, 17, 18, 20, 22, 24, 27, 28, 30, 31]                   # hidden, for questions

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
     "2. In the browser: click the **Workflow** tab (diagram selector on *Main workflow V0*), scroll to the top, sidebar *Focus country* = **India**, "
     "*Model views show* = **Iteration 2**. Zoom the browser to 90% so whole charts fit.",
     "3. Open `slides/DAS732_A3_DataStory_6min.pptx` in **Slide Show** (F5) on slide 1. Hidden slides 21-35 are "
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

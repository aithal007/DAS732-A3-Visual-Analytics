"""6-minute video cut: deck + shot-by-shot recording script (slides AND the live app on localhost).

Reads the full deck built by build_slides.py, keeps 20 story slides in order, moves 12 supporting slides
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
     "We are Team Plot Twist. Our question from A1: did mobile leapfrogging close the digital divide? We test it "
     "with three visual analytics workflows, one per team member.", 0),
    (2, "Varshith", "Next slide (main question, three workflows).",
     "Statistics, unsupervised mining and supervised learning: each workflow has its own hypotheses, its figures "
     "plugged in, a numbered analysis path and one feedback loop. A synthesis combines them.", 0),
    (3, "Parikshith", "Next slide (V0). Trace one lane left to right with the mouse.",
     "This is Keim's diamond, unrolled. Each lane runs iteration 1 on raw data, then a red feedback loop that "
     "transforms the data, then iteration 2. All three lanes feed one answer.", 0),
    (4, "Parikshith", "Next slide (W1 diagram). Point at the hypotheses, then at the red band.",
     "My workflow, W1. Five hypotheses at the top, each with its verdict; the figures numbered in the order I read "
     "them; the red band is my feedback loop.", 0),
    (5, "Parikshith", "Next slide (V1).",
     "Every region follows an S-curve, but Africa takes off almost 12 years after Europe. Is the divide only a "
     "delay?", 0),
    (9, "Parikshith", "Next slide (W1 feedback loop).",
     "My feedback loop changes the data, not just filters it: errors removed by rule, and every curve re-indexed to "
     "its own take-off year and its own saturation level.", 0),
    (10, "Parikshith", "Next slide (V1c). Point at panel b, then panel d.",
     "In their own time, 201 curves collapse onto one S-curve, and the spread between countries falls from 32 to "
     "under 10 points. The divide is a delay.", 0),
    (12, "Parikshith", "Next slide (V1f). Point at the two medians in panel b.",
     "And A1's own question: countries that skipped landlines and saturated on mobile are 44 percent online, against "
     "93 where landlines were built.", 0),
    ("APP:What if?", "Parikshith", "Alt+Tab to the browser, click the 'What if?' tab. Scenario 1 is selected, "
     "country India. Click 'Europe's take-off'; then click 'Reset'.",
     "Now, what if? India took off 16 years after Europe. Give it Europe's take-off year, and its 2025 internet use "
     "jumps from 71 to 99 percent. Same curve, earlier start.", 4),
    ("APP:What if?", "Parikshith", "Click scenario 2 (broadband or mobile). Drag 'Mobile' to 100, then drag "
     "'Fixed broadband' to 20.",
     "And what if India pushed mobile instead? Mobile to 100 adds about three points. Broadband to 20 per 100 adds "
     "about 27. That is why leapfrogging did not close the divide.", 4),
    (13, "Sindhoor", "Alt+Tab back to PowerPoint; advance to W2.",
     "My workflow, W2, clusters countries: which kinds exist, how stable are they, and which countries fit none? "
     "Here are my hypotheses and my feedback loop.", 0),
    (16, "Sindhoor", "Next slide (V5b). Point at the mixed-legacy block.",
     "The 2024 snapshot gives three clusters, but over 200 bootstrap refits, mixed and legacy countries share a "
     "cluster 46 percent of the time: the boundary is unstable.", 0),
    (18, "Sindhoor", "Next slide (V6b). Point at the red rows, then at Fiji.",
     "I refitted the anomaly detector every year. Monaco and Hong Kong are flagged every year: real markets. Fiji only "
     "in 2023 and 2024: a recording error.", 0),
    ("APP:What if?", "Sindhoor", "Alt+Tab to the browser, What if? tab, scenario 3. Point at Fiji at the top of the "
     "map, then switch off 'Keep Fiji's 2024 error'.",
     "What if that error stayed in the data? Here Fiji alone stretches the second principal component. Switch it off "
     "and the map relaxes: connectivity explains 67 instead of 64 percent. So my loop removes it by rule.", 3),
    (20, "Sindhoor", "Alt+Tab back to PowerPoint; advance to the W2 feedback-loop slide.",
     "My feedback loop turns each country into its whole 2000 to 2025 curve and compares curves with dynamic time "
     "warping, so the same S-curve shifted in time counts as similar.", 0),
    (21, "Sindhoor", "Next slide (V8).",
     "Iteration 2 finds three path archetypes: late starters, the S-curve majority and early saturators, far better "
     "separated: silhouette 0.52 against 0.35.", 0),
    (22, "Sindhoor", "Next slide (V8d). Point at the ringed countries.",
     "And the loop changed the answer: 37 of 133 countries regroup. India and Indonesia join the majority; Sweden and "
     "Norway become early saturators.", 0),
    ("APP:Archetypes", "Sindhoor", "Alt+Tab to the browser, click the Archetypes tab. Hover India near the "
     "orange/blue boundary of the map; pick 'T0 late starters' on the right.",
     "In the app, the trajectory map: late starters sit at one end of a single continuum, and India sits right on "
     "the boundary. Against the late starters, its curve has just broken away.", 3),
    (23, "Varshith", "Alt+Tab back to PowerPoint; advance to W3.",
     "My workflow, W3, predicts internet use for countries the model has never seen. The raw baseline leans on "
     "broadband alone; my feedback loop adds 13 history features from W1 and imputation inside each training fold.", 0),
    (26, "Varshith", "Next slide (V10).",
     "Same countries, same folds, same rows: error falls from 11.77 to 10.16 points, minus 14 percent, most in Asia "
     "and Africa.", 0),
    (29, "Varshith", "Next slide (V13).",
     "Then the drift chart: after 2020 the target shifts past the 0.2 threshold while the inputs do not. A model "
     "trained to 2020 will forecast too low.", 0),
    (30, "Varshith", "Next slide (V12). Point at the red points above the bands.",
     "And it does: trained up to 2020, 71 percent of the 2021 to 2025 forecasts are too low, and the hidden red "
     "points fall above the band.", 0),
    ("APP:What if?", "Varshith", "Alt+Tab to the browser, What if? tab, scenario 4 (80% selected). Point at 50%, "
     "then switch on 'Apply conformal re-calibration'.",
     "What if we had trusted that model? Its 80 percent band covers only 50 percent of what happened. Switch on "
     "conformal re-calibration: 83 percent, on the diagonal at every level.", 3),
    (31, "Parikshith", "Alt+Tab back to PowerPoint; advance to V14.",
     "Together: every region is on one S-curve, 6 to 12 years behind Europe; broadband and the landline peak, not "
     "mobile, separate countries today; and every feedback loop improved its check.", 0),
    (32, "Varshith", "Next slide (conclusion). Stop recording after 'Thank you'.",
     "So mobile leapfrogging happened, but it did not close the divide. We cannot claim causation, and the forecasts "
     "assume no further drift. Thank you.", 0),
]
STORY = [1, 2, 3, 4, 5, 9, 10, 12, 13, 16, 18, 20, 21, 22, 23, 26, 29, 30, 31, 32]   # full-deck numbers in the video
APPENDIX = [6, 7, 8, 11, 14, 15, 17, 19, 24, 25, 27, 28]                    # hidden, for questions

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
     "2. In the browser: click the **What if?** tab (all four live demos are there), sidebar *Focus country* = "
     "**India**, *Model views show* = **Iteration 2**. Zoom the browser to 90% so whole charts fit. Click scenario 1 "
     "and leave it there.",
     "3. Open `slides/DAS732_A3_DataStory_6min.pptx` in **Slide Show** (F5) on slide 1. Hidden slides 21-32 are "
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

"""6-minute video cut of the data-story deck.

Reads the full deck built by build_slides.py (run that first), keeps 16 story slides in order, moves 8
supporting slides to the end as hidden appendix slides (slideshow mode skips them; they stay available
for questions), replaces the speaker notes with the 6-minute narration, and writes
video/VIDEO_SCRIPT_6min.md.  Timings: ~150 words per minute + 8 s per live demo + 1 s per transition.

    python src/build_slides.py && python src/build_slides_6min.py
"""
import copy
import math
import re
from pathlib import Path

from pptx import Presentation

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "slides" / "DAS732_A3_DataStory.pptx"
OUT = ROOT / "slides" / "DAS732_A3_DataStory_6min.pptx"
WPS = 2.5          # words per second (150 wpm)
DEMO_S = 8

# full-deck slide number -> (speaker, narration)
STORY = {
    1: ("Parikshith", "We are Team Plot Twist. Our question from A1: did mobile leapfrogging close the digital "
                      "divide? In A3 we test what we saw, with statistics, clustering and prediction, inside a "
                      "visual analytics workflow with one feedback loop."),
    2: ("Varshith", "The brief asks for statistics, unsupervised mining and supervised learning inside a workflow "
                    "with a feedback loop. So we split the question into sub-questions, one per method family, plus "
                    "the workflow itself. The data is A1's OWID panel: 217 countries, 1960 to 2025."),
    3: ("Parikshith", "Keim's diamond drawn as a cycle cannot show which pass produced which finding, so we unroll "
                      "it into two rows. Iteration 1 runs on the raw data and ends with four inferences, K1. The red "
                      "edge is the one feedback loop: K1 becomes a data transformation. Iteration 2 repeats data, "
                      "visualization, models and knowledge on the transformed data, with no self-loops."),
    5: ("Parikshith", "First, statistics. A logistic S-curve fits 201 of 203 countries. The take-off year differs "
                      "by region, p of 4 times 10 to the minus 17: Africa takes off in 2017, almost 12 years after "
                      "Europe. And a later take-off does not mean faster growth. The divide is a delay on one curve."),
    6: ("Parikshith", "Is that delay closing? Relative inequality, the Gini, fell from 0.81 to 0.19. But the "
                      "absolute gap between the median European and African country peaked at 64 points in 2014 "
                      "and is still 52, and low starters close the smallest share of their remaining gap. There is "
                      "no catch-up."),
    7: ("Parikshith", "What separates countries? Nearly 30 percent of country-years have under one broadband line "
                      "per 100, yet internet use there reaches 30 percent: a floor. Mobile's correlation with "
                      "internet use fell from 0.88 in 2005 to 0.42 in 2024, while broadband stays at 0.80. Mobile "
                      "saturated, so it stopped telling countries apart."),
    8: ("Parikshith", "And mobile counts SIMs, not people: the median country has 1.34 SIMs per owner. Fiji jumps "
                      "from 111 to 574 subscriptions in two years while ownership stays at 82 percent. That is a "
                      "recording error, not adoption."),
    11: ("Sindhoor", "Next, unsupervised mining of the 2024 snapshot. k-means gives three archetypes ordered by "
                     "connectivity, separated by landlines and broadband, not mobile. But bootstrap ARI is only "
                     "0.64, and Ward's best-looking silhouette is fake: one of its clusters is just Fiji and Hong "
                     "Kong."),
    12: ("Sindhoor", "Isolation Forest and LOF both rank Fiji as the top anomaly, independently confirming the "
                     "error. [DEMO, Mine tab: hover Fiji, about 8 seconds.]"),
    13: ("Varshith", "Then supervised learning: predict internet share from raw infrastructure. Folds hold out "
                     "whole countries, so the model is always tested on a country it has never seen. Gradient "
                     "boosting reaches 11.8 points error, R-squared 0.86. Broadband dominates, shuffling mobile "
                     "costs almost nothing, and errors are large in mid-adoption countries."),
    14: ("Sindhoor", "This is the feedback loop. Four inferences leave iteration 1, and each one changes the data. "
                     "Fiji is an error: two artefact rules remove 10 cells across all countries. Subscriptions are "
                     "not people: normalise by owners and cap mobile at 100. History matters: add the landline peak "
                     "and reshape to trajectories. Broadband has a floor and the gaps are not random: log transform, "
                     "a floor flag, and imputation inside each training fold. On the right, the transformed data."),
    16: ("Sindhoor", "Iteration 2 clusters each country's whole curve with dynamic time warping. Three archetypes: "
                     "late starters like Tanzania, the S-curve majority, early saturators like South Korea. Two "
                     "thirds of Africa are late starters. The clusters separate far better, silhouette 0.52 against "
                     "0.35, and 37 of 133 countries change group."),
    18: ("Varshith", "Same countries, same folds, same rows: error falls from 11.77 to 10.16 points, minus 14 "
                     "percent, most in Asia and Africa. Imputation adds 2,134 usable rows and error reaches 8.84. "
                     "The ablations show the engineered features carry the gain."),
    21: ("Varshith", "Finally, forecasting. Trained up to 2020 and tested on 2021 to 2025, 71 percent of forecasts "
                     "run low and the 80 percent band covers only 50 percent, because the target drifted after 2020 "
                     "while the inputs did not. Conformal re-calibration widens the band by 4.7 points and coverage "
                     "returns to 83 percent. [DEMO, Forecast & drift tab: India, raw versus conformal, about 8 "
                     "seconds.]"),
    23: ("Parikshith", "Putting it together. When: every region is on the same S-curve, 6 to 12 years behind "
                       "Europe. What: today broadband and the landline peak separate countries, mobile does not. "
                       "Who: the archetypes differ in fixed infrastructure. And the feedback loop improved every "
                       "check: error, cluster separation and forecast coverage."),
    24: ("Varshith", "So mobile leapfrogging happened, but it did not close the divide. We cannot claim causation, "
                     "ownership is mostly imputed, and forecasts assume no further drift. Everything is reproducible "
                     "with one command. Thank you."),
}
APPENDIX = [4, 9, 10, 15, 17, 19, 20, 22]   # V0b, V3b, V4, V9, V8b, V11, V13, V12b

prs = Presentation(FULL)
full_notes = {i: s.notes_slide.notes_text_frame.text for i, s in enumerate(prs.slides, 1)}
titles = {}
for i, s in enumerate(prs.slides, 1):
    titles[i] = next((sh.text_frame.text.splitlines()[0] for sh in s.shapes
                      if sh.has_text_frame and len(sh.text_frame.text) > 20), "")

# reorder: story slides first, appendix last
lst = prs.slides._sldIdLst
ids = list(lst)
order = list(STORY) + APPENDIX
assert sorted(order) == list(range(1, len(ids) + 1)), "deck layout changed; update STORY/APPENDIX"
for el in ids:
    lst.remove(el)
for n in order:
    lst.append(ids[n - 1])

# notes + timing
t, share, lines = 0, {}, []
for pos, n in enumerate(order, 1):
    slide = prs.slides[pos - 1]
    tf = slide.notes_slide.notes_text_frame
    if n in STORY:
        who, text = STORY[n]
        spoken = re.sub(r"\[DEMO[^\]]*\]", "", text)
        dur = math.ceil(len(spoken.split()) / WPS) + (DEMO_S if "[DEMO" in text else 0) + 1
        stamp = f"[{who}, {t // 60}:{t % 60:02d}-{(t + dur) // 60}:{(t + dur) % 60:02d}]"
        tf.text = f"{stamp} {text}"
        share[who] = share.get(who, 0) + dur
        lines += [f"## Slide {pos}: {titles[n]}", "", f"{stamp} {text}", ""]
        t += dur
    else:
        slide._element.set("show", "0")
        body = re.sub(r"^\[[^\]]*\]\s*", "", full_notes[n])
        tf.text = "[APPENDIX, hidden slide, not in the video. Use for questions.] " + body
        lines += [f"### Slide {pos} (hidden appendix): {titles[n]}", "", body, ""]

prs.save(OUT)
summary = {k: f"{v // 60}:{v % 60:02d} ({v / t:.0%})" for k, v in share.items()}
head = ["# A3 video script, 6-minute cut (generated by src/build_slides_6min.py)", "",
        f"Length {t // 60}:{t % 60:02d} at about 150 words per minute.",
        f"Record `slides/{OUT.name}` in slideshow mode: slides {len(STORY) + 1}-{len(order)} are hidden appendix "
        "slides, so the slideshow skips them; keep them for questions.",
        "At each [DEMO ...] cue switch to the app (https://aithal007.github.io/DAS732-A3-Visual-Analytics/, or "
        "`streamlit run app.py` offline) for about 8 seconds, then return to the slides.", "",
        "Coverage of the brief: statistics (slides 4-7), unsupervised mining (8-9, 12), supervised learning and "
        "forecasting (10, 13-14), workflow with the labelled feedback loop (3, 11), second iteration with "
        "visualization (11-14), inferences checked in space (country-held-out folds) and time (2021-25 holdout, "
        "drift).", ""]
(ROOT / "video" / "VIDEO_SCRIPT_6min.md").write_text("\n".join(head + lines), encoding="utf8")
print("saved", OUT.name, f"{len(STORY)} story + {len(APPENDIX)} hidden; length {t // 60}:{t % 60:02d}", summary)

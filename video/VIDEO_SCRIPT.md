# A3 video script (generated from the slide notes by src/build_slides.py)

Length 15:05. Record the deck in slideshow mode. At each [DEMO ...] cue switch to the app (https://aithal007.github.io/DAS732-A3-Visual-Analytics/ or `streamlit run app.py`) for 10-15 seconds, then return. Every report figure appears on a slide.

## Slide 1: From Snapshots to Trajectories

[Parikshith, 0:00-0:20] Good morning. We are Team Plot Twist. In A1 we looked at the Our World in Data internet data and asked whether mobile leapfrogging closed the digital divide. In A3 we test what we saw, with statistics, clustering and prediction, inside a visual analytics workflow with one feedback loop.

## Slide 2: One question, four sub-questions, one feedback loop

[Varshith, 0:20-0:50] Our question extends A1. Four sub-questions, one per family of methods the brief asks for: statistics, unsupervised mining, supervised learning, and the workflow itself. Every figure in the report answers one of them, and every figure is in this video.

## Slide 3: The workflow, unrolled: two full iterations joined by one labelled feedback loop

[Parikshith, 0:50-1:35] On the left, Keim's diamond: data, visualization, models, knowledge. As a cycle it cannot tell us which pass produced which finding, so on the right we unroll it. Iteration 1 runs on the raw A1 data: exploration, models, diagnostics, and four inferences K1. The red edge is the single feedback loop: K1 becomes a data transformation. Iteration 2 repeats the full diamond on the transformed data. The tracker at the bottom of every slide shows where we are.

## Slide 4: System architecture: one shared core, three analysis modules, every number read from artefacts

[Parikshith, 1:35-2:05] How it is built: one shared core module holds the data loading, the artefact rules and the engineered features, so the feedback loop is a single definition. Three analysis modules write tables and figures, and the report, the deck and the browser app only read those artefacts. One command reproduces everything.

## Slide 5: V1  Every region follows the same S-curve, up to 12 years late, and late starters do not catch up

[Parikshith, 2:05-2:50] I fitted a three-parameter logistic S-curve to every country; 201 of 203 fit with R-squared above 0.9. The take-off year, half of saturation, differs by region with p of 4 times 10 to the minus 17. Africa takes off in 2017, almost 12 years after Europe, confirmed by Holm-corrected Mann-Whitney. And on the right: later take-off does not mean faster growth. So the divide is a delay that is not closing on its own.

## Slide 6: V1b  The divide narrowed in relative terms only; low starters close the smallest share of their gap

[Parikshith, 2:50-3:30] Is the divide closing? It depends on how it is measured. Relative inequality, the Gini, fell from 0.81 to 0.19. But the absolute gap between the median European and African country peaked at 64 points in 2014 and is still 52. And on the right: countries that started low closed a SMALLER share of their remaining gap. So there is no catch-up; the divide shrinks only because leaders hit the ceiling.

## Slide 7: V2  Broadband has a floor and a non-linear effect; mobile stopped separating countries after 2005

[Parikshith, 3:30-4:15] Left: almost a third of country-years have no broadband, yet internet use there still reaches 30 percent: that is the floor. Above it internet grows with log broadband, and segmented regression finds a break near 23 per 100. Right: correlation with internet use per year, with bootstrap intervals. In 2005 all three were near 0.9; by 2024 mobile is 0.42 and broadband 0.80. Mobile saturated, so it stopped telling countries apart.

## Slide 8: V3  Subscriptions count SIMs, not people, and Fiji's 2023-24 values are a recording artefact

[Parikshith, 4:15-4:55] The median country has 1.34 SIMs per phone owner, so subscriptions overstate people. Ownership correlates 0.89 with internet use, subscriptions only 0.52. We set one threshold for the whole team, 1.5 SIMs per owner. And Fiji jumps from 111 to 574 subscriptions in two years while ownership stays flat: that is a recording artefact. Sindhoor's anomaly detector finds it independently.

## Slide 9: V3b  Data are missing not at random, so the method changes between iterations

[Parikshith, 4:55-5:35] Before modelling, data quality. Broadband exists only from about 2000, phone ownership only in survey years, and 2025 is almost empty. Crucially, the gaps are not random: country-years without an ownership survey are far less online, 50 against 75 percent. Filling them with an average would bias poor countries upwards, so iteration 1 uses complete cases and iteration 2 uses imputation fitted inside each training fold, plus flags that tell the model a value was missing.

## Slide 10: V4  Two axes describe 82% of 2024 connectivity; the mobile axis is stretched by two countries

[Sindhoor, 5:35-6:10] On the 2024 snapshot of 135 complete countries, PCA puts 64 percent of variance on one connectivity axis, where all five indicators load positively. The second axis is mobile alone, and Fiji and Hong Kong stretch it by themselves.

## Slide 11: V5  Three archetypes ordered by connectivity, separated by landlines and broadband, only moderately stable

[Sindhoor, 6:10-6:50] I scanned k from 2 to 8 for k-means, GMM and Ward. k equal to 3 gives three archetypes ordered by connectivity, mobile-first, mixed and fixed-line legacy, matching A1's four kinds of country. They differ most on landlines and broadband, least on mobile. But bootstrap ARI is only 0.64, and Ward's high silhouette is fake: one of its clusters is just Fiji and Hong Kong.

## Slide 12: V6  Both detectors rank Fiji first; most anomalies are multi-SIM markets

[Sindhoor, 6:50-7:25] Isolation Forest and LOF, rank-averaged. Both put Fiji first. Most of the others are multi-SIM markets; Monaco is there for 95 landlines per 100. This is the model confirming what Parikshith's V3 showed, and it becomes inference K1.1. [DEMO, app tab 1: switch the clustering method; hover Fiji.]

## Slide 13: V7  A raw-infrastructure model explains 86% of variance but leans on broadband alone

[Varshith, 7:25-8:05] We predict internet share from raw broadband, mobile, landlines and year. Folds are grouped by country, so the model is always tested on countries it has never seen. Random forest gets 11.5 points, gradient boosting 11.8. Held-out permutation importance: broadband dominates, mobile adds almost nothing. The errors are large in mid-adoption countries.

## Slide 14: K1 → FEEDBACK LOOP: each inference becomes a change to the data

[Sindhoor, 8:05-8:50] This is the feedback loop. Four inferences leave iteration 1 and each one changes the data. Fiji is an error, so two artefact rules, a level shift and a one-year spike, remove 10 cells across all countries. Subscriptions are not people, so we normalise by owners and cap mobile at 100. History matters, so we add the landline peak as an expanding maximum, past-only, and reshape to trajectories. And broadband is non-linear, so log transform, a floor flag, and imputation inside each training fold.

## Slide 15: V9  The transformation removes ten artefact cells and adds history no single-year value contains

[Parikshith, 8:50-9:25] This is the first view of iteration 2: what the transformation changed. On the left, the removed cells. In the middle, the engineered features against the raw ones: the history features carry information no single-year value has, and SIMs per owner is noise, so we neutralise it. On the right, coverage, which is why we impute. [DEMO, app tab 2: Fiji raw versus cleaned.]

## Slide 16: V8  Grouped by PATH, countries form three S-curve archetypes that cut across the 2024 snapshot

[Sindhoor, 9:25-10:05] In iteration 2 I clustered each country's whole curve, 2000 to 2025, with dynamic time warping. Three archetypes: late starters like Tanzania, the S-curve majority like the Maldives, early saturators like South Korea. Two thirds of Africa are late starters. These clusters separate far better, silhouette 0.52, and 37 of 133 countries change group compared with the 2024 snapshot.

## Slide 17: V8b  The three archetypes are the natural cut of the DTW tree and lie on one timing continuum

[Sindhoor, 10:05-10:40] Is k equal to 3 justified? The dendrogram of DTW distances says yes: late starters split off at the root and early saturators next. The t-SNE map of the same distances lays the countries on one continuum ordered by timing, from Ethiopia to South Korea, with India, Indonesia and Nigeria on the boundary.

## Slide 18: V10  On identical rows and folds, the transformed data lower error by 14%, most in Asia and Africa

[Varshith, 10:40-11:20] Same countries, same folds, same rows. Error falls from 11.77 to 10.16 points, minus 14 percent, most in Asia and Africa, nothing in the Americas. With imputation, 2,134 more rows become usable and error reaches 8.84. The ablations show the engineered features carry most of the gain.

## Slide 19: V11  Ownership, broadband and the landline peak drive the model; its errors are geographic

[Varshith, 11:20-11:55] On the left, SHAP ranks phone ownership, log broadband and the landline peak on top; ownership is mostly imputed, so it acts as a composite index. The map shows where infrastructure under- or over-explains adoption. Kosovo is a data gap, not a finding. [DEMO, app tab 3: toggle iteration 1 and 2.]

## Slide 20: V13  After 2020 the target drifts past the retraining threshold while the inputs do not

[Varshith, 11:55-12:35] This is the drift chart from the lecture on ML workflows: Population Stability Index per year against a 2015 to 2020 baseline, with thresholds 0.1 and 0.2. After 2020 the target crosses 0.2 while every input stays below until 2023. Same infrastructure, more people online. So we predict, before looking, that the forecast will be too low and too confident.

## Slide 21: V12  Forecasts made in 2020 run low; conformal re-calibration restores the cone of uncertainty

[Varshith, 12:35-13:10] Quantile gradient boosting trained up to 2020, tested on 2021 to 2025. As the drift chart predicted, 71% of forecasts are too low and the 80 percent band covers only 50%. Conformalised quantile regression widens it by 4.7 points and coverage returns to 83%. [DEMO, app tab 4: India, raw versus conformal.]

## Slide 22: V12b  The model learned the broadband floor and a history effect; only conformal intervals are calibrated

[Varshith, 13:10-13:50] Two checks on the model. Partial dependence: predictions stay flat while broadband is on the floor and rise between 3 and 30 per 100; the landline history adds 12 points. And the calibration curve tests the intervals at five levels, not one: raw intervals are over-confident everywhere, the conformal ones sit on the diagonal.

## Slide 23: V14  Answer: the divide is a delay on one S-curve, and fixed infrastructure, not mobile, marks the position

[Parikshith, 13:50-14:35] Putting both iterations together. When: every region is on the same S-curve, 6 to 12 years behind Europe. What: today broadband and the landline peak separate countries, mobile does not. Who: the archetypes differ in fixed infrastructure. And the feedback loop improved every check: error on the same rows, cluster separation, and forecast coverage.

## Slide 24: What we know now, and what we cannot claim

[Varshith, 14:35-15:05] To close: mobile leapfrogging happened but did not close the divide. We cannot claim causation, ownership is mostly imputed, and the forecasts assume no further drift. Everything is reproducible with one command. Thank you; we are happy to take questions.

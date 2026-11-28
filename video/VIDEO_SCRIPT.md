# A3 video script (generated from the slide notes by src/build_slides.py)

Length 19:05. Full deck (for the viva). The 6-minute video uses DAS732_A3_DataStory_6min.pptx. Optional: switch to the app (https://aithal007.github.io/DAS732-A3-Visual-Analytics/ or `streamlit run app.py`) for 10-15 seconds, then return. Every report figure appears on a slide.

## Slide 1: From Snapshots to Trajectories

[Parikshith, 0:00-0:20] We are Team Plot Twist. In A1 we asked whether mobile leapfrogging closed the digital divide. In A3 we test what we saw with three visual analytics workflows, one per team member, each with its own hypotheses and its own feedback loop.

## Slide 2: One main question, three workflows, one synthesis

[Varshith, 0:20-0:45] Our main question extends A1. We answer it with three workflows, one per family of methods the brief asks for, and one per team member: statistics, unsupervised mining and supervised learning. Each has its own hypotheses and its own feedback loop, and a synthesis combines them.

## Slide 3: V0  Three unrolled workflows answer one question; each has exactly one labelled feedback loop

[Parikshith, 0:45-1:20] This is the main workflow. On the top right, Keim's diamond; as a cycle it cannot show which pass produced which finding, so we unroll it. Each lane is one workflow: iteration 1 on raw data, its inferences, the red feedback loop that transforms the data, iteration 2 and its knowledge. Bold notes mark knowledge passed between workflows, and all three meet in the synthesis.

## Slide 4: W1  Statistical workflow: hypotheses, plugged-in evidence, a numbered path and one feedback loop

[Parikshith, 1:20-2:00] My workflow, W1, statistics. The question and five hypotheses are written at the top, each with its verdict. Iteration 1 uses four figures, numbered in the order I read them. The red band is the feedback loop: it re-expresses each country's curve in its own time and scale. Iteration 2 tests the hypotheses again on the transformed data.

## Slide 5: V1  Every region follows an S-curve, up to 12 years late, and late starters do not catch up

[Parikshith, 2:00-2:40] Step 1. I fitted a logistic S-curve to every country; 201 of 203 fit with R-squared above 0.9. The take-off year differs by region with p of 4 times 10 to the minus 17: Africa takes off in 2017, almost 12 years after Europe, and later starters do not grow faster.

## Slide 6: V1d  Take-off is a spatial pattern: most of Africa took off 12 years after Europe

[Parikshith, 2:40-3:10] Step 2, where. The same take-off year on the map. North America, Western Europe, Japan and Australia took off first; most of Africa and South Asia about twelve years later. The delay is geographic, not a few outliers.

## Slide 7: V2  Broadband has a floor and a non-linear effect; mobile stopped separating countries after 2005

[Parikshith, 3:10-3:50] Step 3. Almost a third of country-years have no broadband, yet internet there still reaches 30 percent: a floor. Segmented regression finds a break near 23 per 100. On the right, mobile's correlation with internet use fell from 0.88 to 0.42 while broadband stays at 0.80.

## Slide 8: V3  Subscriptions count SIMs, not people, and Fiji's 2023-24 values are a recording artefact

[Parikshith, 3:50-4:25] Step 4. The median country has 1.34 SIMs per phone owner, so subscriptions overstate people. And Fiji jumps from 111 to 574 subscriptions in two years while ownership stays flat: a recording artefact. Sindhoor's workflow finds it independently.

## Slide 9: W1 feedback loop: four inferences become a new coordinate system for the data

[Parikshith, 4:25-5:05] The feedback loop of W1. Fiji becomes a rule applied to every country. Subscriptions are normalised by owners. And the key step: each curve is re-indexed to its own take-off year and divided by its own saturation, and the gap is measured as headroom. V9 underneath shows what changed.

## Slide 10: V1c  Re-indexed to its own take-off, every country follows one S-curve: the divide is a delay

[Parikshith, 5:05-5:45] Iteration 2, step 6. On the left, 201 curves in calendar time. In the middle, the same curves in each country's own time and scale: they collapse onto one S-curve, and the spread between countries falls from 32 to under 10 points. The six regional medians sit on top of each other. The divide is a delay, not a different curve.

## Slide 11: V1e  Closing only in relative terms: the countries furthest behind close the smallest share of their gap

[Parikshith, 5:45-6:20] Step 7. Is the divide closing? On raw levels the two measures disagree. Measured as headroom, the share of the remaining gap each country closed, Africa closed a third, Europe two thirds. It is closing in relative terms only.

## Slide 12: V1f  Leapfrogging did not close the divide: skipped landlines, saturated on mobile, under half online

[Parikshith, 6:20-7:00] Step 8 tests A1's own conclusion with the engineered features. Twenty-two countries, most of them African, skipped landlines and are saturated on mobile, yet only 44 percent are online, against 93 percent where landlines were built. Leapfrogging happened; it did not close the divide.

## Slide 13: W2  Unsupervised workflow: from an unstable snapshot to path archetypes under DTW

[Sindhoor, 7:00-7:40] My workflow, W2, unsupervised mining. Question and hypotheses at the top. Iteration 1 clusters the 2024 snapshot and tests how stable it is. The feedback loop turns each country into its whole curve and changes the distance to dynamic time warping. Iteration 2 clusters paths.

## Slide 14: V4  Two axes describe 82% of 2024 connectivity; the mobile axis is stretched by two countries

[Sindhoor, 7:40-8:10] Step 1. On the 2024 snapshot of 135 countries, PCA puts 64 percent of variance on one connectivity axis. The second axis is mobile alone, and Fiji and Hong Kong stretch it by themselves.

## Slide 15: V5  Three archetypes ordered by connectivity, separated by landlines and broadband

[Sindhoor, 8:10-8:45] Step 2. I scanned k from 2 to 8 for k-means, GMM and Ward. k equal to 3 gives three archetypes ordered by connectivity: mobile-first, mixed and fixed-line legacy. Ward's high silhouette is fake: one cluster is just Fiji and Hong Kong.

## Slide 16: V5b  Snapshot clusters are only moderately stable: mixed and legacy share a cluster in 46% of resamples

[Sindhoor, 8:45-9:20] Step 3, stability. Each cell is how often two countries land in the same cluster over 200 bootstrap refits. The mobile-first block is solid. But mixed and legacy countries share a cluster in 46 percent of refits: the boundary moves.

## Slide 17: V6  Both detectors rank Fiji first; most anomalies are multi-SIM markets

[Sindhoor, 9:20-9:50] Step 4. Isolation Forest and LOF, rank-averaged. Both put Fiji first; most others are multi-SIM markets, and Monaco has 95 landlines per 100.

## Slide 18: V6b  Four anomalies persist every year (structural); Fiji appears only in 2023-24 (artefact)

[Sindhoor, 9:50-10:25] Step 4 over time. I refitted the detector every year. Monaco, Hong Kong, Andorra and Burundi are flagged every year: real markets. Fiji appears only in 2023 and 2024: a recording error. That becomes the shared artefact rule.

## Slide 19: V6c  Each top anomaly is unusual on one indicator

[Sindhoor, 10:25-10:55] Step 5, why. Each bar is how far a country is from its own cluster. Fiji and Hong Kong on mobile, Monaco and Andorra on landlines.

## Slide 20: W2 feedback loop: snapshots become trajectories under a time-shift-invariant distance

[Sindhoor, 10:55-11:35] My feedback loop. Each country becomes its whole curve from 2000 to 2025, artefacts removed. And the distance changes to dynamic time warping: below, Brazil and Bolivia follow the same curve almost seven years apart, and DTW scores them five times closer than a same-year comparison.

## Slide 21: V8  Grouped by PATH, countries form three S-curve archetypes that cut across the 2024 snapshot

[Sindhoor, 11:35-12:15] Step 7. Three path archetypes: late starters like Tanzania, the S-curve majority like the Maldives, early saturators like South Korea. They separate far better than the snapshot clusters, silhouette 0.52 against 0.35.

## Slide 22: V8d  The feedback loop regroups 37 of 133 countries

[Sindhoor, 12:15-12:50] Step 8, what the loop changed. The same map, coloured by iteration 1 and iteration 2. 37 of 133 countries change group: India and Indonesia move to the majority, Sweden and Norway become early saturators.

## Slide 23: W3  Supervised workflow: history features, fair comparison, drift and re-calibrated forecasts

[Varshith, 12:50-13:30] My workflow, W3, supervised learning. Three hypotheses at the top. Iteration 1 is a raw baseline. The feedback loop adds history features, defined from W1's inferences, and imputation inside each training fold. Iteration 2 compares fairly, explains the model, checks drift and forecasts.

## Slide 24: V7  A raw-infrastructure model explains 86% of variance but leans on broadband alone

[Varshith, 13:30-14:10] Step 1. We predict internet share from raw broadband, mobile, landlines and year. Folds hold out whole countries. Gradient boosting gets 11.8 points; broadband dominates, mobile adds almost nothing, and errors are large in mid-adoption countries.

## Slide 25: W3 feedback loop: new predictors and new rows, then a drift-triggered re-calibration

[Varshith, 14:10-14:45] My feedback loop adds thirteen past-only features, defined from W1's inferences, and imputation fitted inside each training fold, which adds 2,134 rows. And an inner loop: when the drift chart crosses 0.2, the forecast intervals are re-calibrated.

## Slide 26: V10  On identical rows and folds, the transformed data lower error by 14%, most in Asia and Africa

[Varshith, 14:45-15:25] Iteration 2, step 2. Same countries, same folds, same rows. Error falls from 11.77 to 10.16 points, minus 14 percent, most in Asia and Africa. With imputation, error reaches 8.84.

## Slide 27: V11  Ownership, broadband and the landline peak drive the model; its errors are geographic

[Varshith, 15:25-16:00] Step 3. SHAP ranks phone ownership, log broadband and the landline peak on top; ownership is mostly imputed, so it acts as a composite index. The map shows where infrastructure under- or over-explains adoption.

## Slide 28: V12b  The model learned W1's broadband floor and a history effect; only conformal intervals are calibrated

[Varshith, 16:00-16:35] Step 4. Partial dependence: predictions stay flat while broadband is on the floor, exactly W1's finding. And the calibration curve: raw intervals are over-confident at every level, the conformal ones sit on the diagonal.

## Slide 29: V13  After 2020 the target drifts past the retraining threshold while the inputs do not

[Varshith, 16:35-17:10] Step 5, the drift chart. Population Stability Index against a 2015 to 2020 baseline. After 2020 the target crosses 0.2 while the inputs stay below: same infrastructure, more people online.

## Slide 30: V12  Forecasts made in 2020 run low; conformal re-calibration restores the cone of uncertainty

[Varshith, 17:10-17:50] Step 6. Trained up to 2020, tested on 2021 to 2025: 71% of forecasts are too low and the 80 percent band covers only 50%. Conformal re-calibration restores 83%.

## Slide 31: V14  Answer: the divide is a delay on one S-curve, and fixed infrastructure, not mobile, marks the position

[Parikshith, 17:50-18:35] Putting the three workflows together. When, from W1: every region is on the same S-curve, 6 to 12 years behind Europe. What: broadband and the landline peak separate countries today, mobile does not. Who, from W2: the archetypes differ in fixed infrastructure. And every feedback loop improved its check: cluster separation, error on the same rows, and forecast coverage.

## Slide 32: What we know now, and what we cannot claim

[Varshith, 18:35-19:05] To close: mobile leapfrogging happened but did not close the divide. We cannot claim causation, ownership is mostly imputed, and the forecasts assume no further drift. Everything is reproducible. Thank you.

# AsPredicted preregistration #311640: the rare-feature face study, text as submitted

The registration text below was transcribed from the submission on 2026-09-15.

## 1) Have any data been collected for this study already?

No

## 2) What's the main question being asked or hypothesis being tested in this study?

We are asking whether a rare feature, such as a face tattoo or a green mohawk, makes a face more memorable. Accounts based on intrinsic memorability treat memorability as a property of the face itself. Distinctiveness accounts (e.g., Simpson et al., 2026) instead argue that recognition depends on telling studied items from new ones. On this view, a rare feature provides strong evidence of a match to memory, but this evidence on its own does not distinguish the studied face from an unstudied face with the same feature. Whether a rare feature improves memory discriminability should therefore depend on what the face is tested against, not on the face itself.

We predict that rare-feature faces will appear highly memorable when scored against unrelated lures, but not memorable at all when the same faces are scored against lures that share their rare feature. If the rare feature instead made a face intrinsically memorable, it should improve memory in both comparisons.

We use REM (Retrieving Effectively from Memory; Shiffrin & Steyvers, 1997) to motivate these predictions. REM has no intrinsic memorability term and computes evidence only relative to the studied list, with each feature match weighted by how rare the feature is. We expect REM to predict performance in this experiment.

## 3) Describe the key dependent variable(s) specifying how they will be measured.

The primary dependent variables are the proportion of "old" responses to each of the four types of test item (studied rare faces, rare-feature foils, studied ordinary faces, and never-studied ordinary faces) and da, estimated from confidence ROCs. On each test trial, participants will make an old/new judgment followed by a four-point confidence rating (Guess, Maybe, Probably, Certain). These two responses give eight response categories, which yield seven-point ROCs. We use da rather than d′ because recognition ROCs are generally asymmetric. Secondary measures are hit rate, false-alarm rate, and criterion c. Accuracy on the 1-back task during study will be used to exclude participants.

We will estimate da by fitting an unequal-variance signal detection (UVSD) model. In this model, the distribution for never-studied ordinary faces is fixed at N(0, 1), and studied ordinary faces, studied rare faces, and rare-feature foils each have their own mean and SD. All four item types share one set of criteria. We will also report the equal-variance model as a baseline. Because the likelihood surface was multimodal in pilot data, we will fit the model from 12 starting points. We will then restart from the best solution so far until the log-likelihood improves by less than 0.01, for up to six rounds. Bootstrap resamples will be fit starting from the full-sample solution.

## 4) How many and which conditions will participants be assigned to?

There are four within-subject conditions, formed by crossing face type (rare vs. ordinary) with study status (studied vs. not studied). Each participant will study 50 faces for 3 s each while performing a 1-back repeat-detection task to maintain attention. They will then complete an arithmetic filler task followed by a 36-trial old/new recognition test.

Six of the studied faces are critical. They come from six pairs, one for each demographic group (e.g., white male). The two faces in a pair are of the same age, sex, and race and have the same rare feature, but are clearly different identities. Each pair has a different rare feature on a different part of the face or head (e.g., a green mohawk or vivid blue lipstick). The studied face in each pair will be chosen at random for each participant. The other 44 studied faces are ordinary faces with none of the six rare features.

At test, three of the six pairs will be tested with the studied face (old) and the other three with the unstudied face from that pair (new), counterbalanced across participants. These trials give the hit rate for studied rare faces and the false-alarm rate for rare-feature foils. The remaining 30 test trials are ordinary faces (15 studied and 15 never studied), which provide the baseline hit rate and the false-alarm rate for unrelated lures.

## 5) Specify exactly which analyses you will conduct to examine the main question/hypothesis.

Our hypotheses are motivated by REM. In REM, memory holds noisy, partial copies of the studied faces. Each test face is compared to every stored copy, and the model computes how much more likely the observed agreement is if the face was studied than if it was not. A match on a rare feature (e.g., a green mohawk) is strong evidence, because two different people rarely share that feature by chance. A match on a common feature (e.g., brown hair) is weak evidence.

We simulated REM in this design across 729 combinations of the environmental parameter g, the number of features w, the storage rate u, the copy accuracy c, the number of dimensions the rare feature occupies, and the placement of the criterion. Across these settings, REM and REM-like models consistently make three predictions:

1. Hit rates will be higher for studied rare faces than for studied ordinary faces. This is because a studied rare face matches its stored copy on the rare feature, which is weighted more heavily than the common features an ordinary face matches on.

2. False-alarm rates will be higher for rare-feature foils than for never-studied ordinary faces. This is because a foil matches the stored copy of the studied face in its pair on the rare feature, and this match is also weighted heavily. REM predicts a large increase in false alarms, which intrinsic memorability accounts do not predict.

3. The difference da(rare vs. unrelated lures) − da(rare vs. own foil) will be positive. Relative to unrelated lures (never-studied ordinary faces), the rare feature increases hits but not false alarms, so memory appears better. Relative to the foil that shares the rare feature, it increases both hits and false alarms, so the advantage effectively cancels.

All three predictions will be tested two-tailed at alpha = .05, and we will report 95% confidence intervals. Predictions 1 and 2 compare proportions within participants and will be tested with a bootstrap over participants (10,000 resamples of the per-participant rates) and a Wilcoxon signed-rank test. Prediction 3 is a contrast between fitted da values and will be tested with a bootstrap over participants (10,000 resamples, refitting the model in each resample). For this test, the p value will be twice the smaller of two proportions: resampled differences at or below zero, and resampled differences at or above zero. We will also report da(rare vs. own foil) − da(studied ordinary vs. never-studied ordinary) as an estimate with a 95% bootstrap interval. This difference estimates what the rare feature is worth when each type of face is scored against its own type of lure.

## 6) Describe exactly how outliers will be defined and handled, and your precise rule(s) for excluding observations.

Participants will be excluded if:

1. They do not complete the study.
2. They have fewer than 4 of 6 hits, or more than 5 false alarms, on the 1-back task during study.

## 7) How many observations will be collected or what will determine sample size?

We will collect data until we have 100 participants who are not excluded under the rules in Question 6.

## 8) Anything else you would like to pre-register?

Nothing else to pre-register.

## 9) Name

Face memory for rare features

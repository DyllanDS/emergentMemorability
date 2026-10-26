# statistics/: the tests and estimates of the paper

This folder holds two tables. `all_tests.csv` lists every statistical test and estimate reported in the
paper and its SI Appendix, with its degrees of freedom, 95% interval and effect size where they apply;
the table at the end of this file lists its 63 test ids. `participant_flow.csv` gives the participants
of each experiment before and after the exclusions. The modules compute these values from the data in
this repository; a result that needs an input the repository does not include (EXTERNAL_INPUTS.md) comes
from the saved results in the module's `data/provided/` folder.

## How to rebuild

After running any module, from the repository root:

    python statistics/collect_all_tests.py

The script needs only the Python standard library and takes about a second (`python run_all.py` runs it
after the modules). It joins the modules' `outputs/tests.csv` files into `all_tests.csv` and their
`outputs/participant_flow.csv` files into `participant_flow.csv`. It copies every value as the module
wrote it, so that p values below the smallest number a computer stores (such as 5.81097e-682) stay
text, and writes each script path from the repository root. It also checks that every test id has at
least one row, and exits with code 1 when a check fails.

## all_tests.csv

There is one row per statistic, sorted by test id (T19b follows T19). A test has several rows when it
covers several contrasts or conditions, or when its module adds a related statistic computed from the
same data (e.g., the same correlation with a second resampling unit).

| Column | Content |
|---|---|
| `test_id` | the test id (table below) |
| `ledger_ids` | the rows of `shared/ledger/paper_numbers.csv` (and of `reported_numbers.csv`) that the test belongs to |
| `location` | where the paper reports the test, a figure caption or a section |
| `test` | what is tested, in words |
| `statistic_name`, `statistic` | the statistic (r or ρ, always named) and its value |
| `n`, `unit` | the number of units and what they are |
| `df` | the degrees of freedom |
| `p`, `p_type` | the p value and its kind (parametric, permutation, bootstrap or none), one-sided or two-sided |
| `effect_size_name`, `effect_size` | the effect size |
| `ci_low`, `ci_high` | the 95% interval |
| `ci_method` | how the interval was computed (percentile bootstrap, Fisher z, t-based, a range over cells, runs or splits, or none); for an interval that needs an input the repository does not include, the input it needs (EXTERNAL_INPUTS.md) |
| `resampling_unit` | what each resample draws (participants, items, concepts, images, participant sessions or simulated items) |
| `n_resamples` | the number of resamples, shuffles or splits |
| `seed` | the seed of the random draws |
| `script` | the script that computes the row, as a path from the repository root |

## participant_flow.csv

There is one row per experiment, and for the image-set experiments one row per image set plus their
total (test T54; from modules 03, 04 and 06). The columns give the sessions run
(`sessions_before_exclusion`), the sessions that each exclusion rule removed (`excluded_*`, where
`excluded_more_than_one_rule` counts the sessions that failed more than one rule), all excluded sessions
(`excluded_total`), the sessions removed by the cap on the sample size (`removed_by_cap`) and the
sessions kept (`kept`). For the face study, `passed_rule` counts the sessions that passed the
preregistered rule, and `passing_sessions_not_needed` counts those that passed after the sample of 100
was complete. The `rules` column states the rules, and `source` names the module table that the row
comes from.

## Conventions

Unless a row says otherwise, an interval is a 95% percentile bootstrap interval from 10,000 resamples
drawn with `numpy.random.default_rng(20261013)`. The `n_resamples`, `seed` and `ci_method` columns give
the count, the seed and the method of every row. Where the paper reports a resampling result from a run
with its own seed and number of resamples, the script repeats that run: Fig. 7d, for example, uses 1,000
participant resamples drawn with MATLAB `rng(42)`, and SI Appendix, Table S1 (variance explained and the
reduction) uses 1,000 participant resamples with seed 7. In SI Appendix, Section 5 the intervals with serial
position and lag held constant come from the 400 participant-session resamples of the saved results (seed
0), and the other intervals from 2,000 participant-session resamples (seed 20261013), which module A5
computes from the trial file (M1) and otherwise reads from its `data/provided/trial_intervals.csv`. Each of
the 14 refitted embeddings of SI Appendix, Fig. S16 uses 2,000 concept resamples.

In our experiments (the inversion experiment, the face study and the three image sets) the bootstrap
resamples participants, except for correlations over items or concepts, the same-image comparisons of SI
Appendix, Fig. S8 and the item means of Fig. S9, which resample the items, concepts or images. In the
published datasets it resamples the THINGS concepts, in the trial-level THINGS data it resamples
participant sessions, and in the simulations it resamples simulated items. Some tests have a second
interval with another unit (e.g., participants for a correlation over items). For a correlation across
features (the 49 features of Figs. 1d, 5c, 5d and 6c to 6h, and the 6 to 81 features of each refit of SI
Appendix, Fig. S16), each resample draws concepts, items or participants and recomputes every feature
correlation. For Figs. 5c, 5d and 6c to 6h and the published embedding of SI Appendix, Fig. S16, the p
value comes from the same resamples: twice the share of resamples on the far side of zero (zero
included). For Fig. 1d and the refits it is the parametric p.

For a paired t test, the rows give t, df, the two-sided p, the t-based 95% interval of the mean
difference and Cohen's d_z (the mean difference divided by the SD of the differences). For a
correlation they give r or ρ, n and the two-sided p. Some correlations have a Fisher z interval in place
of a bootstrap interval, as `ci_method` says, with the standard error of Bonett and Wright (2000) for a
Spearman ρ.

A split-half reliability divides the participants (or, where a dataset has no participant data, each
item's trials) into two random halves, correlates the item scores of the two halves and applies the
Spearman-Brown correction, 2r / (1 + r); the rows give the number of random splits and their seed.

The bootstrap tests of the predicted reductions in Fig. 7d and SI Appendix, Table S1 are one-sided (the p
value is the share of resamples in which the summed squared correlation did not drop). Every other p
value is two-sided, including that of the same reduction over the 559 concepts in both image sets (T19b),
and no p value is corrected for multiple comparisons.

Numbers in the tables keep up to 6 significant digits and are not rounded as in the text.
`reported_numbers.csv` at the repository root compares each number reported in the paper with its computed
value at the printed precision (a rounded value agrees when it lies within half a unit of the last
printed digit).

`shared/python/bootstrap.py` holds the resampling functions (the percentile bootstrap and its interval,
MATLAB's prctile for the intervals of the MATLAB analyses, the permutation test, split-half reliability
with the Spearman-Brown correction, the Fisher z interval, and the face study's registered bootstrap,
which seeds each resample with its number), and `shared/python/tests_table.py` writes each module's
`outputs/tests.csv`.

## The tests

| Test | Where in the paper | What | Module |
|---|---|---|---|
| T01 | Fig. 1b; Methods, Basic REM Simulation of Memorability | REM simulation: memorability against summed similarity over the 1,600 simulated items | 01 |
| T02 | Fig. 1d | REM simulation: correlation across the 49 features (each feature's correlation with memorability against its correlation with summed similarity) | 01 |
| T03 | Fig. 2a | Bylinskii simulation: memorability within against across categories (1,575 target items) | 02 |
| T04 | Fig. 2a | Bylinskii simulation: summed similarity against memorability, within and across categories | 02 |
| T05 | Fig. 2b | Bylinskii simulation: summed similarity within against across categories | 02 |
| T06 | Fig. 3e | inversion experiment: published memorability against memorability in this experiment, 40 critical items | 03 |
| T07 | A Rare Feature's Advantage Depends on What the Face Is Tested Against; SI Appendix, Fig. S11a | face study: the drop in memorability when a new face carries the rare feature | 04 |
| T08 | A Rare Feature's Advantage Depends on What the Face Is Tested Against; SI Appendix, Fig. S11a | face study: rare faces whose feature a new face carries, against ordinary faces | 04 |
| T09 | Fig. 4b; SI Appendix, Section 3, The Model Predicted Each Preregistered Face Result; Fig. S10a | face study: hit and false alarm rates, rare against ordinary faces (preregistered predictions 1 and 2) | 04 |
| T10 | Fig. 5c | THINGS reanalysis: correlation across the 49 features, human similarity space | 05 |
| T11 | Fig. 5d | THINGS reanalysis: correlation across the 49 features, monkey IT space | 05 |
| T12 to T17 | Figs. 6c to 6h | image sets: the six correlations across features | 06 |
| T18 | Figs. 7b and 7c | image sets: spread of summed similarity, reduced against maintained set | 06 |
| T19 | Fig. 7d; Reducing Variability In Distinctiveness Reduces Variability in Memorability | image sets: reduction of the summed squared feature correlations from the maintained to the reduced set (one-sided bootstrap) | 06 |
| T19b | Reducing Variability In Distinctiveness Reduces Variability in Memorability | image sets: the same reduction over the 559 concepts in both sets (two-sided bootstrap; `code/shared_concepts_test.py`) | 06 |
| T20 | Methods, Basic REM Simulation of Memorability; SI Appendix, Section 2, REM Parameters | REM at the standard parameter values of Shiffrin and Steyvers (1997), with the Fig. 1 run beside it | A2a |
| T21 | SI Appendix, Section 1; Methods, Experimental Procedures and Analysis | response criterion c | A1 |
| T22 | SI Appendix, Section 1; Fig. S1; Methods, Experimental Procedures and Analysis | corrected recognition against d_a and d′ | A1 |
| T24 | SI Appendix, Section 2, REM Parameters; Fig. S2a | REM parameter sweep (32 simulations) and the sweep over the number of features | A2a |
| T25 | SI Appendix, Section 2, REM Parameters; Fig. S2b | d′ against ρ over the simulations of the sweep | A2a |
| T26 | SI Appendix, Section 2, Category Structure and Decision Criteria; Fig. S3 | category structure and decision criteria, 18 simulations | A2b |
| T27 | SI Appendix, Section 2, Continuous Recognition; Fig. S4 | continuous recognition: change from the least to the most similar preceding items, and Spearman correlations with the bin number (descriptive, without p), data and model | A2c |
| T28 | SI Appendix, Section 2, List Context; Fig. S5 | list context: change from uncrowded to crowded study lists, random and designed lists | A2d |
| T29 | SI Appendix, Section 2, Measured Representations; Fig. S6 | REM with the measured SPoSE features of THINGS concepts | A2e |
| T30 | SI Appendix, Fig. S7; Section 2, A Hybrid-Similarity Exemplar Model | the hybrid-similarity exemplar model and the GCM | A2f |
| T31 | SI Appendix, Section 3, The Model Predicts Memory From Participants' Own Lists; Fig. S8a | model against observed memorability of the 40 critical items | A3a |
| T32 | SI Appendix, Section 3, The Model Predicts Memory From Participants' Own Lists and Similar Items Change Memory Mainly Through False Alarms; Figs. S8b and S8c | the same image in two image sets: change in d′, hit rate and false alarm rate, observed and model | A3a |
| T33 | SI Appendix, Section 3, Similar Items Change Memory Mainly Through False Alarms; Figs. S9a and S9b | hit rate, false alarm rate and d′ against the number of similar items | A3b |
| T34 | SI Appendix, Section 1; Fig. S9c | THINGS per-image hit and false alarm rates against the memorability score computed from the trial records | A3b |
| T35 | SI Appendix, Table S1 (concept level); Section 3, The Image Set Results Do Not Depend on the Exclusions | concept-level correlations with and without the exclusions | A3d |
| T36 | SI Appendix, Table S1 (feature level) | feature-level correlations with and without the exclusions | A3d |
| T37 | SI Appendix, Table S1 (variance explained and reduction); Section 3, The Image Set Results Do Not Depend on the Exclusions | the Fig. 7d reduction with and without the exclusions (one-sided bootstrap) | A3d |
| T38 | SI Appendix, Section 5, The Same Image After Different Streams; Fig. S13 | THINGS trial data: d′, hit rate and false alarm rate by the similarity of the preceding images, over the whole stream and with serial position and lag held constant | A5 |
| T39 | SI Appendix, Section 5, The Same Image After Different Streams | THINGS trial data: feature crowding against category crowding as predictors of false alarms | A5 |
| T40 | SI Appendix, Section 5, Kramer et al.'s Quartile Test; Fig. S14a | Kramer et al.'s quartile test in six categories | A5 |
| T41 | SI Appendix, Section 5, Kramer et al.'s Quartile Test; Fig. S14b | animal images: d′ by the crowding of the preceding images (all participants and each quartile) and the difference between the quartiles | A5 |
| T42 | SI Appendix, Section 5, A Similarity Space Without Human Judgments | the context effect in the SPoSE space and in deep-network spaces | A5 |
| T43 | SI Appendix, Section 6, Faces and Scenes; Fig. S15a | faces and scenes: held-out correlation of summed similarity with memorability | A6a |
| T44 | SI Appendix, Section 6, Faces and Scenes; Fig. S15b | faces and scenes: summed similarity against hit rate and false alarm rate | A6a |
| T45 | SI Appendix, Section 6, Faces and Scenes; Fig. S15a | faces and scenes: split-half reliability of memorability | A6a |
| T46 | SI Appendix, Section 4, One Measure of Distinctiveness Across Analyses | THINGS concepts: crowding in the deep-network spaces against memorability, hit rate, false alarm rate and d′ | A4b |
| T47 | SI Appendix, Section 6, Embedding Size; Fig. S16 | the published SPoSE embedding and its 14 refits | A6b |
| T48 | SI Appendix, Section 3, The Model Predicted Each Preregistered Face Result; Fig. S10 | REM's predictions for the face study at 729 and 243 parameter settings (counts) | A3c |
| T49 | SI Appendix, Section 3, The Model Predicted Each Preregistered Face Result; Figs. S10a and S10b | face study: the observed contrasts, hits and false alarms (Fig. S10a) and the effect of the lure set on d_a (Fig. S10b) | A3c |
| T50 | Fig. 4c; SI Appendix, Section 3, The Preregistered Discriminability Analysis of the Face Study | face study: d_a from the registered unequal-variance fit | A3c |
| T51 | SI Appendix, Section 3, The Model Predicted Each Preregistered Face Result and The Preregistered Discriminability Analysis of the Face Study; Fig. S11b | face study: the effects of the lure set and of the feature itself on d_a | A3c |
| T52 | Reducing Variability In Distinctiveness Reduces Variability in Memorability | the largest category share of the reduced image set | 06 |
| T53 | Methods, Representational Similarity Analysis | variance of the monkey IT responses kept by the first 49 principal components | 05 |
| T54 | Methods, Experimental Procedures and Analysis | participant flow (also `participant_flow.csv`) | 03, 04, 06 |
| T55 | Reducing Variability In Distinctiveness Reduces Variability in Memorability; Fig. 7 caption; Methods, Experimental Procedures and Analysis | image sets: reliability-corrected variance of concept memorability over the 559 concepts in both sets, maintained against reduced (two-sided permutation test over participants) | 06b |
| T56 | Reducing Variability In Distinctiveness Reduces Variability in Memorability; Methods, Experimental Procedures and Analysis | image sets: across the 49 features, the drop in each feature's squared correlation with memorability from the maintained to the reduced set against its squared correlation with summed similarity in the maintained set | 06b |
| T57 | Reducing Variability In Distinctiveness Reduces Variability in Memorability; Methods, Experimental Procedures and Analysis | the same, controlling for each feature's squared correlation with memorability in the maintained set (partial ρ) | 06b |
| T58 to T61 | SI Appendix, Section 4, One Measure of Distinctiveness Across Analyses | the same concept in two image sets: change in crowding at τ = 0.05 against change in false alarm rate (T58, maintained to reduced; T60, maintained to changed) and in hit rate (T59, T61) | A4a |
| T62, T63 | SI Appendix, Section 4, One Measure of Distinctiveness Across Analyses | THINGS concepts: crowding at τ = 0.05 against hit rate between higher-level categories (T62, descriptive, without p) and within them (T63) | A4a |

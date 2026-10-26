# 06b: Variability of memorability in the reduced image set (Fig. 7 caption and main text)

This module reproduces two results on the 559 concepts tested in both the maintained and the reduced image set,
from *Reducing Variability In Distinctiveness Reduces Variability in Memorability*, the Fig. 7 caption and the
Methods section *Experimental Procedures and Analysis* (Feature Analysis): the reliability-corrected variance of
concept memorability, 38% lower in the reduced set (two-sided permutation p = 0.020), and the feature
dose-response, the correlation across the 49 features between the drop in each feature's predictive power and its
tie to distinctiveness in the maintained set (Spearman ρ = 0.56, and 0.37 as a partial correlation). It computes
both from the trials of module 06, and its tests T55 to T57 go into `statistics/all_tests.csv`. Module 06 computes
the Fig. 7d reduction over the same 559 concepts (34%, test T19b).

## How to run

From the repository root:

    python 06b_reduced_set_variance_fig7/code/run_module.py

The module takes about 10 seconds on the tested laptop (Apple M2 Pro, 16 GB). It needs Python 3 with numpy, scipy
and pandas (tested versions in `environment.md`) and no MATLAB, because the one MATLAB output it reads is included
in the repository.

## Inputs

- `06_image_set_experiments_fig6_fig7/data/maintained_trials.csv` and `reduced_trials.csv` (module 06; columns in
  its README), read through `shared/python/imageset_loader.py`, which applies the exclusions of the Methods and
  keeps 150 participants per set. `participant_concept_counts` in module 06's `code/stats_fig6_fig7.py` counts the
  old trials, hits, new trials and false alarms of each participant for each concept.
- `06_image_set_experiments_fig6_fig7/outputs/summed_similarity_by_concept.csv`: the within-set summed similarity
  of each concept, written by the MATLAB step of module 06 and included in the repository.
- `shared/things/concept_table.csv`: the 49 SPoSE features of each concept.
- `shared/ledger/paper_numbers.csv`: the values printed in the paper, which `outputs/numbers.csv` is judged against.

## What the code does

**Concept memorability.** The corrected recognition (CR) of a concept is its hit rate minus its false alarm rate,
each pooled over the trials of a group of participants (responses 1 to 3 count as "old"), as in Fig. 6. The shared
concepts are the 559 concepts with a CR in both sets.

**Reliability-corrected variance** (`reliability_variance.py`). The 150 participants of a set are split at random
into two halves of 75, and CR is computed from the pooled trials of each half. The covariance of the two halves'
CR across the shared concepts (numpy `cov`, with n - 1 in the denominator, over the concepts that have a CR in
both halves) is averaged over 200 splits; trial-to-trial noise is independent between the halves, so it does not
enter the covariance. The maintained set gives 0.00770 and the reduced set 0.00476, a reduction of
100 x (1 - reduced / maintained) = 38.2%.

**Permutation test** (`reliability_variance.py`). The 300 participants of both sets are assigned at random to two
groups of 150, 2,000 times. In each permutation the reliability-corrected variance of each group is averaged over
10 splits, and the difference between the two groups enters the null distribution. The test statistic is the
observed difference, maintained minus reduced, with each set averaged over 200 splits drawn for the test
(0.00300), and the two-sided p is (1 + the number of null differences at least as large in absolute value) /
2,001 = 0.020.

**Feature dose-response** (`dose_response.py`). Over the shared concepts, each of the 49 features gets a Spearman
correlation with CR in each set and one with summed similarity in the maintained set (tied values take their
average rank). A feature's drop is its squared correlation with CR in the maintained set minus that in the reduced
set, and its tie to distinctiveness is its squared correlation with summed similarity. Across the 49 features, the
Spearman correlation of drop and tie is 0.56 (p < 0.001). For the partial correlation, the ranks of drop and of
tie are each regressed by least squares (with an intercept) on the ranks of the feature's squared correlation
with CR in the maintained set, and the Spearman correlation of the two residuals is 0.37 (p = 0.008). Both p
values are those of scipy's `spearmanr` (t distribution, 47 degrees of freedom).

**Seed and threads.** One generator, `numpy.random.default_rng(20261024)`, draws every split and permutation in
the fixed order listed at the top of `code/reliability_variance.py`, and the seed reproduces the reported p only
in that order. The dose-response draws no random numbers. `run_module.py` limits the numerical libraries to two
processor threads when no limit is set; the results are the same with any number of threads.

## Outputs

- `outputs/numbers.csv`: each number of the paper this module computes, with the printed value, the computed
  value and whether they agree at the printed precision (columns in the top-level README).
- `outputs/tests.csv`: T55 (the permutation test, with the percent reduction as its effect size and the 95% range
  of the null distribution), T56 (the dose-response ρ) and T57 (the partial ρ), each with n and p (columns in
  `statistics/README.md`).
- `outputs/reliability_variance.csv`: the reliability-corrected variance of each set, with the numbers of shared
  concepts and participants.
- `outputs/permutation_test.csv`: the observed difference, the 2.5 and 97.5 percentiles of the null distribution
  and the two-sided p.
- `outputs/feature_dose_response.csv`: for each feature, the Spearman correlation with CR in each set and with
  summed similarity in the maintained set.
- `outputs/dose_response.csv`: the two correlations across the features, with p and n.
- `outputs/logs/run_module.log`: the screen output of the run.

## Code

| File | What it does |
|---|---|
| `code/run_module.py` | runs the two scripts below and writes `outputs/numbers.csv` and `outputs/tests.csv` |
| `code/reliability_variance.py` | the reliability-corrected variance of each set and the permutation test |
| `code/dose_response.py` | the feature dose-response and its partial version |

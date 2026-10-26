"""
stats_from_trials.py: the uncertainty statistics of SI Appendix, Section 5 that need the trial tables
(tests T38, T39, T41 and T42: intervals the saved reports do not hold). Used by stats_section5.py, and
only when the Kramer et al. trial file has been turned into the trial tables in
external/kramer_trials/derived/ (EXTERNAL_INPUTS.md, M1); stats_section5.py saves the results in
data/provided/trial_intervals.csv and reads that file when the trial file is not available.

WHAT IS RESAMPLED, AND WHY
    The unit is the participant session, as in the intervals of sdt_by_context.py: each image's trials
    are binned once, on the whole sample (as the analysis scripts do), and a resample draws sessions
    with replacement and pools their counts in every bin. 2,000 resamples, numpy
    default_rng(20261013), percentile intervals (2.5 and 97.5). The 400 resamples of
    sdt_by_context.py (numpy default_rng(0)) are replayed as well, and the replay is checked against
    the saved per-bin intervals.

    T38  bin 1 minus bin 20 d' over the whole stream (0.508) and its split on the z scale into a hit
         part and a false alarm part
    T39  the ratio of the feature-crowding and category-count coefficients for false alarms (4.6),
         from the per-participant coefficients of crowding.py's model M3
    T41  the animal d' drops (20 bins; five bins in all participants, the top and the bottom quartile)
         and the top-minus-bottom quartile difference in d'
    T42  the held-out deep-network five-bin drop (0.512; resampling held-out sessions) and the SPoSE
         five-bin drop (0.386)

INPUT   external/kramer_trials/derived/trials_with_context.npz, context_k.npz, category_crowding.npz,
        kramer_quartile.npz, fit_on_sequence_heldout_counts.npz (written by the chain in run_module.py)
OUTPUT  nothing by itself: returns the numbers to stats_section5.py
SEED    20261013 for the 2,000 resamples (one generator per statistic, made fresh from the seed); 0 for
        the replay of the 400 resamples of sdt_by_context.py.
"""
import numpy as np
from scipy.stats import norm

import a5_common
import sdt as shared_sdt

RELEASE_SEED = 20261013
RELEASE_RESAMPLES = 2000
PAPER_SEED = 0
PAPER_RESAMPLES = 400
MIN_PRIOR = 20
NUMBER_OF_BINS = 20


def percentile_interval(values):
    """The 2.5 and 97.5 percentiles (numpy's default linear interpolation, as in the analysis scripts)."""
    low, high = np.percentile(np.asarray(values, dtype=float), [2.5, 97.5])
    return float(low), float(high)


def load_trials():
    """The trial table with context, and every session's index 0 .. n - 1 (sessions in sorted order)."""
    data = np.load(a5_common.DERIVED / 'trials_with_context.npz')
    trials = {key: data[key] for key in data.files}
    sessions = np.unique(trials['sub'].astype(np.int64))
    index_of_session = np.zeros(sessions.max() + 1, dtype=np.int64)
    index_of_session[sessions] = np.arange(len(sessions))
    trials['session_index'] = index_of_session[trials['sub'].astype(np.int64)]
    trials['number_of_sessions'] = len(sessions)
    return trials


def resample_weights(generator, number_of_units):
    """How many times each unit is drawn in one resample with replacement (the draw is
    generator.integers(0, n, n), the same call as the bootstraps of sdt_by_context.py)."""
    drawn = generator.integers(0, number_of_units, number_of_units)
    return np.bincount(drawn, minlength=number_of_units).astype(float)


# ----------------------------------------------------------------------------- T38 and T42 (SPoSE)
def whole_stream_counts(trials):
    """Per session and bin, the counts behind Fig. S13's open circles (ctx_top10, target repeats):
    the same masks and bins as sdt_by_context.py."""
    image = trials['image_nr'].astype(np.int64)
    trial_type = trials['ttype']
    enough = trials['n_prior'] >= MIN_PRIOR
    context = trials['ctx_top10'].astype(float)
    fa_mask = np.isin(trial_type, [1, 3]) & enough
    fa_bins = shared_sdt.within_image_bin(np.where(fa_mask, context, np.nan), image)
    false_alarms, first_showings = shared_sdt.cell_counts(trials['session_index'], fa_bins,
                                                           (trials['perf'] == 13).astype(float),
                                                           trials['number_of_sessions'])
    hit_mask = (trial_type == 2) & enough & (trials['lag'] > 0)
    hit_bins = shared_sdt.within_image_bin(np.where(hit_mask, context, np.nan), image)
    hits, repeats = shared_sdt.cell_counts(trials['session_index'], hit_bins, (trials['perf'] == 11).astype(float),
                                           trials['number_of_sessions'])
    return hits, repeats, false_alarms, first_showings


def whole_stream_statistics(hits, repeats, false_alarms, first_showings):
    """From pooled counts per bin: the bin 1 minus bin 20 d' contrast, its hit and false alarm parts on
    the z scale, and the five-bin SPoSE drop of spose_quintiles.py (fifth = mean of four bin rates)."""
    dprime, criterion, d_a, hit_rate, fa_rate = shared_sdt.sdt(hits, repeats, false_alarms, first_showings)
    contrast = dprime[0] - dprime[-1]
    hit_part = norm.ppf(hit_rate[0]) - norm.ppf(hit_rate[-1])
    fa_part = norm.ppf(fa_rate[-1]) - norm.ppf(fa_rate[0])
    fifth_dprime = []
    for fifth in range(5):
        fifth_hit = np.mean(hit_rate[fifth * 4:fifth * 4 + 4])
        fifth_fa = np.mean(fa_rate[fifth * 4:fifth * 4 + 4])
        fifth_dprime.append(norm.ppf(fifth_hit) - norm.ppf(fifth_fa))
    return dict(contrast=contrast, hit_part=hit_part, fa_part=fa_part,
                five_bin_drop=fifth_dprime[0] - fifth_dprime[4], dprime=dprime)


def t38_whole_stream(trials, saved_whole_stream):
    """T38 (0.508) and T42 (SPoSE 0.386): the point values, the 400 resamples of sdt_by_context.py
    replayed (checked against the saved per-bin intervals) and 2,000 resamples (seed 20261013)."""
    hits, repeats, false_alarms, first_showings = whole_stream_counts(trials)
    number_of_sessions = trials['number_of_sessions']
    point = whole_stream_statistics(hits.sum(0), repeats.sum(0), false_alarms.sum(0), first_showings.sum(0))
    if np.max(np.abs(point['dprime'] - saved_whole_stream['dprime'])) > 1e-9:
        raise SystemExit('T38: the whole-stream d\' recomputed from the trial table differs from sdt_ctx_top10_target.npz')

    results = dict(point=point, number_of_sessions=number_of_sessions)
    for label, seed, number_of_resamples in [('paper', PAPER_SEED, PAPER_RESAMPLES),
                                             ('release', RELEASE_SEED, RELEASE_RESAMPLES)]:
        generator = np.random.default_rng(seed)
        resampled = {key: [] for key in ('contrast', 'hit_part', 'fa_part', 'five_bin_drop')}
        dprime_by_bin = np.empty((number_of_resamples, NUMBER_OF_BINS))
        for resample in range(number_of_resamples):
            drawn = generator.integers(0, number_of_sessions, number_of_sessions)
            statistics = whole_stream_statistics(hits[drawn].sum(0), repeats[drawn].sum(0),
                                                 false_alarms[drawn].sum(0), first_showings[drawn].sum(0))
            for key in resampled:
                resampled[key].append(statistics[key])
            dprime_by_bin[resample] = statistics['dprime']
        if label == 'paper':
            # the replay must give the saved per-bin intervals (same draws, same order)
            low, high = np.percentile(dprime_by_bin, [2.5, 97.5], axis=0)
            largest = max(np.max(np.abs(low - saved_whole_stream['dlo'])), np.max(np.abs(high - saved_whole_stream['dhi'])))
            results['paper_replay_largest_difference'] = float(largest)
            if largest > 1e-9:
                raise SystemExit('T38: the replay of the 400 resamples of sdt_by_context.py does not give the saved '
                                 'per-bin intervals')
        results[label] = {key: percentile_interval(values) for key, values in resampled.items()}
        results[label]['resamples'] = number_of_resamples
        results[label]['seed'] = seed
    return results


# ----------------------------------------------------------------------------- T39
def t39_ratio(trials, saved_crowding):
    """T39 (4.6): per-participant coefficients of crowding.py's model M3 for false alarms (feature
    crowding and same-category count together, with serial position), the ratio of their means, and
    its interval over 2,000 resamples of the participants with a coefficient."""
    import crowding as crowding_script
    context_k = np.load(a5_common.DERIVED / 'context_k.npz')
    category_crowding = np.load(a5_common.DERIVED / 'category_crowding.npz')
    raw_session = trials['sub'].astype(np.int64)
    image = trials['image_nr'].astype(np.int64)
    trial_type, response = trials['ttype'], trials['perf']
    trial_number = trials['trial'].astype(float)
    enough = context_k['n_prior'] >= crowding_script.MIN_PRIOR
    feature = context_k['top10'].astype(float)
    category_count = category_crowding['cat_crowd'].astype(float)

    mask = np.isin(trial_type, [1, 3]) & enough
    outcome = np.where(mask, (response == 13).astype(float), np.nan)
    position = np.where(mask, trial_number, np.nan)
    feature_z = np.where(mask, feature, np.nan)
    category_z = np.where(mask, category_count, np.nan)
    feature_z = (feature_z - np.nanmean(feature_z)) / np.nanstd(feature_z)
    category_z = (category_z - np.nanmean(category_z)) / np.nanstd(category_z)
    feature_centred = crowding_script.group_center(feature_z, image)
    category_centred = crowding_script.group_center(category_z, image)
    coefficients = crowding_script.fit(outcome, [feature_centred, category_centred, position], raw_session,
                                       ['feature', 'category', 'position'])
    has_coefficient = np.isfinite(coefficients['feature']) & np.isfinite(coefficients['category'])
    feature_b = coefficients['feature'][has_coefficient]
    category_b = coefficients['category'][has_coefficient]
    point_ratio = feature_b.mean() / category_b.mean()
    printed = saved_crowding['false_alarms']['M3']
    if abs(feature_b.mean() - printed['feature']['b']) > 5e-6 or abs(category_b.mean() - printed['category']['b']) > 5e-6:
        raise SystemExit('T39: the M3 coefficients recomputed from the trial tables differ from crowding.txt')

    generator = np.random.default_rng(RELEASE_SEED)
    number_of_participants = len(feature_b)
    ratios = []
    for resample in range(RELEASE_RESAMPLES):
        drawn = generator.integers(0, number_of_participants, number_of_participants)
        ratios.append(feature_b[drawn].mean() / category_b[drawn].mean())
    return dict(feature_mean=float(feature_b.mean()), category_mean=float(category_b.mean()),
                ratio=float(point_ratio), interval=percentile_interval(ratios),
                participants=number_of_participants, resamples=RELEASE_RESAMPLES, seed=RELEASE_SEED)


# ----------------------------------------------------------------------------- T41
def category_setup(trials):
    """What kramer_quartile.py computes before its per-category loop."""
    import kramer_quartile as quartile_script
    link = a5_common.concept_categories_alphabetical()
    category_of_concept = link['category'].to_numpy().astype(np.int32)
    concept = trials['concept'].astype(np.int32)
    category = category_of_concept[concept - 1]
    trial_type, response, lag = trials['ttype'], trials['perf'], trials['lag']
    setup = dict(link=link, category=category, script=quartile_script,
                 is_fa_trial=np.isin(trial_type, [1, 3]), is_hit_trial=(trial_type == 2) & (lag > 0),
                 fa_y=(response == 13).astype(float), hit_y=(response == 11).astype(float),
                 enough=trials['n_prior'] >= quartile_script.MIN_PRIOR)
    return setup


def binned_counts_per_session(context, image, session, number_of_sessions, hit_mask, hit_y, fa_mask, fa_y, number_of_bins):
    """Per session and within-image bin of context: hits, repeats, false alarms, first showings
    (the binning of kramer_quartile.py's context_dprime, kept per session)."""
    hit_bins = shared_sdt.within_image_bin(np.where(hit_mask, context, np.nan), image, number_of_bins)
    fa_bins = shared_sdt.within_image_bin(np.where(fa_mask, context, np.nan), image, number_of_bins)
    size = number_of_sessions * number_of_bins
    counts = {}
    for name, bins, outcome in [('hits', hit_bins, hit_y), ('false_alarms', fa_bins, fa_y)]:
        binned = bins >= 0
        cell = session[binned] * number_of_bins + bins[binned].astype(np.int64)
        trials_in_cell = np.bincount(cell, minlength=size).reshape(number_of_sessions, number_of_bins).astype(float)
        successes = np.bincount(cell, weights=outcome[binned], minlength=size).reshape(number_of_sessions, number_of_bins)
        if name == 'hits':
            counts['hits'], counts['repeats'] = successes, trials_in_cell
        else:
            counts['false_alarms'], counts['first_showings'] = successes, trials_in_cell
    return counts


def pooled_drop(counts, weights_of_sessions):
    """d' of the least minus the most crowded bin, from counts pooled with the given session weights."""
    dprime, criterion, d_a, hit_rate, fa_rate = shared_sdt.sdt(weights_of_sessions @ counts['hits'],
                                                               weights_of_sessions @ counts['repeats'],
                                                               weights_of_sessions @ counts['false_alarms'],
                                                               weights_of_sessions @ counts['first_showings'])
    return dprime[0] - dprime[-1]


def t41_animal_drops(trials, saved_summary):
    """T41: the animal drops (20 bins; five bins in all participants, the top and the bottom quartile)
    and the top-minus-bottom quartile d', with 2,000 resamples of all participant sessions
    (one draw per resample serves every statistic; quartile membership stays fixed)."""
    setup = category_setup(trials)
    quartile_script = setup['script']
    saved = np.load(a5_common.DERIVED / 'kramer_quartile.npz')
    number_of_sessions = trials['number_of_sessions']
    session = trials['session_index']
    image = trials['image_nr'].astype(np.int64)
    context = trials['ctx_top10'].astype(float)
    name = 'animal'
    category_id = int(setup['link'].loc[setup['link']['category_label'] == name, 'category'].iloc[0])
    in_category = setup['category'] == category_id
    quartile_of_session = saved['%s_quartile_of_participant' % name]
    quartile_of_trial = quartile_of_session[session]
    top = in_category & (quartile_of_trial == 3)
    bottom = in_category & (quartile_of_trial == 0)
    is_hit, is_fa = setup['is_hit_trial'], setup['is_fa_trial']
    hit_y, fa_y, enough = setup['hit_y'], setup['fa_y'], setup['enough']

    analyses = {}
    for label, selection, number_of_bins in [('drop_20_bins', in_category & enough, quartile_script.N_BINS),
                                             ('drop_5_bins_all', in_category & enough, quartile_script.N_BINS_Q),
                                             ('drop_5_bins_top', top & enough, quartile_script.N_BINS_Q),
                                             ('drop_5_bins_bottom', bottom & enough, quartile_script.N_BINS_Q)]:
        analyses[label] = binned_counts_per_session(context, image, session, number_of_sessions, is_hit & selection, hit_y,
                                                    is_fa & selection, fa_y, number_of_bins)

    # pooled d' of a whole quartile (kramer_quartile.py's pooled_sdt), counts per session
    quartile_counts = {}
    for label, selection in [('top', top), ('bottom', bottom)]:
        hits = np.bincount(session[selection & is_hit], weights=hit_y[selection & is_hit], minlength=number_of_sessions)
        repeats = np.bincount(session[selection & is_hit], minlength=number_of_sessions).astype(float)
        false_alarms = np.bincount(session[selection & is_fa], weights=fa_y[selection & is_fa], minlength=number_of_sessions)
        first_showings = np.bincount(session[selection & is_fa], minlength=number_of_sessions).astype(float)
        quartile_counts[label] = (hits, repeats, false_alarms, first_showings)

    def statistics(weights_of_sessions):
        values = {label: pooled_drop(counts, weights_of_sessions) for label, counts in analyses.items()}
        quartile_dprime = {}
        for label, (hits, repeats, false_alarms, first_showings) in quartile_counts.items():
            dprime = shared_sdt.sdt(weights_of_sessions @ hits, weights_of_sessions @ repeats,
                                    weights_of_sessions @ false_alarms, weights_of_sessions @ first_showings)[0]
            quartile_dprime[label] = float(dprime)
        values['quartile_difference'] = quartile_dprime['top'] - quartile_dprime['bottom']
        return values

    point = statistics(np.ones(number_of_sessions))
    row = saved_summary[name]
    expected = dict(drop_20_bins=row['dprime_ctx_drop'], drop_5_bins_all=row['drop_all_5bin'],
                    drop_5_bins_top=row['drop_top_quartile'], drop_5_bins_bottom=row['drop_bottom_quartile'],
                    quartile_difference=row['dprime_q_diff'])
    for key, value in expected.items():
        if abs(point[key] - value) > 1e-9:
            raise SystemExit('T41: %s recomputed from the trial table (%.6f) differs from kramer_quartile_summary.csv (%.6f)'
                             % (key, point[key], value))
    generator = np.random.default_rng(RELEASE_SEED)
    resampled = {key: [] for key in point}
    for resample in range(RELEASE_RESAMPLES):
        weights_of_sessions = resample_weights(generator, number_of_sessions)
        values = statistics(weights_of_sessions)
        for key in resampled:
            resampled[key].append(values[key])
    intervals = {key: percentile_interval(values) for key, values in resampled.items()}
    participants_per_quartile = {quartile: int(np.sum(quartile_of_session == quartile)) for quartile in range(4)}
    return dict(point=point, release=intervals, resamples=RELEASE_RESAMPLES, seed=RELEASE_SEED,
                participants_per_quartile=participants_per_quartile)


# ----------------------------------------------------------------------------- T42 (deep network)
def t42_heldout(saved_fit):
    """T42: the held-out five-bin drop of the fitted deep-network combination (0.512), resampling the
    held-out sessions (2,000 resamples). d' per bin as fit_on_sequence.py computes it: rates
    clipped to [0.0001, 0.9999]."""
    path = a5_common.DERIVED / 'fit_on_sequence_heldout_counts.npz'
    if not path.exists():
        return None
    counts = np.load(path)
    hits, repeats = counts['hits'], counts['repeats']
    false_alarms, first_showings = counts['false_alarms'], counts['first_showings']
    number_of_sessions = len(counts['sessions'])

    def drop(weights_of_sessions):
        hit_rate = np.clip((weights_of_sessions @ hits) / np.maximum(weights_of_sessions @ repeats, 1), 1e-4, 1 - 1e-4)
        fa_rate = np.clip((weights_of_sessions @ false_alarms) / np.maximum(weights_of_sessions @ first_showings, 1),
                          1e-4, 1 - 1e-4)
        dprime = norm.ppf(hit_rate) - norm.ppf(fa_rate)
        return dprime[0] - dprime[-1]

    point = drop(np.ones(number_of_sessions))
    saved_drop = saved_fit['dprime'][0] - saved_fit['dprime'][4]
    if abs(point - saved_drop) > 1e-9:
        raise SystemExit('T42: the held-out drop recomputed from the saved counts differs from fit_on_sequence.json')
    generator = np.random.default_rng(RELEASE_SEED)
    drops = []
    for resample in range(RELEASE_RESAMPLES):
        drops.append(drop(resample_weights(generator, number_of_sessions)))
    return dict(point=float(point), interval=percentile_interval(drops), sessions=number_of_sessions,
                held_out_trials=float(repeats.sum() + first_showings.sum()), resamples=RELEASE_RESAMPLES,
                seed=RELEASE_SEED)

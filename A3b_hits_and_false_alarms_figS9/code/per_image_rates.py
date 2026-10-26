"""
per_image_rates.py: Fig. S9c, the per-image hit and false alarm rates of the original THINGS
memorability data and how the memorability score relates to them (SI Appendix, Section 1).

What it does
  From the trial records of the THINGS memorability study (Kramer et al. 2023): the hit rate and the
  false alarm rate of every photograph with at least 30 old and 30 new trials, their means and SDs,
  and the Pearson correlation, across the photographs, of corrected recognition (hit rate minus
  false alarm rate, the memorability score) with each of the two rates. Old trials are the long-lag
  target repeats (trial type 2); new trials are first showings (trial types 1 and 3); a hit is
  response code 11 on an old trial and a false alarm code 13 on a new trial. Fig. S9c plots the two
  correlations (+0.9873 and -0.0773 over 26,107 photographs, mean false alarm rate 0.0201).

  Corrected recognition in these data is very nearly a hit-rate score. False alarms sit near floor
  because the paradigm puts responding at a conservative criterion, so they contribute almost nothing
  to hits minus false alarms even though they do move with context. Rates are computed from the trial
  records rather than from the published summary table, so the denominators are known.

Modes (the script prints which one it uses)
  regenerate  the trial table external/kramer_trials/derived/trials.npz is here (module A5's extract
              step writes it from the Kramer et al. trial file), or the trial file itself is in
              external/kramer_trials/ (then module A5's extract step, A5_things_context_effects_
              figS13_S14/code/extract_trials.py, runs first; it needs h5py and takes long).
              The report is written to outputs/regenerated/per_image_rates.txt and compared with the
              saved results in data/provided/per_image_rates.txt.
  provided    neither is here (the trial records are not included in the repository; input M1 of
              EXTERNAL_INPUTS.md). The provided-output message is printed and the saved results in
              data/provided/per_image_rates.txt are read and used.
The per-image count table (external/kramer_trials/things_per_image_counts.csv, input M2 of
EXTERNAL_INPUTS.md) does not switch this script to regeneration: its old trials include the short-lag
vigilance repeats, while Fig. S9c counts the long-lag target repeats of the trial records.

Inputs   external/kramer_trials/derived/trials.npz (not included; one row per trial with the columns
         sub, trial, image_nr, ttype, perf, concept), data/provided/per_image_rates.txt
Outputs  outputs/regenerated/per_image_rates.txt (regenerate mode only). Only the aggregate numbers
         of the report are written; nothing per photograph, per trial or per participant.
Seeds    none.   Runtime: seconds (plus module A5's extract step when only the trial file is here).
Run (from the repository root):   python A3b_hits_and_false_alarms_figS9/code/per_image_rates.py
"""
import os
import subprocess
import sys

sys.dont_write_bytecode = True
for thread_setting in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(thread_setting, '2')     # at most two threads per numerical library (same results)

import numpy as np  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a3b_common  # noqa: E402  (also puts shared/python on the path)

MIN_OBS = 30          # a photograph enters with at least 30 old and at least 30 new trials


def run_extract_step():
    """Module A5's extract step: the trial file (M1) -> external/kramer_trials/derived/trials.npz."""
    extract_script = os.path.join(a3b_common.RELEASE_ROOT, a3b_common.EXTRACT_STEP_RELATIVE)
    print('the trial table %s is not here yet: running module A5\'s extract step (%s); this takes long'
          % (a3b_common.TRIAL_TABLE_RELATIVE, a3b_common.EXTRACT_STEP_RELATIVE), flush=True)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg')
    exit_code = subprocess.call([sys.executable, extract_script], cwd=a3b_common.RELEASE_ROOT, env=environment)
    if exit_code != 0 or not os.path.exists(a3b_common.trial_table_path()):
        raise SystemExit('module A5\'s extract step failed (exit code %d); %s was not written'
                         % (exit_code, a3b_common.TRIAL_TABLE_RELATIVE))


def compute_report_lines(trial_table_file):
    """The per-image rates and their correlations, as the lines of the report."""
    trials = np.load(trial_table_file)
    image_of_trial = trials["image_nr"].astype(np.int64)
    trial_type = trials["ttype"]
    response_code = trials["perf"]
    number_of_slots = int(image_of_trial.max()) + 1          # one slot per image number

    # The counts per photograph. np.bincount adds up, for every image number, the values of the trials
    # with that number (vectorized: an explicit loop over 6.2 million trials would be very slow).
    is_old = trial_type == 2                     # target repeats carry the hit opportunity
    is_new = np.isin(trial_type, [1, 3])         # first showings carry the false alarm one
    hits = np.bincount(image_of_trial[is_old], (response_code[is_old] == 11).astype(float), number_of_slots)
    n_old = np.bincount(image_of_trial[is_old], None, number_of_slots).astype(float)
    false_alarms = np.bincount(image_of_trial[is_new], (response_code[is_new] == 13).astype(float), number_of_slots)
    n_new = np.bincount(image_of_trial[is_new], None, number_of_slots).astype(float)

    keep = (n_old >= MIN_OBS) & (n_new >= MIN_OBS)
    hit_rate = hits[keep] / n_old[keep]
    false_alarm_rate = false_alarms[keep] / n_new[keep]
    corrected_recognition = hit_rate - false_alarm_rate

    r_hit, p_hit = pearsonr(corrected_recognition, hit_rate)
    r_false_alarm, p_false_alarm = pearsonr(corrected_recognition, false_alarm_rate)

    # a p below double precision is written as '< 1e-300' instead of 0
    def format_p(value):
        if value == 0:
            return '< 1e-300'
        return f'{value:.3g}'

    lines = [
        "PER-IMAGE RATES IN THE ORIGINAL THINGS DATA",
        "=" * 60,
        f"images with >={MIN_OBS} old and >={MIN_OBS} new observations: {int(keep.sum()):,}",
        f"median opportunities per image: {np.median(n_old[keep]):.0f} old, "
        f"{np.median(n_new[keep]):.0f} new",
        f"hit rate      mean {hit_rate.mean():.4f}  SD {hit_rate.std(ddof=1):.4f}",
        f"false alarm   mean {false_alarm_rate.mean():.4f}  SD {false_alarm_rate.std(ddof=1):.4f}",
        f"corrected rec mean {corrected_recognition.mean():.4f}  SD {corrected_recognition.std(ddof=1):.4f}",
        f"corrected recognition vs hit rate:         r = {r_hit:+.4f}  p = {format_p(p_hit)}",
        f"corrected recognition vs false alarm rate: r = {r_false_alarm:+.4f}  p = {format_p(p_false_alarm)}",
    ]
    # Printed only (not written to the report): the size of the trial table
    print('trial table: %s trials from %s participant sessions; %s images have trials'
          % (format(len(image_of_trial), ','), format(len(np.unique(trials["sub"])), ','),
             format(int(np.sum(n_old + n_new > 0)), ',')))
    return lines


def compare_with_paper_run(regenerated_file):
    """Every number of the regenerated report compared with the saved results
    (data/provided/per_image_rates.txt)."""
    paper_run = a3b_common.read_per_image_rates(a3b_common.PROVIDED_RATES_FILE)
    regenerated = a3b_common.read_per_image_rates(regenerated_file)
    differences = []
    for name in ['images', 'minimum_observations', 'median_old', 'median_new', 'hit_mean', 'hit_sd', 'fa_mean',
                 'fa_sd', 'cr_mean', 'cr_sd', 'r_cr_hit', 'p_cr_hit_text', 'r_cr_fa', 'p_cr_fa_text']:
        if regenerated[name] != paper_run[name]:
            differences.append('%s: regenerated %s, saved %s' % (name, regenerated[name], paper_run[name]))
    if differences:
        print('DIFFERENT from the saved results (data/provided/per_image_rates.txt):')
        for difference in differences:
            print('  ' + difference)
    else:
        print('the regenerated report equals the saved results (data/provided/per_image_rates.txt) in every '
              'printed number')
    return differences


def main():
    print('per_image_rates.py: Fig. S9c, per-image rates of the original THINGS data (module A3b)')
    mode = a3b_common.panel_c_mode()
    if os.path.exists(os.path.join(a3b_common.RELEASE_ROOT, a3b_common.COUNT_TABLE.relative_path)):
        print('note: %s (M2) is present but not used here: its old trials include the short-lag vigilance '
              'repeats, while Fig. S9c counts the long-lag target repeats of the trial records only (see README)'
              % a3b_common.COUNT_TABLE.relative_path)

    if mode == 'provided':
        values = a3b_common.read_per_image_rates(a3b_common.PROVIDED_RATES_FILE)
        print('mode: provided (saved results): %s' % a3b_common.relative(a3b_common.PROVIDED_RATES_FILE))
        print('  %d images with >= %d old and >= %d new trials (median %d old, %d new); hit rate mean %.4f SD %.4f; '
              'false alarm rate mean %.4f SD %.4f; r(score, hit rate) = %+.4f (p %s); r(score, false alarm rate) '
              '= %+.4f (p %s)' % (values['images'], values['minimum_observations'], values['minimum_observations'],
                                  values['median_old'], values['median_new'], values['hit_mean'], values['hit_sd'],
                                  values['fa_mean'], values['fa_sd'], values['r_cr_hit'], values['p_cr_hit_text'],
                                  values['r_cr_fa'], values['p_cr_fa_text']))
        return

    if not os.path.exists(a3b_common.trial_table_path()):
        run_extract_step()
    print('mode: regenerate from %s' % a3b_common.TRIAL_TABLE_RELATIVE)
    lines = compute_report_lines(a3b_common.trial_table_path())
    text = "\n".join(lines)
    print(text)
    os.makedirs(a3b_common.REGENERATED_FOLDER, exist_ok=True)
    with open(a3b_common.REGENERATED_RATES_FILE, 'w', encoding='utf-8') as handle:
        handle.write(text + "\n")
    print('wrote %s' % a3b_common.relative(a3b_common.REGENERATED_RATES_FILE))
    compare_with_paper_run(a3b_common.REGENERATED_RATES_FILE)


if __name__ == "__main__":
    main()

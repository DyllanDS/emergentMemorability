"""
continuous_rem.py: the REM model run through continuous recognition on the trial sequences of the
THINGS memorability study (SI Appendix, Section 2, Continuous Recognition; the model lines of Fig. S4).

WHAT IT DOES
  Each of 4,000 simulated participants follows the trial sequence of one of 400 participants of the
  THINGS memorability study (Kramer et al., 2023), ten simulated participants per sequence: the same
  295 trials, with first showings and repeats in the same order and at the same lags. Only the items
  are simulated: every distinct image of a sequence is replaced by an item with REM's generated
  features (geometric feature values), so the items have no memorability of their own.
  The model runs the task trial by trial:
    1. the item on screen is compared with every trace stored on earlier trials (REM's likelihood
       ratio, Shiffrin and Steyvers, 1997); the model says "old" when the average likelihood ratio
       over those traces is above its criterion;
    2. after the response the item is stored as a new trace, whether it was a first showing or a
       repeat; nothing is ever forgotten, so a repeat is judged against a memory that holds its own
       earlier trace and every item stored since;
    3. the criterion is not REM's usual 1.0: it is searched so that the model's false alarm rate is
       the participants' 2 percent.
  The model runs at the standard storage probability (0.30) and at the storage probability (of 0.30,
  0.45, 0.60, 0.75, 0.90) whose overall d' is closest to TARGET_DPRIME (0.75 in the saved run in
  data/provided/).
  Each item's trials are then split into 20 bins by the similarity of the 10 most similar items shown
  before them (similarity = the proportion of features with the same value), and the hit rate, the
  false alarm rate, d' and the criterion c are computed per bin with the same binning and SDT
  functions as the data analysis (shared/python/sdt.py). The report also lists, for each storage
  probability, the mean per-item hit and false alarm rates (items with at least 30 trials of each
  kind) and the correlation of corrected recognition with each.

INPUTS
  The trial structures of 400 sessions: for every trial, its slot (which distinct image of the session
  it shows) and its trial type (1 = first showing of a target, 2 = repeat of a target, 3 = filler,
  4 = vigilance repeat). They are derived from the Kramer et al. trial file, which this release does
  not contain (EXTERNAL_INPUTS.md, M1 and M6). They are read from the first of these that exists:
    1. the file given with --templates (a template file written by make_trial_templates.py)
    2. data/trial_templates_400.npz (the template file, M6, when present)
    3. the file given with --trial-table, or the trial table that module A5's extract_trials.py
       writes from the trial file (external/kramer_trials/derived/trials.npz, next to the trial
       file; never part of the release)
  Without any of them the script stops with the provided-output message; run_module.py then uses the
  saved run in data/provided/.

OUTPUTS (this module's outputs/ folder, or the folder given with --outputs)
  continuous_rem.txt          the report, one block per storage probability
  continuous_bins.npz         the 20-bin curves and summary numbers of each storage probability, keys
                              '<point>_<name>' with point 'standard' or 'accuracy-matched' (the same
                              keys as data/provided/continuous_bins.npz)
  continuous_cell_counts.npz  per simulated participant and bin: hits and target-repeat trials, false
                              alarms and first-showing trials (input of the bootstrap in stats_figS4.py)

SEEDS    numpy.random.default_rng(7): the item features, then both main runs (one stream, in this
         order); default_rng(99): every step of the criterion search (a fresh generator per step);
         default_rng(1234): the d' check at each storage probability. One process, run serially.
RUNTIME  about 4 minutes on an Apple M2 Pro (14,400 simulated sessions of 295 trials).

RUN (from the repository root)
  python A2c_continuous_recognition_figS4/code/continuous_rem.py
  python A2c_continuous_recognition_figS4/code/continuous_rem.py --templates <file> --outputs <folder>
"""
import argparse
import os
import sys
import time

import numpy as np
from scipy.stats import norm, spearmanr
from scipy.special import logsumexp

sys.dont_write_bytecode = True   # write no __pycache__ folder into shared/python or code/ (set before the imports below)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402
import sdt as S     # noqa: E402  binning and SDT functions shared with the THINGS data analysis (module A5)

PATHS = release_paths.module_paths(__file__, make_output_folders=False)
RELEASE_ROOT = release_paths.release_root()

# Where the trial structures can come from (see INPUTS above), relative to the repository root.
TEMPLATE_FILE = os.path.join(PATHS['data'], 'trial_templates_400.npz')
# the trial table module A5's extract_trials.py writes from the Kramer et al. trial file (a5_common.TRIAL_TABLE)
A5_TRIAL_TABLE = os.path.join(RELEASE_ROOT, 'external', 'kramer_trials', 'derived', 'trials.npz')
KRAMER_TRIAL_FILE = os.path.join(RELEASE_ROOT, 'external', 'kramer_trials', 'imbysub-all-11-16-21.mat')

# ---- model parameters: the standard Shiffrin & Steyvers values ----
G, W, U, C = 0.40, 20, 0.30, 0.70   # geometric parameter g, number of features w, storage u, copy fidelity c
N_DATABASE = 1600                    # generated items a session's distinct images are drawn from
N_SUBJECTS = 4000                    # simulated participants per storage probability
N_TEMPLATES = 400                    # trial structures: the first 400 sessions with 295 trials
TEMPLATE_LENGTH = 295                # trials per session (sessions with fewer trials are skipped)
TOPK = 10                            # context = mean similarity of the 10 most similar preceding items
N_BINS = 20                          # bins within item
MIN_PRIOR = 50                       # a trial is binned only when at least 50 earlier trials showed other items
TARGET_FALSE_ALARM_RATE = 0.020      # the participants' false alarm rate (2 percent)
TARGET_DPRIME = 2.93                 # target overall d' for choosing the second storage probability
STORAGE_GRID = [0.30, 0.45, 0.60, 0.75, 0.90]
USUAL_REM_CRITERION = 1.0            # REM's usual criterion in the study-test simulations; not used in this
                                     # model (check_provided.py compares the searched criteria with it)
RNG = np.random.default_rng(7)


# ----------------------------------------------------------------------------- the trial structures
def load_templates(trial_table_file, n_needed=N_TEMPLATES):
    """The participants' trial structures from a trial table: (slot of each trial, trial type of each
    trial).

    The first n_needed sessions (in the order of the trial table) with at least 295 trials; for each,
    the slot of every trial is the position of its image number among the session's distinct image
    numbers (np.unique), so a repeat has the slot of its first showing."""
    d = np.load(trial_table_file)
    sub, image, ttype = d["sub"], d["image_nr"], d["ttype"]
    bounds = np.searchsorted(sub, np.arange(sub[-1] + 2))
    out = []
    for s in range(len(bounds) - 1):
        a, b = bounds[s], bounds[s + 1]
        if b - a < TEMPLATE_LENGTH:
            continue
        img = image[a:b]
        _, slot = np.unique(img, return_inverse=True)
        out.append((slot.astype(np.int32), ttype[a:b].astype(np.int8)))
        if len(out) >= n_needed:
            break
    return out


def load_templates_from_template_file(template_file, n_needed=N_TEMPLATES):
    """The same trial structures from a template file of make_trial_templates.py: one row per
    session of 'slot' and 'trial_type' (structure only: no responses, image numbers or ids)."""
    d = np.load(template_file)
    slot_table = d['slot']
    trial_type_table = d['trial_type']
    if slot_table.shape[0] < n_needed:
        sys.exit('continuous_rem.py: %s holds %d trial structures; %d are needed'
                 % (template_file, slot_table.shape[0], n_needed))
    out = []
    for row in range(n_needed):
        out.append((slot_table[row].astype(np.int32), trial_type_table[row].astype(np.int8)))
    return out


def relative_to_release(path):
    """A path as it is printed: relative to the repository root when it lies inside it."""
    absolute = os.path.abspath(path)
    if absolute.startswith(RELEASE_ROOT + os.sep):
        return os.path.relpath(absolute, RELEASE_ROOT)
    return absolute


def find_templates(templates_file=None, trial_table_file=None):
    """The trial structures from the first source that exists (see INPUTS in the header).
    Returns (templates, description of the source), or (None, the reason) when there is none."""
    if templates_file is not None:
        if not os.path.exists(templates_file):
            sys.exit('continuous_rem.py: the template file %s does not exist' % templates_file)
        return load_templates_from_template_file(templates_file), 'template file ' + relative_to_release(templates_file)
    if os.path.exists(TEMPLATE_FILE):
        return load_templates_from_template_file(TEMPLATE_FILE), 'template file ' + relative_to_release(TEMPLATE_FILE)
    if trial_table_file is not None:
        if not os.path.exists(trial_table_file):
            sys.exit('continuous_rem.py: the trial table %s does not exist' % trial_table_file)
        return load_templates(trial_table_file), 'trial table ' + relative_to_release(trial_table_file)
    if os.path.exists(A5_TRIAL_TABLE):
        return load_templates(A5_TRIAL_TABLE), 'trial table ' + relative_to_release(A5_TRIAL_TABLE)
    if os.path.exists(KRAMER_TRIAL_FILE):
        return None, ('the Kramer et al. trial file is in external/kramer_trials/, but the trial table module A5 '
                      'builds from it is not (%s); run python A5_things_context_effects_figS13_S14/code/'
                      'run_module.py first' % relative_to_release(A5_TRIAL_TABLE))
    return None, ('the trial structures of 400 sessions (%s, EXTERNAL_INPUTS.md M6, or the Kramer et al. trial '
                  'file, M1, in external/kramer_trials/)' % relative_to_release(TEMPLATE_FILE))


# ----------------------------------------------------------------------------- the model
def build_similarity_matrix(db):
    """Similarity between simulated items: the proportion of features sharing a value, which is what
    the likelihood ratio in this model responds to (an item's similarity to itself is 1)."""
    n_items = db.shape[0]
    sim = np.zeros((n_items, n_items), dtype=np.float32)
    for i in range(n_items):
        sim[i] = (db == db[i]).mean(1)
    np.fill_diagonal(sim, 1.0)
    return sim


def encode(items, g, u, c, rng):
    """Store each presented item as a noisy, incomplete trace.
    Each feature is copied wrongly with probability 1 - c (replaced by a new geometric value) and
    stored at all with probability u (0 = nothing stored)."""
    tr = items.copy()
    wrong = rng.random(items.shape) < (1 - c)
    n = int(wrong.sum())
    if n:
        # NumPy's geometric counts trials to the first success, so it already has support
        # {1, 2, ...}, the same support the likelihood's base rate g(1-g)^(v-1) assumes. Do not
        # add 1. (MATLAB's geornd counts failures and starts at 0, so the MATLAB REM code of this
        # repository writes geornd + 1.)
        tr[wrong] = rng.geometric(g, n)
    tr[rng.random(items.shape) >= u] = 0
    return tr


def log_lambda(probe, traces, logfac, stored):
    """Log likelihood ratio of the probe against each stored trace.

    Vectorized form of Shiffrin & Steyvers (1997) Eq. 4A: a match on a stored feature contributes its
    rarity-weighted factor, a mismatch contributes log(1 - c), and unstored features contribute
    nothing."""
    match = stored & (traces == probe)
    n_mismatch = stored.sum(1) - match.sum(1)
    return (match * logfac).sum(1) + n_mismatch * np.log(1 - C)


def run_subject(db, template, crit, rng, sim_matrix, u=U):
    """One continuous-recognition stream. Returns per-trial records:
    the item of each trial, its trial type, the response (True = "old"), the context similarity (mean
    similarity of the TOPK most similar preceding items other than the item itself; NaN before
    TOPK such items) and the number of preceding trials that showed other items."""
    # The variables, in the model's terms. check_model_mechanics.py writes the same session out trial
    # by trial with explicit loops and checks that both give the same responses with the same random
    # numbers.
    #   seq       the database item shown on each trial (the template's slots, filled with items)
    #   traces     one memory trace per earlier trial; stored marks the features that were stored
    #   logfac     for each stored feature, the log of its match factor (c + (1 - c) p(v)) / p(v), with
    #              p(v) = g (1 - g)^(v - 1) the base rate of the stored value v (rarer values count more)
    #   ll         the log likelihood ratio of the probe against each earlier trace (log_lambda)
    #   log_fam    familiarity: the log of the mean likelihood ratio over the earlier traces; the
    #              response is "old" when it exceeds the log of the criterion
    #   ctx        the context similarity: the mean similarity of the TOPK most similar earlier items
    #   n_prior    the number of earlier trials that showed other items
    slot, ttype = template
    n_slot = slot.max() + 1
    items = rng.choice(len(db), size=n_slot, replace=False)   # slot -> db index
    seq = items[slot]
    T = len(seq)

    traces = np.zeros((T, W), dtype=np.int32)
    logfac = np.zeros((T, W))
    stored = np.zeros((T, W), dtype=bool)
    n_tr = 0

    resp = np.zeros(T, dtype=bool)
    ctx = np.full(T, np.nan)
    n_prior = np.zeros(T, dtype=np.int32)

    for t in range(T):
        probe_item = seq[t]
        probe = db[probe_item]

        # 1. decide: compare the probe with every trace stored on earlier trials
        if n_tr:
            ll = log_lambda(probe, traces[:n_tr], logfac[:n_tr], stored[:n_tr])
            # compare in log space; a single rare-feature match can otherwise overflow the
            # exponential. log of the mean likelihood ratio over the n_tr traces:
            log_fam = logsumexp(ll) - np.log(n_tr)
            resp[t] = log_fam > np.log(crit)

        # distinctiveness relative to what preceded, excluding this item's own earlier
        # presentations (the same definition as the data analysis of module A5)
        prior = seq[:t]
        if t:
            keep = prior != probe_item
            n_prior[t] = keep.sum()
            if n_prior[t] >= TOPK:
                sims = sim_matrix[probe_item, prior[keep]]
                ctx[t] = np.sort(sims)[-TOPK:].mean()

        # 2. store: every presentation, first showing or repeat, lays down a new trace (never forgotten)
        traces[n_tr] = encode(probe[None, :], G, u, C, rng)[0]
        stored[n_tr] = traces[n_tr] > 0
        v = traces[n_tr]
        pbase = G * (1 - G) ** np.maximum(v - 1, 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            logfac[n_tr] = np.where(stored[n_tr],
                                    np.log((C + (1 - C) * pbase) / pbase), 0.0)
        n_tr += 1

    return seq, ttype, resp, ctx, n_prior


def simulate(db, templates, crit, sim_matrix, n_sub, rng, label="", u=U):
    """n_sub simulated participants; participant i follows template i modulo the number of templates."""
    recs = {k: [] for k in ("sub", "item", "ttype", "resp", "ctx", "n_prior")}
    t0 = time.time()
    for i in range(n_sub):
        seq, tt, resp, ctx, npr = run_subject(
            db, templates[i % len(templates)], crit, rng, sim_matrix, u)
        recs["sub"].append(np.full(len(seq), i, dtype=np.int32))
        recs["item"].append(seq.astype(np.int32))
        recs["ttype"].append(tt)
        recs["resp"].append(resp)
        recs["ctx"].append(ctx)
        recs["n_prior"].append(npr)
        if i and i % 500 == 0:
            print(f"    {label}{i}/{n_sub} ({time.time()-t0:.0f}s)", flush=True)
    return {k: np.concatenate(v) for k, v in recs.items()}


def rates(r):
    """Aggregate hit and false alarm rates. Types 2 and 4 are repeats; types 1 and 3 are first showings."""
    rep = np.isin(r["ttype"], [2, 4])
    new = np.isin(r["ttype"], [1, 3])
    return r["resp"][rep].mean(), r["resp"][new].mean()


def calibrate_crit(db, templates, sim_matrix, u, target_fa=TARGET_FALSE_ALARM_RATE, quiet=False):
    """Criterion that gives the target false alarm rate at this storage probability.
    Nine halvings on a logarithmic scale between 0.2 and 2,000, each with 100 simulated participants
    (the first 100 trial structures) and a fresh generator default_rng(99); the criterion is the
    geometric mean of the final bounds."""
    lo, hi = 0.2, 2000.0
    for _ in range(9):
        mid = np.sqrt(lo * hi)
        r = simulate(db, templates, mid, sim_matrix, 100,
                     np.random.default_rng(99), u=u)
        hit, fa = rates(r)
        if not quiet:
            print(f"    u={u:.2f} criterion {mid:9.3f} -> FA {fa:.4f} "
                  f"hit {hit:.4f}", flush=True)
        if fa > target_fa:
            lo = mid
        else:
            hi = mid
    return np.sqrt(lo * hi)


def dprime_of(db, templates, sim_matrix, u, crit):
    """Overall d' of 200 simulated participants (generator default_rng(1234)); rates kept within
    0.0001 and 0.9999."""
    r = simulate(db, templates, crit, sim_matrix, 200,
                 np.random.default_rng(1234), u=u)
    hit, fa = rates(r)
    za = lambda q: norm.ppf(min(max(q, 1e-4), 1 - 1e-4))  # noqa: E731
    return za(hit) - za(fa), hit, fa


def calibrate_storage(db, templates, sim_matrix, target_d=TARGET_DPRIME):
    """The storage probability of STORAGE_GRID whose overall d' is closest to target_d, with its
    criterion and d' (a tie keeps the earlier value of the grid).

    At the standard storage probability the model's overall d' over a 295-trial stream is below the
    participants'. Storage is the parameter varied because it scales with study time in Shiffrin and
    Steyvers (1997).
    """
    print(f"  calibrating storage to d' = {target_d}", flush=True)
    best = None
    for u in STORAGE_GRID:
        crit = calibrate_crit(db, templates, sim_matrix, u, quiet=True)
        d, hit, fa = dprime_of(db, templates, sim_matrix, u, crit)
        print(f"    u={u:.2f} crit={crit:9.3f} -> d'={d:.2f} "
              f"hit={hit:.3f} fa={fa:.4f}", flush=True)
        if best is None or abs(d - target_d) < abs(best[2] - target_d):
            best = (u, crit, d)
    return best


def analyse(r, db_n, templates, label, u, crit, w):
    """Report one operating point: the overall rates, the per-item rates and the context effect binned
    within item. Returns the binned curves and summary numbers, plus the per-participant cell counts."""
    hit, fa = rates(r)
    w("-" * 96)
    w(f"OPERATING POINT: {label}")
    w(f"  storage u = {u:.2f}, criterion = {crit:.3f}")
    w(f"  aggregate hit rate {hit:.4f}, false alarm rate {fa:.4f}   "
      f"(empirical: 0.808 and 0.020)")
    w("-" * 96)

    # ---- 1. per-item hit and false alarm rates (items with at least 30 trials of each kind) and the
    #         correlation of corrected recognition (hit rate minus false alarm rate) with each ----
    rep = np.isin(r["ttype"], [2, 4])
    new = np.isin(r["ttype"], [1, 3])
    hk = np.bincount(r["item"][rep], weights=r["resp"][rep], minlength=db_n)
    hn = np.bincount(r["item"][rep], minlength=db_n)
    fk = np.bincount(r["item"][new], weights=r["resp"][new], minlength=db_n)
    fn = np.bincount(r["item"][new], minlength=db_n)
    ok = (hn >= 30) & (fn >= 30)
    H, F = hk[ok] / hn[ok], fk[ok] / fn[ok]
    cr = H - F
    cr_H = np.corrcoef(cr, H)[0, 1]
    cr_F = np.corrcoef(cr, F)[0, 1]
    w(f"  items with >=30 observations of each kind: {ok.sum():,}")
    w(f"  per-item hit rate  mean {H.mean():.4f} SD {H.std(ddof=1):.4f}   "
      f"FA rate mean {F.mean():.4f} SD {F.std(ddof=1):.4f}")
    w(f"  corrected recognition vs hit rate  r = {cr_H:+.4f}")
    w(f"  corrected recognition vs FA rate   r = {cr_F:+.4f}")
    w("")

    # ---- 2. the context effect, binned within item as in the data ----
    # False alarms: first showings (types 1 and 3); hits: repeats of targets (type 2). A trial is
    # binned only when at least MIN_PRIOR earlier trials showed other items.
    S.N_BINS = N_BINS
    enough = r["n_prior"] >= MIN_PRIOR
    item = r["item"].astype(np.int64)
    sub = r["sub"].astype(np.int64)
    n_sub = int(sub.max()) + 1
    fb = S.within_image_bin(np.where(new & enough, r["ctx"], np.nan), item)
    hb = S.within_image_bin(np.where((r["ttype"] == 2) & enough, r["ctx"], np.nan),
                            item)
    fkk, fnn = S.cell_counts(sub, fb, r["resp"].astype(float), n_sub)
    hkk, hnn = S.cell_counts(sub, hb, r["resp"].astype(float), n_sub)
    dp, cc, da, Hb, Fb = S.sdt(hkk.sum(0), hnn.sum(0), fkk.sum(0), fnn.sum(0))
    rho_d = spearmanr(np.arange(N_BINS), dp).statistic
    rho_c = spearmanr(np.arange(N_BINS), cc).statistic

    w("  context binned within item, least to most similar preceding stream:")
    w(f"    d'        {dp[0]:.3f} -> {dp[-1]:.3f}  (change {dp[-1]-dp[0]:+.3f}), "
      f"rho = {rho_d:+.3f}")
    w(f"    hit rate  {Hb[0]:.4f} -> {Hb[-1]:.4f}")
    w(f"    FA rate   {Fb[0]:.4f} -> {Fb[-1]:.4f}")
    w(f"    criterion {cc[0]:+.3f} -> {cc[-1]:+.3f}, rho = {rho_c:+.3f}")
    w("")
    result = dict(u=u, crit=crit, hit=hit, fa=fa, cr_H=cr_H, cr_F=cr_F,
                  dprime=dp, crit_bin=cc, H=Hb, F=Fb, rho_d=rho_d, rho_c=rho_c)
    cell_counts = dict(hit_successes=hkk, hit_trials=hnn, false_alarm_successes=fkk, false_alarm_trials=fnn)
    return result, cell_counts


def as_whole_numbers(counts):
    """Cell counts (floats from np.bincount with weights) as int32, after checking they are whole."""
    rounded = np.rint(counts)
    if np.max(np.abs(counts - rounded)) > 1e-9:
        raise ValueError('cell counts are not whole numbers')
    return rounded.astype(np.int32)


def main():
    parser = argparse.ArgumentParser(description='REM in continuous recognition on the THINGS trial sequences '
                                                 '(model side of Fig. S4).')
    parser.add_argument('--templates', help='template file of make_trial_templates.py (slot, trial_type)')
    parser.add_argument('--trial-table', help='trial table of the Kramer et al. data (sub, image_nr, ttype), '
                                              'as module A5 writes it')
    parser.add_argument('--outputs', help='folder for the outputs (default: this module\'s outputs/)')
    arguments = parser.parse_args()

    templates, source = find_templates(arguments.templates, arguments.trial_table)
    if templates is None:
        print(release_paths.PROVIDED_MESSAGE % source)
        sys.exit('continuous_rem.py: no trial structures, nothing simulated; the saved run is in '
                 'data/provided/ (python A2c_continuous_recognition_figS4/code/run_module.py uses it)')
    out_folder = arguments.outputs if arguments.outputs else PATHS['outputs']
    os.makedirs(out_folder, exist_ok=True)
    print('trial structures from the %s' % source, flush=True)

    # Feature values on NumPy's geometric support {1, 2, ...}, matching the g(1-g)^(v-1) base rate
    # the likelihood ratio uses. No +1: unlike MATLAB's geornd (support {0, 1, ...}), NumPy's
    # already starts at 1, and shifting it would make a value's true frequency disagree with its
    # assumed rarity.
    db = RNG.geometric(G, size=(N_DATABASE, W))

    # similarity between simulated items: the proportion of features sharing a value, which is what
    # the likelihood ratio in this model responds to
    print("building item similarity matrix...", flush=True)
    sim = build_similarity_matrix(db)

    print(f"{len(templates)} trial structures loaded "
          f"({len(templates[0][0])} trials each)", flush=True)

    # Two operating points: the standard storage probability, at which the model is less accurate than
    # the participants, and the storage probability of STORAGE_GRID whose overall d' is closest to
    # TARGET_DPRIME.
    crit_std = calibrate_crit(db, templates, sim, U)
    u_match, crit_match, d_match = calibrate_storage(db, templates, sim)
    print(f"\nstandard storage u={U:.2f}, criterion {crit_std:.3f}")
    print(f"accuracy-matched  u={u_match:.2f}, criterion {crit_match:.3f} "
          f"(d'={d_match:.2f})\n", flush=True)

    fh = open(os.path.join(out_folder, "continuous_rem.txt"), "w", encoding="utf-8")

    def w(s=""):
        print(s)
        fh.write(s + "\n")

    w("REM IN CONTINUOUS RECOGNITION")
    w("=" * 96)
    w(f"g={G}, w={W}, copy={C}. {N_SUBJECTS:,} simulated participants, each")
    w(f"inheriting the {len(templates[0][0])}-trial structure of one participant in the original")
    w("study: same sequence of first presentations and repeats, same lags, same")
    w("proportions of targets, fillers and vigilance repeats. Only the items are")
    w("simulated, so they have no memorability of their own.")
    w("")
    w("The criterion is set at each storage probability so that the false alarm")
    w("rate is 2%. The model runs at the standard storage probability and at the")
    w("storage probability whose overall d' is closest to the participants'")
    w("(matched accuracy).")
    w("")

    res = {}
    counts = {}
    for tag, u, crit in [("standard storage", U, crit_std),
                         ("accuracy-matched storage", u_match, crit_match)]:
        r = simulate(db, templates, crit, sim, N_SUBJECTS, RNG,
                     label=f"{tag[:8]} ", u=u)
        res[tag], counts[tag] = analyse(r, N_DATABASE, templates, tag, u, crit, w)
    fh.close()

    np.savez(os.path.join(out_folder, "continuous_bins.npz"),
             **{f"{k.split()[0]}_{f}": v[f]
                for k, v in res.items()
                for f in ("dprime", "crit_bin", "H", "F", "rho_d", "rho_c",
                          "cr_H", "cr_F", "u", "crit", "hit", "fa")})

    # per simulated participant (rows) and bin (columns), for the bootstrap of stats_figS4.py
    cell_count_arrays = {}
    for tag, point_counts in counts.items():
        for name, values in point_counts.items():
            cell_count_arrays[f"{tag.split()[0]}_{name}"] = as_whole_numbers(values)
    np.savez_compressed(os.path.join(out_folder, "continuous_cell_counts.npz"), **cell_count_arrays)

    for file_name in ("continuous_rem.txt", "continuous_bins.npz", "continuous_cell_counts.npz"):
        print(f"wrote {relative_to_release(os.path.join(out_folder, file_name))}")


if __name__ == "__main__":
    main()

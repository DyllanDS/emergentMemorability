"""
rem_final_spec.py: the preregistered REM predictions for the rare-feature face study (AsPredicted
#311640), run before any data were collected (SI Appendix, Section 3, "Predictions made before data
collection"; Fig. S10a).

WHAT IT DOES
REM (Shiffrin & Steyvers, 1997) is run through the face experiment's design at 729 parameter
settings: g {0.3, 0.4, 0.5} x number of features w {15, 20, 25} x storage probability u {0.2, 0.3,
0.4} x copying accuracy c {0.6, 0.7, 0.8} x the number of features the rare value occupies nfeat
{1, 3, 5} x the criterion fa {0.2, 0.42, 0.6} (placed so that this share of never-studied faces is
called old). At each setting 500 simulated participants study 50 faces (38 fillers, 6 rare-feature
faces, 6 ordinary faces whose partners are held back) and are tested on 36. The report counts the
settings at which each difference has the predicted sign, among them
    hits: rare > ordinary         preregistered prediction 1 (712 of the 729 settings)
    FA: rare foil > unrelated     preregistered prediction 2 (729 of 729)
    interaction > 0               prediction 3 in the four-term form this script prints (729 of 729);
                                  rem_729_rates.py also counts the registered two-term form
                                  d_a(rare vs never-studied) - d_a(rare vs its own foil) (729 of 729)

INPUTS   none (a simulation)
OUTPUTS  outputs/rem_final_spec.txt, the registered report; at the end it is compared, line for line,
         with the saved run data/paper_run/rem_final_spec.txt, the output the paper reports (its
         lines end in CR LF; the comparison ignores line ends)
SEEDS    setting i of the grid (i = 0 to 728, in itertools.product order of the grid below) draws
         its simulated participants from numpy.random.default_rng(i)
RUNTIME  about 2 minutes on an Apple M2 Pro (one process, about 60 MB)
RUN      python A3c_face_study_model_and_sdt_figS10_S11/code/rem_final_spec.py   (from the repository root)
The sweep runs inside main(), so rem_729_rates.py and rem_like_data.py can import evidence() and
GRID without running it.

REM's predictions for the design of the face study.

Fillers are ordinary faces, so they are barred only from carrying the rare value, not from looking
ordinary. The ordinary pair shares nothing that any two faces of the same demographic do not already
share (the filler pool spans the same demographics, so there is no separate population of unrelated
faces for it to be distinguished from). The ordinary pair is therefore two faces drawn like any
others, differing from a filler only in that its partner is held back.

The ordinary condition is thus a relabeling of the filler condition, and the model's commitments
concern only the rare feature.
"""
import itertools, os, sys
sys.dont_write_bytecode = True
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402

PATHS = release_paths.module_paths(__file__)
REPORT_FILE = os.path.join(PATHS['outputs'], 'rem_final_spec.txt')
PAPER_RUN_FILE = os.path.join(PATHS['paper_run'], 'rem_final_spec.txt')

N_SUB = 500                                  # simulated participants per setting
N_STUDY_FILLER, N_CRIT_PER_COND = 38, 6      # 38 fillers; 6 rare pairs and 6 ordinary pairs
# the six kinds of test face: studied rare, studied filler, studied ordinary-pair face, the rare
# face's never-studied partner (same rare value), a never-studied unrelated face, the ordinary
# face's never-studied partner
LAB = ["rare_old", "filler_old", "ord_old", "rare_foil", "unrelated", "ord_foil"]

# The grid. Standard Shiffrin and Steyvers values sit inside it; each axis is wide enough
# that a prediction surviving all of it is not an artifact of one parameter choice.
GRID = dict(g=[0.3, 0.4, 0.5], w=[15, 20, 25], u=[0.2, 0.3, 0.4], c=[0.6, 0.7, 0.8],
            nfeat=[1, 3, 5], fa=[0.2, 0.42, 0.6])


def evidence(g, w, u, c, nfeat, seed=0, n_sub=None):
    """Per-probe likelihood ratios and their labels, for n_sub simulated participants.

    Separated from simulate() so the same evidence can be scored two ways: directly, as
    moments of the log-odds, or by placing criteria on it and fitting the model the data
    are fitted with. The two are not the same estimator and need not agree.
    """
    rng = np.random.default_rng(seed)
    # The rare value: the smallest feature value whose base rate g (1 - g)^(v - 1) under REM's
    # geometric feature distribution is below 0.02 (g = 0.3: value 9, base rate 0.017; g = 0.4:
    # value 7, 0.019; g = 0.5: value 6, 0.016).
    rare_v = max(2, int(np.ceil(np.log(0.02 / g) / np.log(1 - g))) + 1)

    def draw(n):
        # n faces of w features, each feature value drawn from the geometric distribution with
        # parameter g; on the first nfeat features no face may carry the rare value by chance.
        out = rng.geometric(g, size=(n, w))
        bad = out[:, :nfeat] == rare_v          # nobody but a rare pair carries the rare value
        while bad.any():
            out[:, :nfeat][bad] = rng.geometric(g, size=int(bad.sum()))
            bad = out[:, :nfeat] == rare_v
        return out

    # base rate of a feature value v
    pv = lambda v: g * (1.0 - g) ** (v - 1)
    phi, lab = [], []
    for _ in range(n_sub or N_SUB):
        fillers = draw(N_STUDY_FILLER)
        # the rare pair shares the rare value; the ordinary pair shares nothing special
        rare_st = draw(N_CRIT_PER_COND); rare_st[:, :nfeat] = rare_v
        rare_fo = draw(N_CRIT_PER_COND); rare_fo[:, :nfeat] = rare_v
        ord_st, ord_fo = draw(N_CRIT_PER_COND), draw(N_CRIT_PER_COND)

        # Study: 38 fillers, 6 rare and 6 ordinary faces are stored. Each feature is stored with
        # probability u; a stored feature is copied correctly with probability c, otherwise a
        # random value is stored; 0 marks a feature that was not stored.
        items = np.vstack([fillers, rare_st, ord_st])
        t = np.zeros_like(items)
        got = rng.random(items.shape) < u
        right = rng.random(items.shape) < c
        t[got & right] = items[got & right]
        wrong = got & ~right
        t[wrong] = rng.geometric(g, size=int(wrong.sum()))

        # Test: of the 6 pairs of each kind, 3 are tested with the studied face and 3 with the
        # never-studied partner (the same 3 for rare and ordinary pairs); 12 studied fillers and
        # 12 never-studied unrelated faces: 36 probes.
        probes, labels = [], []
        o = rng.permutation(N_CRIT_PER_COND)
        for st, fo, nm in ((rare_st, rare_fo, "rare"), (ord_st, ord_fo, "ord")):
            probes.append(st[o[:3]]); labels += ["%s_old" % nm] * 3
            probes.append(fo[o[3:]]); labels += ["%s_foil" % nm] * 3
        probes.append(fillers[rng.permutation(N_STUDY_FILLER)[:12]]); labels += ["filler_old"] * 12
        probes.append(draw(12)); labels += ["unrelated"] * 12

        # REM's evidence: for each stored trace, the likelihood ratio of every feature (a stored
        # feature that matches the probe: (c + (1 - c) p(v)) / p(v), large for a rare value; a stored
        # feature that does not match: 1 - c; a feature not stored: 1), multiplied over features;
        # the odds that the probe was studied (phi) is the mean over the 50 traces.
        # Names: P holds the 36 probes
        # and T the 50 stored traces, laid out as [probe, trace, feature]; stored marks the features
        # each trace stored; p is the base rate of each probe value; lam is the likelihood ratio of
        # each feature of each probe-trace pair; its log summed over the features is the log
        # likelihood ratio of the pair, and phi averages exp() of that over the traces.
        P = np.vstack(probes)[:, None, :]; T = t[None, :, :]
        stored = T != 0; p = pv(P)
        lam = np.where(stored & (P == T), (c + (1.0 - c) * p) / p, np.where(stored, 1.0 - c, 1.0))
        phi.append(np.exp(np.log(lam).sum(axis=2)).mean(axis=1)); lab.append(np.array(labels))

    return np.concatenate(phi), np.concatenate(lab)


def simulate(g, w, u, c, nfeat, fa, seed=0):
    phi, lab = evidence(g, w, u, c, nfeat, seed=seed)
    # Old/new decision: the criterion is placed so that a share fa of the never-studied unrelated
    # faces is called old; the rate of each kind of face is its share called old.
    cv = np.quantile(phi[lab == "unrelated"], 1.0 - fa)
    said = phi > cv
    rate = {k: said[lab == k].mean() for k in LAB}

    def da(a, b):
        # d_a between two kinds of face, read off the evidence: the difference of the mean log odds
        # over the root mean square of their standard deviations
        la, lb = np.log(phi[lab == a]), np.log(phi[lab == b])
        return ((la.mean() - lb.mean())
                * np.sqrt(2.0 / (la.std(ddof=1) ** 2 + lb.std(ddof=1) ** 2)))

    inter = ((da("rare_old", "unrelated") - da("ord_old", "unrelated"))
             - (da("rare_old", "rare_foil") - da("ord_old", "ord_foil")))
    # What the feature is worth once each kind of face is scored against its own kind of lure.
    # The interaction says how much of the rare advantage is the comparison set; this says what
    # is left over, and the model is not obliged to leave anything.
    buys = da("rare_old", "rare_foil") - da("ord_old", "unrelated")
    return rate, inter, buys


# Each claim is checked in every parameter setting: r is the setting's rate of "old" answers for each
# kind of face (rare_old, ord_old, filler_old, rare_foil, ord_foil, unrelated) and i its d_a
# interaction; main() counts the settings in which the claim holds.
CLAIMS = [
    ("hits: rare > ordinary", lambda r, i: r["rare_old"] > r["ord_old"]),
    ("hits: rare > filler", lambda r, i: r["rare_old"] > r["filler_old"]),
    ("hits: filler > ordinary", lambda r, i: r["filler_old"] > r["ord_old"]),
    ("FA: rare foil > ordinary foil", lambda r, i: r["rare_foil"] > r["ord_foil"]),
    ("FA: rare foil > unrelated", lambda r, i: r["rare_foil"] > r["unrelated"]),
    ("FA: ordinary foil vs unrelated (either way)", lambda r, i: r["ord_foil"] < r["unrelated"]),
    ("rare foil < rare hits", lambda r, i: r["rare_foil"] < r["rare_old"]),
    ("interaction > 0", lambda r, i: i > 0),
]


def main():
    keys = list(GRID)
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    hits = {n: 0 for n, _ in CLAIMS}
    gaps = {n: [] for n in ("ord_vs_filler", "ordfoil_vs_unrel")}
    buys_all, inter_all = [], []
    means = {k: [] for k in LAB}
    for i, vals in enumerate(combos):
        # setting i uses seed i
        rate, inter, buys = simulate(seed=i, **dict(zip(keys, vals)))
        for n, f in CLAIMS:
            hits[n] += bool(f(rate, inter))
        buys_all.append(buys); inter_all.append(inter)
        gaps["ord_vs_filler"].append(rate["ord_old"] - rate["filler_old"])
        gaps["ordfoil_vs_unrel"].append(rate["ord_foil"] - rate["unrelated"])
        for k in LAB:
            means[k].append(rate[k])

    n = len(combos)
    out = ["REM predictions for the face design", "=" * 60, "",
           "%d parameter combinations. The ordinary pair shares nothing special, so the" % n,
           "ordinary condition should behave exactly like the filler condition.", ""]
    for name, _ in CLAIMS:
        k = hits[name]
        tag = "COMMITMENT" if k == n else ("%d%%" % round(100 * k / n))
        out.append("  %-46s %4d/%d  %s" % (name, k, n, tag))
    out += ["", "Is the ordinary condition a relabeling of the filler condition?",
            "  ordinary studied minus filler studied : mean %+.4f (SD %.4f)"
            % (np.mean(gaps["ord_vs_filler"]), np.std(gaps["ord_vs_filler"])),
            "  ordinary foil minus unrelated new     : mean %+.4f (SD %.4f)"
            % (np.mean(gaps["ordfoil_vs_unrel"]), np.std(gaps["ordfoil_vs_unrel"])),
            "", "  mean rates: " + "  ".join("%s %.3f" % (k, np.mean(means[k])) for k in LAB)]

    b = np.array(buys_all); ia = np.array(inter_all)
    out += ["", "Does the rare feature still buy discriminability once each kind of face is",
            "scored against its own kind of lure?  d_a(rare vs its own foil) - d_a(ordinary",
            "vs never-studied), across the same grid.", "",
            "  positive (a net benefit survives)   %4d/%d  %d%%" % ((b > 0).sum(), n, round(100 * (b > 0).mean())),
            "  mean %+.3f, SD %.3f, range %+.3f to %+.3f" % (b.mean(), b.std(), b.min(), b.max()),
            "",
            "  for comparison, the interaction     mean %+.3f, range %+.3f to %+.3f" % (ia.mean(), ia.min(), ia.max()),
            "  share of the rare advantage that is the comparison set: mean %d%%"
            % round(100 * (ia / (ia + b)).mean())]
    txt = "\n".join(out) + "\n"
    open(REPORT_FILE, "w", encoding="utf-8").write(txt)
    print(txt)

    # Compare the report with the saved run in data/paper_run/ (the output the paper reports), line for line.
    with open(PAPER_RUN_FILE, encoding="utf-8") as handle:
        paper_run_lines = handle.read().replace("\r\n", "\n").split("\n")
    report_lines = txt.split("\n")
    if report_lines == paper_run_lines:
        print("outputs/rem_final_spec.txt equals the saved run data/paper_run/rem_final_spec.txt "
              "line for line (%d lines): yes" % (len(report_lines) - 1))
    else:
        print("outputs/rem_final_spec.txt DIFFERS from the saved run data/paper_run/rem_final_spec.txt:")
        for line_number, (new_line, old_line) in enumerate(zip(report_lines, paper_run_lines), 1):
            if new_line != old_line:
                print("  line %d: this run %r; saved run %r" % (line_number, new_line, old_line))
        if len(report_lines) != len(paper_run_lines):
            print("  this run has %d lines, the saved run %d" % (len(report_lines), len(paper_run_lines)))


if __name__ == "__main__":
    main()

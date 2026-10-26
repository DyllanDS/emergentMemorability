"""Hybrid-similarity GCM engine of SI Appendix, Section 2 (A Hybrid-Similarity Exemplar Model).

The design follows the REM simulations of Fig. 1: items are vectors of w features drawn from a
geometric distribution with parameter g, each feature is stored with probability u and stored
correctly with probability cenc (otherwise replaced by a fresh draw), and distinctiveness is
mean Euclidean distance to the rest of the set.

Only the familiarity rule changes. The hybrid-similarity model of Nosofsky and Zaki (2003)
and Knapp, Nosofsky and Busey (2006) combines continuous similarity with discrete
distinctive features. A feature value counts as a distinctive discrete feature when its
frequency on that dimension, in a reference population, is below q. The reference population
is one of:
    world  the generating distribution (the observer's lifetime experience, as in REM's base rates)
    set    the experiment's whole stimulus set
    list   the items this participant studied (computed separately for every simulated participant)
    mix    an even blend of list and world frequencies
At g = .65, q = .10 marks the values >= 3 under the world and set definitions. Then

    s(probe, trace) = exp(-c * d) * exp(gamma * n_match_distinctive - delta * n_mismatch_distinctive)
    familiarity     = sum of s over the studied traces
    respond "old" if familiarity exceeds a criterion

where d is the Euclidean distance over the trace's stored features (rescaled to all w features:
multiplied by sqrt(w / number stored)), n_match_distinctive counts stored features on
which probe and trace share a distinctive value, and n_mismatch_distinctive counts stored
features on which the two differ and at least one has a distinctive value (one distinctive
value against a common one, or two different distinctive values, as in Nosofsky & Zaki,
2003). gamma = delta = 0 is the ordinary GCM.

List-relative frequencies are computed from the items as studied, not from the noisy traces:
the participant is assumed to know which values were common on the list they saw.
"""
import numpy as np
from scipy.stats import norm


# ---- 1. items and noisy encoding ----
def make_items(rng, n, w, g):
    # numpy's geometric starts at 1, the support of the feature values in the paper.
    return rng.geometric(g, (n, w)).astype(np.int16)


def encode(rng, X, u, cenc, g):
    """Noisy traces and their stored-feature masks; X has shape (..., n_items, w)."""
    stored = rng.random(X.shape) < u
    wrong = rng.random(X.shape) >= cenc
    tr = np.where(wrong, rng.geometric(g, X.shape).astype(np.int16), X)
    return tr, stored


# ---- 2. feature-value frequencies in each reference population ----
# Values above V_MAX are pooled; at g >= .30 they have world probability below 1e-12.
V_MAX = 80


def world_freq(g, w):
    """[dimension, value] probability of each value under the generating distribution."""
    v = np.arange(V_MAX + 1)
    pmf = np.where(v >= 1, g * (1 - g) ** np.maximum(v - 1, 0), 0.0)
    return np.tile(pmf, (w, 1))


def set_freq(X):
    w = X.shape[1]
    f = np.zeros((w, V_MAX + 1))
    for j in range(w):
        vals, counts = np.unique(np.minimum(X[:, j], V_MAX), return_counts=True)
        f[j, vals] = counts / len(X)
    return f


def list_freq(Xstudy):
    """Per-participant [run, dimension, value] frequency within that participant's study list."""
    R, L, w = Xstudy.shape
    f = np.zeros((R, w, V_MAX + 1))
    r = np.repeat(np.arange(R), L * w)
    d = np.tile(np.repeat(np.arange(w)[None], L, 0).ravel(), R)
    np.add.at(f, (r, d, np.minimum(Xstudy, V_MAX).ravel()), 1.0 / L)
    return f


def ref_table(p, Xstudy):
    """Distinctiveness table for these runs: [dim, value] (shared) or [run, dim, value] (per run)."""
    ref = p.get("ref", "set")
    if ref == "set":
        return p["set_freq"] < p["q"]
    wf = world_freq(p["g"], p["w"])
    if ref == "world":
        return wf < p["q"]
    lf = list_freq(Xstudy)
    if ref == "list":
        return lf < p["q"]
    return 0.5 * lf + 0.5 * wf[None] < p["q"]


# ---- 3. summed hybrid similarity of each probe to the studied traces ----
def familiarity(probes, traces, stored, p, tab=None):
    """probes (R, P, w); traces, stored (R, T, w) -> familiarity (R, P).
    tab: distinctiveness table from ref_table, [dim, value] or [run, dim, value]."""
    # The same rule, written with explicit loops one probe, trace and feature at a time, is in
    # stats_figS7.py, which checks this function against it. Names, in the model's terms, for every
    # array laid out as [run, probe, trace, feature]:
    #   pr, tr    the probe's and the trace's feature values
    #   st        True where the trace stored that feature
    #   cnt       the number of features the trace stored (at least 1)
    #   sq        the squared probe-trace differences, summed over the stored features
    #   d         the Euclidean distance, rescaled to all w features: sqrt(sq / cnt * w)
    #   logs      the log of the similarity, -c * d
    #   dp, dt    True where the probe's (trace's) value is a distinctive value: its frequency in the
    #             reference population (tab) is below q
    #   match     stored features on which probe and trace share the same distinctive value
    #   mism      stored features on which they differ and at least one of the two values is distinctive
    #   result    familiarity of each probe = the sum over the studied traces of
    #             exp(-c * d + gamma * match - delta * mism)
    pr = probes[:, :, None, :].astype(np.float32)
    tr = traces[:, None, :, :].astype(np.float32)
    st = stored[:, None, :, :]
    cnt = np.maximum(st.sum(-1), 1)
    sq = (((pr - tr) ** 2) * st).sum(-1)
    d = np.sqrt(sq / cnt * probes.shape[-1])
    logs = -p["c"] * d
    if p["gamma"] > 0 or p["delta"] > 0:
        dims = np.arange(probes.shape[-1])
        if tab.ndim == 2:
            dp = tab[dims, np.minimum(probes, V_MAX)][:, :, None, :]
            dt = tab[dims, np.minimum(traces, V_MAX)][:, None, :, :]
        else:
            rr = np.arange(tab.shape[0])[:, None, None]
            dp = tab[rr, dims, np.minimum(probes, V_MAX)][:, :, None, :]
            dt = tab[rr, dims, np.minimum(traces, V_MAX)][:, None, :, :]
        match = (dp & dt & (pr == tr) & st).sum(-1)
        mism = ((dp | dt) & (pr != tr) & st).sum(-1)
        logs = logs + p["gamma"] * match - p["delta"] * mism
    return np.exp(logs).sum(-1)


# ---- 4. distinctiveness measure and z-transform ----
def crowding(X):
    """Mean Euclidean distance to every other item, negated so larger means more crowded."""
    Xf = X.astype(np.float64)
    sq = (Xf ** 2).sum(1)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * Xf @ Xf.T, 0))
    np.fill_diagonal(D, np.nan)
    return -np.nanmean(D, 1), D


def zc(x, lo=1e-3):
    return norm.ppf(np.clip(x, lo, 1 - lo))

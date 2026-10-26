"""
extract_trials.py: the trial table of the original THINGS memorability study (Kramer et al. 2023),
step 1 of the SI Appendix, Section 5 chain.

WHAT IT DOES
    Reads the Kramer et al. trial file (MATLAB v7.3, so h5py is needed) and writes one row per trial:
    participant session, trial number within the session, image number, trial type and response.
    Sessions whose image order is empty (people who left at once) are dropped; the file has 21,572
    sessions and 21,137 are kept (6,223,953 trials). The session's name field (an MTurk worker id)
    is not read.

    Codes in the source:
      imtypeorder  1 = target first presentation   2 = target repeat
                   3 = filler first presentation   4 = vigilance repeat
      perforder   11 = hit   12 = miss   13 = false alarm   14 = correct rejection

INPUT   external/kramer_trials/imbysub-all-11-16-21.mat   (EXTERNAL_INPUTS.md M1; from the authors of
                                                           Kramer et al. 2023)
        shared/things/things_image_mapping.csv             (image number -> concept number)
OUTPUT  external/kramer_trials/derived/trials.npz          (sub, trial, image_nr, ttype, perf, concept;
                                                            one row per trial; not distributed)
SEEDS   none.  RUNTIME  long (every trial's image name is an HDF5 reference read one at a time).
Run from the repository root:  python A5_things_context_effects_figS13_S14/code/extract_trials.py
"""
import time
import numpy as np
import h5py
import pandas as pd

import a5_common

RAW = a5_common.RELEASE_ROOT / a5_common.RAW_TRIAL_FILE_RELATIVE
OUT = a5_common.make_derived_folder()


def main():
    f = h5py.File(RAW, "r")
    sd = f["subdata"]
    n_sub = sd["imorder"].shape[0]

    # ---- 1. decode the image dictionary once, so per-trial strings become ints ----
    # imorder stores the id strings themselves rather than dictionary indices, but
    # decoding the dictionary first lets us cache string -> int and skip re-parsing
    # the same ~26k strings six million times.
    dic = f["imdictionary"]
    cache = {}
    for i in range(dic.shape[0]):
        ref = dic[i, 0]
        txt = "".join(chr(int(x)) for x in np.array(f[ref][()]).flatten())
        cache[txt] = int(txt)

    # ---- 2. walk subjects, dereferencing the three per-trial fields ----
    sub_id, trial_no, image_nr, ttype, perf = [], [], [], [], []
    dropped = []
    t0 = time.time()
    for i in range(n_sub):
        # Participants who abandoned the task immediately have an empty imorder
        # cell, which lands as a plain numeric array rather than object
        # references. There is no image identity to recover, so they are dropped.
        ds = f[sd["imorder"][i, 0]]
        if h5py.check_ref_dtype(ds.dtype) is None:
            dropped.append((i, len(np.array(f[sd["imtypeorder"][i, 0]][()]).flatten())))
            continue

        imo = np.array(ds[()]).reshape(-1)
        ito = np.array(f[sd["imtypeorder"][i, 0]][()]).flatten()
        pfo = np.array(f[sd["perforder"][i, 0]][()]).flatten()
        n = min(len(ito), len(pfo), len(imo))
        if n == 0:
            continue

        ids = np.empty(n, dtype=np.int32)
        for t in range(n):
            ref = imo[t]
            txt = "".join(chr(int(x)) for x in np.array(f[ref][()]).flatten())
            v = cache.get(txt)
            ids[t] = v if v is not None else int(txt)

        # the session's "name" field (an MTurk worker id) is not read

        sub_id.append(np.full(n, i, dtype=np.int32))
        trial_no.append(np.arange(n, dtype=np.int16))
        image_nr.append(ids)
        ttype.append(ito[:n].astype(np.int8))
        perf.append(pfo[:n].astype(np.int8))

        if i % 1000 == 0:
            el = time.time() - t0
            print(f"  {i}/{n_sub} subjects  ({el:.0f}s elapsed)", flush=True)

    tab = dict(
        sub=np.concatenate(sub_id),
        trial=np.concatenate(trial_no),
        image_nr=np.concatenate(image_nr),
        ttype=np.concatenate(ttype),
        perf=np.concatenate(perf),
    )

    # ---- 3. attach concept membership, the space similarity is defined over ----
    m = pd.read_csv(a5_common.image_mapping_file())
    img2conc = np.zeros(int(m.image_nr.max()) + 1, dtype=np.int16)
    img2conc[m.image_nr.values] = m.concept_nr.values  # 1-based concept numbers
    tab["concept"] = img2conc[tab["image_nr"]]

    np.savez_compressed(OUT / "trials.npz", **tab)

    n = len(tab["sub"])
    print(f"\nwrote {a5_common.relative(OUT / 'trials.npz')}")
    print(f"  {n:,} trials from {len(np.unique(tab['sub'])):,} subjects")
    if dropped:
        mx = max(t for _, t in dropped)
        print(f"  dropped {len(dropped)} subjects with no recoverable image order "
              f"(max {mx} trials each)")
    print(f"  trial types: {dict(zip(*np.unique(tab['ttype'], return_counts=True)))}")
    print(f"  responses:   {dict(zip(*np.unique(tab['perf'], return_counts=True)))}")
    print(f"  concepts covered: {len(np.unique(tab['concept']))}")
    print(f"  elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

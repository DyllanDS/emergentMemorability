Folder for two inputs that are not public (EXTERNAL_INPUTS.md, M1 and M2).

M1. The trial file of Kramer et al. (2023), imbysub-all-11-16-21.mat (about 6.3 GB, MATLAB v7.3), which
the original authors share on request. With it in this folder, module A5 computes SI Appendix,
Section 5 (Figs. S13 and S14). Run A5 first, then A2c (the model side of SI Appendix, Fig. S4), A3b
(SI Appendix, Fig. S9c) and A4a (the THINGS results of SI Appendix, Section 4), which read the trial
table that A5 writes:

    python A5_things_context_effects_figS13_S14/code/run_module.py
    python A2c_continuous_recognition_figS4/code/run_module.py
    python A3b_hits_and_false_alarms_figS9/code/run_module.py
    python A4a_crowding_across_scales_figS12_tableS2/code/run_module.py

Module A5 writes per-trial tables to derived/ in this folder. They hold per-trial data of a study that
is not public, so do not share them.

M2. The per-image count table things_per_image_counts.csv (columns image_nr, n_old, hits, n_new,
false_alarms; the old trials include every repeat of a photograph). With it in this folder, module A4b
computes the memorability side of the THINGS concepts analysis (SI Appendix, Section 4) and module A3b
the original THINGS hit and false alarm rates of the more and less memorable critical items of Fig. 3
(the THINGS side of its halves comparison).

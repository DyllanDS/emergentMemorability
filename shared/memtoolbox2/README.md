# shared/memtoolbox2/: the place for MemToolbox2

Module A3c (`A3c_face_study_model_and_sdt_figS10_S11/`) fits the preregistered unequal-variance
signal-detection model of the face study with the MemToolbox2 Python package (its `UVSDModel`, `ByCell` and
`MLE` and the code they import). The package is not included (EXTERNAL_INPUTS.md, M5).

The fitted values of the face study are saved in `A3c_face_study_model_and_sdt_figS10_S11/data/provided/`
(the point fit, the bootstrap resamples and the reports). Module A3c reads the d_a values of Fig. 4c,
SI Appendix, Figs. S10b and S11b, and SI Appendix, Section 3 from these files, and without the package it
prints `PROVIDED OUTPUT: regenerating needs ...; see EXTERNAL_INPUTS.md`.

To fit the model again, put the package here, so that `shared/memtoolbox2/memtoolbox/` (or
`shared/memtoolbox2/python/memtoolbox/`) exists. Module A3c uses only a copy in this folder and
never one found elsewhere on the Python path. With the package in place,
`python A3c_face_study_model_and_sdt_figS10_S11/code/run_module.py` also runs the registered fit (about
1.5 minutes), the leave-one-pair-out refits (about 7 minutes) and the fits of the 243-setting model run
(several hours; `--skip-long` leaves them out), and compares them with the saved results. The module does
not run the two bootstraps again (10,000 resamples of the participants, about 7.4 hours on 22 workers, and
1,000 resamples of the six rare features); their saved resamples are used with or without the package.
The README of module A3c gives their commands.

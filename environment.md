# Environment: what the code needs to run

## Tested setup

- Apple M2 Pro (10 cores) with 16 GB of memory, macOS 27.0.1.
- MATLAB R2023b (23.2.0, Apple silicon) with the Statistics and Machine Learning Toolbox.
- Python 3.12.4 with numpy 1.26.4, scipy 1.13.1, pandas 2.2.2, matplotlib 3.8.4, scikit-learn 1.9.0 and
  h5py 3.11.0, the versions pinned in `requirements.txt`.

The pinned packages need Python 3.11 or 3.12, because scikit-learn 1.9.0 requires Python 3.11 or later and
numpy 1.26.4 and several of the other pinned versions have no packages for Python 3.13. To install them in a
new virtual environment, run these commands from the repository root (`python3.11` works in place of
`python3.12`):

    python3.12 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt

## What each part needs

| Software | Used by |
|---|---|
| MATLAB with the Statistics and Machine Learning Toolbox | modules 01, 02, 03, 05 and 06, and the MATLAB steps of A1, A2a, A2b, A2d and A2e. No parallel pool is started, so the Parallel Computing Toolbox is not needed. `python run_all.py --skip-matlab` runs everything else and uses the MATLAB outputs in the repository. |
| Python with numpy, scipy, pandas and matplotlib | every Python step |
| scikit-learn | A3a (the principal components of the model space), A6a (the ridge step), A5 (with M1 only) |
| h5py | A6a (it reads the Isola et al. participant matrices, a MATLAB v7.3 file), A5 with M1 (it reads the trial file) |
| the Python standard library only | the collectors `collect_numbers.py` and `statistics/collect_all_tests.py` |

`run_all.py` finds MATLAB through the environment variable `MATLAB` (the full path of the `matlab` program or
of the MATLAB folder), else `matlab` on the PATH, else a standard installation folder (R2023b first).

## Optional software (only for the inputs of EXTERNAL_INPUTS.md)

| Software | Needed for |
|---|---|
| openpyxl | reading the score file of the 10k US Adult Faces Database (M4) |
| PyTorch, torchvision, timm, the OpenAI CLIP package and CORnet | extracting deep-network embeddings from images (M3, M4 and M8; the extraction was tested with PyTorch 2.5.0) |
| PyTorch | refitting the SPoSE embeddings of SI Appendix, Fig. S16 (M9) |
| MemToolbox2 (Python) | refitting the face study's unequal-variance model (M5); not publicly available |

The OpenAI CLIP package installs with `python -m pip install git+https://github.com/openai/CLIP.git` and
CORnet with `python -m pip install git+https://github.com/dicarlolab/CORnet` (the package that
`pip install clip` installs is unrelated). `shared/embedding_extraction/README.md` lists the networks, layers,
weights and preprocessing.

## Settings the run scripts make

- `MPLBACKEND=Agg` (figures are drawn without a screen) and `PYTHONDONTWRITEBYTECODE=1` (no `__pycache__`
  folders): `run_all.py` and the modules' `run_module.py` set both for every step they start. To run a single
  script by hand, set them yourself (in a macOS or Linux shell):

      MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 python <module>/code/<script>.py

- `PYTHONUTF8=1` (some steps print d′, ρ or τ, which the Windows code pages lack): `run_all.py` sets it for
  its steps. When a module's scripts are run directly on Windows, set it too.
- MATLAB starts with `restoredefaultpath` (in `run_all.py`, `run_all.m`, the MATLAB commands of the module
  READMEs and the modules that start MATLAB themselves), for that MATLAB session only. Folders a user has
  added to the MATLAB path could otherwise replace MATLAB's own functions (another toolbox's `tcdf.m`, for
  example, changes the p values that MATLAB's `corr` returns far below any printed threshold).
- Every script finds the folders of the repository from its own location, so the code runs from wherever the
  repository is cloned. The commands in the READMEs are given from the repository root.

## Memory, disk and time

- Disk: the repository takes about 620 MB, most of it the deep-network embeddings in `shared/`.
- Memory: at most about 2.5 GB in one process (module 01, while it compares its new run with the saved run;
  the ridge step of A6a needs about 2 GB). Module A2f runs two worker processes at once (about 3 to 4 GB
  together in its parameter sweep), as does A3c with the optional MemToolbox2 (M5); the other modules run
  their steps one process at a time. With the optional inputs, the deep-network step of A5 needs about 5 GB
  (M1 and M3), and extracting the DINOv3 embeddings needs about 17 GB with the standard extractor
  (`shared/embedding_extraction/README.md` gives a low-memory route).
- Time on the tested laptop:

| Command | Time |
|---|---|
| `python run_all.py` | about 1 hour 40 minutes |
| `python run_all.py --skip-long` | about 1 hour |
| `matlab -batch "run_all"` | about 35 minutes |
| `matlab -batch "run_all(true)"` | about 15 minutes |

Each module README gives the runtime of that module.

## Numbers across machines

- Every random computation uses a fixed seed (MATLAB `rng(seed, 'twister')`, Python
  `numpy.random.default_rng(seed)`, scikit-learn `random_state=seed` and PyTorch `torch.manual_seed(seed)`),
  and the code keeps the order of the random draws behind each reported number. The simulations of modules
  A2a, A2b, A2d and A2e draw all their cells from one random stream in file order, so they run serially,
  never in a parallel pool, and a single cell gives its reported values only within the whole simulation.
  Run the modules in the order of `run_all.py`, because some modules read files that earlier modules write.
  The module READMEs give each seed.
- Some results depend on the last digits of floating-point arithmetic, which can differ between machines,
  numerical libraries and operating systems (for example p values far below any printed threshold, and
  principal components computed in single precision, with differences of about 1e-5 in modules A3a and A4b).
  The model simulations of A3a carry these differences into their results: between the Anaconda and pip
  installations of the pinned packages, the simulated rates of an image-by-set cell differ by up to about
  0.004 and the values of Fig. S8 by up to about 0.0004. The designed lists of A2f order items at equal
  distances as numpy's default sort does on Apple silicon and on x86 processors without AVX-512; on processors
  with AVX-512, numpy orders such ties differently, which changes those lists and moves their d′ values in the
  second decimal, so A2f's comparison with its saved run lists these values and the changes the SI Appendix
  prints for the designed lists (ledger L6-108) can round differently. On the tested setups no printed number
  changes. Where a module compares a new run with a saved run, it allows for differences in the last digits;
  module 03, for example, stops only if a printed value of Fig. 3e would change.
- The figures that the Python scripts draw were made with Anaconda's matplotlib 3.8.4 (FreeType 2.12.1) and
  Pillow 10.3.0; with the pip packages of `requirements.txt`, redrawn figures differ slightly in text
  antialiasing and in the decoded Fig. 4a face images (JPEG files), while the plotted values in `outputs/figure_data/`
  differ only as described above.
- pandas reads the CSV tables of the repository exactly only with `float_precision='round_trip'` (its default
  parser can be off in the last digit), and every shared loader passes it. MATLAB `readtable` and Python
  `float()` read them exactly.

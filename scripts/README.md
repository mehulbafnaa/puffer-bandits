Scripts
=======

This folder contains automation scripts to run reproducible sweeps.

run_all.sh
----------
Matrix runner for bandit experiments using `uv`. It supports a device and seed matrix, category toggles, and environment‑variable overrides for hyperparameters.

Usage
  - Single device: `bash scripts/run_all.sh mps`
  - Device matrix: `DEVICES="cpu mps" bash scripts/run_all.sh`
  - Seeds: `SEEDS="0 1" bash scripts/run_all.sh mps`

Categories (env toggles)
  - `DO_SMOKE=1|0`       # quick serial TUI runs (default 1)
  - `DO_THROUGHPUT=1|0`  # larger MP runs (default 1)
  - `DO_NEURAL=1|0`      # neural agent runs (default 1)

Core knobs (env)
  - `K=10 D=8`             # problem sizes
  - `WORKERS=8`            # MP workers (runs must be divisible)
  - `RUNS_BASIC, T_BASIC`  # non‑contextual defaults (1024, 2000)
  - `RUNS_CTX, T_CTX`      # contextual defaults (512, 600)
  - `RUNS_NEURAL, T_NEURAL`# neural defaults (256, 400)

Hyperparameters (env)
  - `ALPHAS` (LinUCB), `VS` (LinTS)
  - `GAMMAS_EXP3`, `GAMMAS_EXP3IX`
  - `DUCB_C`, `DUCB_DISCOUNT`, `SW_C`, `SW_WINDOW`
  - `NEURAL_FEATURES`, `NEURAL_ENSEMBLES`, `NEURAL_HIDDEN`, `NEURAL_DEPTH`, `NEURAL_DROPOUT`

Outputs
  Logs are written to `sweeps_logs/<timestamp>_<device>_s<seed>/*.log` and the underlying runners can also save CSV/plots.


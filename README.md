# MAB_GPU (uv-based)

This package provides GPU-ready bandit agents and a PufferLib vectorized runner.
It uses a `pyproject.toml` and recommends the fast `uv` tool for dependency
management and execution.

## Prereqs
- Python 3.9+
- `uv` installed (`pipx install uv` or follow https://docs.astral.sh/uv/)

## Setup with uv

```
# Create a virtualenv and install project deps
uv venv
uv sync

# Install PyTorch per your platform (choose one)
# CUDA (Linux/WSL2):
uv pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
# CPU-only (portable):
uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
# Apple Silicon (MPS):
uv pip install torch torchvision torchaudio

# Install local PufferLib for imports
uv pip install -e ../PufferLib
```

## Run

```
# Using uv script alias from pyproject (alias: `runner` or `mab-gpu-runner`)
uv run mab-gpu-runner --algo klucb --k 10 --T 100000 --runs 8192 --device cuda

# Backward compatible aliases also work:
#   --agent klucb   == --algo klucb
#   --num-envs N    == --runs N
```

Notes:
- On macOS (MPS), use `--device mps`.
- If you prefer no GPU, use `--device cpu` and a smaller `--num-envs`.
- The environment remains CPU (batched via PufferLib); agent logic is on GPU/MPS/CPU.

Additional algorithms (non-contextual):

```
# KL-UCB (Bernoulli)
uv run mab-gpu-runner --algo klucb --k 10 --T 50000 --runs 4096 --kl-alpha 3.0 --device cuda

# Discounted UCB for non-stationary bandits
uv run mab-gpu-runner --algo ducb --k 10 --T 50000 --runs 4096 --discount 0.99 --c 2.0 --nonstationary --sigma 0.1 --device cuda

# Sliding-Window UCB for non-stationary bandits (window-limited)
uv run mab-gpu-runner --algo swucb --k 10 --T 50000 --runs 4096 --window 200 --c 2.0 --nonstationary --sigma 0.1 --device cuda
```

### Advanced runner (contextual/adversarial)

We also include a contextual bandit environment and advanced agents (LinUCB, LinTS, EXP3/EXP3-IX, NeuralTS, NeuralLinearTS).

```
# Contextual bandit (k arms, d-dim features), LinUCB on GPU
uv run mab-gpu-advanced --env contextual --algo linucb --k 10 --d 8 --T 5000 --runs 4096 --alpha 1.0 --lam 1.0 --device cuda

# Linear Thompson Sampling
uv run mab-gpu-advanced --env contextual --algo lints --k 10 --d 8 --T 5000 --runs 4096 --v 0.1 --lam 1.0 --device cuda

# EXP3 (adversarial-style) on Bernoulli bandit
uv run mab-gpu-advanced --env bernoulli --algo exp3 --k 10 --T 5000 --runs 4096 --gamma 0.07 --device cuda

# NeuralTS (bootstrapped ensemble) on contextual bandit
uv run mab-gpu-advanced --env contextual --algo neuralts --k 10 --d 8 --T 5000 --runs 4096 --hidden 128 --depth 2 --ensembles 5 --dropout 0.1 --device cuda

# EXP3-IX (implicit exploration) on Bernoulli bandit
uv run mab-gpu-advanced --env bernoulli --algo exp3ix --k 10 --T 5000 --runs 4096 --gamma 0.05 --device cuda

# NeuralLinearTS (encoder + Bayesian linear head)
uv run mab-gpu-advanced --env contextual --algo neurallinear --k 10 --d 8 --T 5000 --runs 4096 --features 64 --linlam 1.0 --linv 0.1 --device cuda
```

The contextual env returns per-step features of shape `(k, d)` and rewards drawn from a
logistic-Bernoulli model. The runner computes mean reward, % optimal, and cumulative regret
with 95% CIs and saves figures under `plots_gpu/`.

### Native PufferLib Environments (TUI)

We also provide native PufferLib environments for both bandits with an in-terminal dashboard (Rich TUI). Use the native runner for tight integration:

```
# Contextual (native Puffer, serial)
uv run mab-gpu-puffer --env contextual --algo linucb --k 10 --d 8 --T 600 --runs 256 --vector serial --device mps --tui

# Bernoulli (native Puffer, serial)
uv run mab-gpu-puffer --env bernoulli --algo klucb --k 10 --T 600 --runs 512 --vector serial --device mps --tui

# High-throughput (multiprocessing)
uv run mab-gpu-puffer --env contextual --algo lints --k 10 --d 8 --T 1000 --runs 1024 --vector mp --num-workers 8 --device mps --tui
```

### Notes on running
This project is intended to be run via `uv` only. Use the provided script aliases (installed by `uv sync`) with `uv run` as shown above. Direct execution with `python .../runner_*.py` is not supported.

### Sweeps and plots

Run a quick sweep of common hyperparameters and append results to CSVs:

```
bash scripts/run_gpu_sweeps.sh cuda
```

Then generate a comparison plot, for example KL-UCB alpha sweep:

```
uv run python MAB_GPU/plot_sweeps.py --file plots_gpu/summary.csv --algo klucb --param kl_alpha --out plots_gpu/sweep_klucb_alpha.png
```

## Uninstall / Clean

```
uv pip uninstall -y "mab-gpu"
rm -rf .venv
```
## Layout (functional core, OO shell)

- `core/`: Pure tensor functions (KL-UCB bisection, EXP3 updates, Sherman–Morrison, etc.).
- `utils/`: Device utilities and small helpers.
- `agents.py`: Bernoulli-bandit agents (KL-UCB, DUCB, SW-UCB) using `core/` ops.
- `advanced_agents.py`: Contextual/adversarial agents (LinUCB/LinTS, EXP3/EXP3-IX, NeuralTS, NeuralLinearTS) using `core/` ops.
- `bandit_env.py`, `contextual_env.py` (+ `envs/__init__.py` re-exports): Minimal Gymnasium envs.
- `runner_puffer.py`, `runner_puffer_advanced.py`: PufferLib vectorized runners (non-contextual and contextual/adversarial).
The native environments are implemented in `MAB_GPU/puffer_envs.py` and integrate directly with `pufferlib.vector.Serial` and `pufferlib.vector.Multiprocessing`.

## Run Catalog (Suggested)

Below are curated commands grouped by environment and algorithm. Replace `--device mps` with `cpu` or `cuda` as needed. Keep `--runs` divisible by `--num-workers` in MP mode.

### Bernoulli (Non‑Contextual)

Smoke (TUI, serial):

```
uv run mab-gpu-puffer --env bernoulli --algo klucb --k 10 --T 600 --runs 512 --vector serial --device mps --tui --log-every 20
```

Throughput (TUI, MP):

```
uv run mab-gpu-puffer --env bernoulli --algo klucb --k 10 --T 2000 --runs 1024 --vector mp --num-workers 8 --device mps --tui --log-every 100
```

Non‑stationary (TUI, serial):

```
uv run mab-gpu-puffer --env bernoulli --algo ducb --k 10 --T 1000 --runs 512 --vector serial --device mps --tui --log-every 20 --nonstationary --sigma 0.1
uv run mab-gpu-puffer --env bernoulli --algo swucb --k 10 --T 1000 --runs 512 --vector serial --device mps --tui --log-every 20
```

### Contextual (Linear)

LinUCB (TUI, serial):

```
uv run mab-gpu-puffer --env contextual --algo linucb --k 10 --d 8 --T 600 --runs 256 --vector serial --device mps --tui --log-every 10 --alpha 1.0 --lam 1.0
```

LinTS (TUI, MP):

```
uv run mab-gpu-puffer --env contextual --algo lints --k 10 --d 8 --T 1000 --runs 1024 --vector mp --num-workers 8 --device mps --tui --log-every 50 --v 0.1 --lam 1.0
```

### Contextual (Neural)

NeuralTS (TUI, MP):

```
uv run mab-gpu-puffer --env contextual --algo neuralts --k 10 --d 8 --T 600 --runs 256 --vector mp --num-workers 8 --device mps --tui --log-every 50 --ensembles 3 --hidden 128 --depth 2 --dropout 0.1
```

NeuralLinearTS (TUI, MP):

```
uv run mab-gpu-puffer --env contextual --algo neurallinear --k 10 --d 8 --T 600 --runs 256 --vector mp --num-workers 8 --device mps --tui --log-every 50 --features 32 --linlam 1.0 --linv 0.1
```

### Adversarial‑Style (Bernoulli)

```
uv run mab-gpu-advanced --env bernoulli --algo exp3 --k 10 --T 1000 --runs 1024 --device mps --num-workers 8 --log-every 200 --save-csv
uv run mab-gpu-advanced --env bernoulli --algo exp3ix --k 10 --T 1000 --runs 1024 --device mps --num-workers 8 --log-every 200 --save-csv
```

### Classic With Plots/CSV

```
uv run mab-gpu-runner --algo klucb --k 10 --T 2000 --runs 1024 --device mps --num-workers 8 --log-every 200 --save-csv
uv run mab-gpu-advanced --env contextual --algo linucb --k 10 --d 8 --T 1000 --runs 1024 --device mps --num-workers 8 --log-every 200 --save-csv
uv run mab-gpu-advanced --env contextual --algo lints --k 10 --d 8 --T 1000 --runs 1024 --device mps --num-workers 8 --log-every 200 --save-csv
```

## Sweep Matrix (Scripted)

Use the sweep script to run a matrix of devices, seeds, and categories (smoke/throughput/neural). Logs are saved per run.

```
# Default (device=mps, seed=0, run all categories)
bash scripts/run_all.sh

# Device matrix and multiple seeds
DEVICES="cpu mps" SEEDS="0 1" bash scripts/run_all.sh

# Neural-only on CUDA
DO_SMOKE=0 DO_THROUGHPUT=0 DO_NEURAL=1 bash scripts/run_all.sh cuda

# Tune LinTS v values
VS="0.05 0.1 0.2" bash scripts/run_all.sh mps
```

Hyperparameters can be overridden via environment variables:

```
ALPHAS, VS, GAMMAS_EXP3, GAMMAS_EXP3IX, DUCB_C, DUCB_DISCOUNT, SW_C, SW_WINDOW,
NEURAL_FEATURES, NEURAL_ENSEMBLES, NEURAL_HIDDEN, NEURAL_DEPTH, NEURAL_DROPOUT
```

All logs go under `sweeps_logs/<timestamp>_<device>_s<seed>/` with one `.log` per run.

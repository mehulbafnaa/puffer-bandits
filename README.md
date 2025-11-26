# puffer-bandits

Elegant, scalable bandit research toolkit built on PufferLib and uv. Environments run on CPU (vectorized); agents compute on your chosen `--device` (`cpu|mps|cuda`).

## Highlights
- Nine algorithms across four families
  - Non‑contextual: KLUCB, DiscountedUCB, SlidingWindowUCB
  - Contextual: LinUCB, LinTS
  - Adversarial: EXP3, EXP3‑IX
  - Neural: NeuralTS, NeuralLinearTS
- Three runners for different workflows
  - Classic: non‑contextual with optional CSV/plots
  - Advanced: contextual/adversarial/neural (vectorized)
  - Native: PufferLib environments with an optional Rich TUI
- Single‑file dataset playback (`.npz`) for contextual experiments
- Repro‑friendly: uv CLIs, seeds, scripted sweeps

## Install
- Requirements: Python 3.9+, `uv` (`pipx install uv`)
- Create an environment and install dependencies:
  - `uv venv && uv sync`
- Install PyTorch (choose one):
  - CUDA: `uv pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio`
  - CPU:  `uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio`
  - MPS:  `uv pip install torch torchvision torchaudio`

## Quick Start
- Native contextual (MPS, MP, Rich TUI):
  - `uv run --active puffer-bandits-native --env contextual --algo linucb --k 10 --d 8 --T 1000 --runs 1024 --vector mp --num-workers 8 --device mps --tui --log-every 100`
- Advanced contextual (LinTS, CSV/plots optional):
  - `uv run --active puffer-bandits-advanced --env contextual --algo lints --k 10 --d 8 --T 1000 --runs 1024 --device mps --num-workers 8 --log-every 100`
- Classic non‑contextual (KL‑UCB):
  - `uv run --active puffer-bandits-runner --algo klucb --k 10 --T 2000 --runs 1024 --device cpu --num-workers 1 --log-every 200 --save-csv`

## Runners & Flags
- `puffer-bandits-runner` (non‑contextual)
  - `--algo {klucb, ducb, swucb}`
- `puffer-bandits-advanced` (contextual/adversarial/neural)
  - `--algo {linucb, lints, exp3, exp3ix, neuralts, neurallinear}`
- `puffer-bandits-native` (PufferLib envs + TUI)
  - `--env {contextual, bernoulli, dataset}`
  - `--vector {mp, serial}`, `--num-workers N`, `--tui`
  - `--device {cpu, mps, cuda}`

## Dataset Mode (.npz)
- Use `--env dataset --data-path path/to/data.npz`
- File schema:
  - `X`: `(N, k, d)` contexts (required)
  - `P`: `(N, k)` probabilities OR `Y`: `(N, k)` binary outcomes
- Keys configurable with `--x-key/--p-key/--y-key`

## Rich TUI
- Enable `--tui` on the native runner for live KPIs, action histograms, and performance
- If Matplotlib complains about a non‑writable home dir, prefix with `MPLCONFIGDIR=.matplotlib_cache`

## Sweeps
- Turnkey profiles (smoke/throughput/neural):
  - `bash scripts/run_all.sh mps`
- Knobs: `DO_SMOKE=1|0 DO_THROUGHPUT=1|0 DO_NEURAL=1|0 WORKERS K D` and per‑algo ranges (`ALPHAS`, `VS`, `GAMMAS_*`, `SW_WINDOW`, etc.)
- Logs under `sweeps_logs/`; CSV/plots saved in `plots_gpu/` when enabled

## Tips
- Target the project venv: `uv run --active …`
- Ensure `--runs` is divisible by `--num-workers` (or set `--num-workers 1`)
- Neural methods are heavier; use fewer runs (e.g., 256–512) or fewer workers
- For small contextual workloads, CPU can be faster than MPS; use `--force-device` to override heuristics

## Layout
- Code: `puffer_bandits/{agents.py, agents_ctx/, core/, utils/, runner_*.py, puffer_envs.py, ui/tui.py}`
- Scripts: `scripts/run_all.sh`, catalog `runs.txt`
- Sample data: `data/synth_ctx.npz` (gitignored)

## License
- See `LICENSE`


# puffer-bandits

Status: In active development (alpha). Interfaces and defaults may change. Expect rapid iteration and occasional breaking changes. If you hit issues, please open an issue or PR.

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
- Requirements: Python 3.12+, `uv` (`pipx install uv`)
- Create an environment and install dependencies:
  - `uv venv && uv sync`
- Install PyTorch (choose one):
  - CUDA: `uv pip install --index-url https://download.pytorch.org/whl/cu121 torch`
  - CPU:  `uv pip install --index-url https://download.pytorch.org/whl/cpu torch`
  - MPS:  `uv pip install torch`
- Optional: plotting extras
  - `uv pip install 'puffer-bandits[plots]'`

## Quick Start
- Minimal, config-first (recommended):
  - `uv run --active puffer-bandits --config configs/ctx_linucb.yaml`
  - Override on the fly: `uv run --active puffer-bandits --config configs/ctx_linucb.yaml --set runs=2048 --set tui=false`
- Native contextual (explicit):
  - `uv run --active puffer-bandits-native --env contextual --algo linucb --k 10 --d 8 --T 1000 --runs 1024 --vector mp --num-workers 8 --device mps --tui --log-every 100`
- Advanced contextual (LinTS):
  - `uv run --active puffer-bandits-advanced --env contextual --algo lints --k 10 --d 8 --T 1000 --runs 1024 --device mps --num-workers 8 --log-every 100`
- Classic non‑contextual (KL‑UCB):
  - `uv run --active puffer-bandits-runner --algo klucb --k 10 --T 2000 --runs 1024 --device cpu --num-workers 1 --log-every 200 --save-csv`

## Runners
- `puffer-bandits` (config-first, minimal CLI):
  - Required: `--config path.yaml`
  - Optional: `--set key=val` (repeatable), `--runner {native,advanced,classic}` (default: `native`), `--preset` (native only)
- `puffer-bandits-native` (full CLI) — PufferLib envs + TUI
- `puffer-bandits-advanced` (full CLI) — contextual/adversarial/neural
- `puffer-bandits-runner` (full CLI) — classic non‑contextual

See config keys and examples in `docs/CONFIG.md`.

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
- Logs under `sweeps_logs/`; CSV/plots saved in `plots/` (formerly `plots_gpu/`) when enabled

## Logging to Weights & Biases (optional)
- Enable with `--wandb` on any runner. Recommended extras:
  - `--wandb-project puffer-bandits` `--wandb-tags mps,linucb` `--run-name my-run`
  - Offline mode: `--wandb-offline` (or set `WANDB_MODE=offline`)
- Examples:
  - Minimal (config-first): `uv run --active puffer-bandits --config configs/ctx_linucb.yaml --set wandb=true --set wandb_project=puffer-bandits`
  - Native: `uv run --active puffer-bandits-native --env contextual --algo linucb ... --wandb --wandb-project puffer-bandits`
  - Advanced: `uv run puffer-bandits-advanced --env bernoulli --algo exp3 ... --wandb`
  - Classic: `uv run puffer-bandits-runner --algo klucb ... --wandb --save-csv`
- Metrics logged per `--log-every`: mean_reward, %_optimal (when available), cumulative_regret, and simple perf stats.

## Tips
- Target the project venv: `uv run --active …`
- Ensure `--runs` is divisible by `--num-workers` (or set `--num-workers 1`)
- Neural methods are heavier; use fewer runs (e.g., 256–512) or fewer workers
- On small contextual workloads, CPU can outperform MPS; use `--force-device` to override heuristics

### macOS TLS note (pufferlib build)
If `uv sync` fails building `pufferlib` with a TLS error like `SSL: CERTIFICATE_VERIFY_FAILED` on Python 3.12 (macOS), try one of:
- Use Homebrew Python: `brew install python@3.12` then `uv venv --python $(brew --prefix)/bin/python3.12 && uv sync`
- Or run `Install Certificates.command` from your Python 3.12 framework (typically under `/Applications/Python 3.12/`).

## Layout
- Code: `puffer_bandits/{agents.py, agents_ctx/, core/, utils/, runner_*.py, puffer_envs.py, ui/tui.py}`
- Scripts: `scripts/run_all.sh`, catalog `runs.txt`
- Sample data: `data/synth_ctx.npz` (gitignored)

## Development
- Code layout: `puffer_bandits/{agents.py, agents_ctx/, core/, utils/, runner_*.py, puffer_envs.py, ui/tui.py}`
- Tests (optional): `uv run --active pytest -q`
- Contributing: PRs welcome. Please include a small repro or smoke command.

## License & Acknowledgements
- License: see `LICENSE`
- Thanks: PufferLib, PyTorch, Gymnasium, Rich, and uv

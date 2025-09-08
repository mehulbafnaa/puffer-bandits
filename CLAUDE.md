# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

MAB_GPU is a high-performance multi-armed bandit implementation using PyTorch and PufferLib for GPU acceleration. It provides both classical bandit algorithms and advanced contextual/neural variants with vectorized environments.

## Key Commands

### Development Setup
```bash
# Install dependencies and setup environment
uv venv
uv sync

# Install PyTorch (choose based on platform)
# CUDA: uv pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
# CPU: uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
# Apple Silicon: uv pip install torch torchvision torchaudio

# Install PufferLib dependency
uv pip install -e ../PufferLib
```

### Testing and Quality
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_agents_basics.py

# Linting and type checking
ruff check .
mypy .
```

### Running Experiments

The package provides three main entry points via `uv run`:

**Basic Multi-Armed Bandits:**
```bash
uv run mab-gpu-runner --algo klucb --k 10 --T 100000 --runs 8192 --device cuda
```

**Advanced Contextual/Adversarial:**
```bash
uv run mab-gpu-advanced --env contextual --algo linucb --k 10 --d 8 --T 5000 --runs 4096 --device cuda
```

**Native PufferLib with TUI:**
```bash
uv run mab-gpu-puffer --env contextual --algo linucb --k 10 --d 8 --T 600 --runs 256 --vector serial --device mps --tui
```

### Automated Sweeps
```bash
# Run comprehensive algorithm sweep
bash scripts/run_all.sh [device]

# Generate sweep comparison plots
uv run python MAB_GPU/plot_sweeps.py --file plots_gpu/summary.csv --algo klucb --param kl_alpha --out plots_gpu/sweep_klucb_alpha.png
```

## Architecture

### Core Design Pattern: Functional Core, OO Shell
- `core/`: Pure tensor functions for mathematical operations (KL-UCB bisection, EXP3 updates, Sherman-Morrison, etc.)
- `agents.py` & `advanced_agents.py`: Object-oriented agent classes that compose core functions
- `agents_ctx/`: Modular contextual agent implementations

### Agent Hierarchy
- **Agent/CtxAgent**: Base classes with standard interface (`reset()`, `select_actions()`, `update()`)
- **Classical**: KLUCB, DUCB (discounted), SWUCB (sliding window)
- **Contextual**: LinUCB, LinTS (linear Thompson sampling)
- **Adversarial**: EXP3, EXP3-IX
- **Neural**: NeuralTS (ensemble), NeuralLinearTS (hybrid)

### Environment Types
- **Bernoulli**: Standard k-armed bandit with Bernoulli rewards
- **Contextual**: Linear contextual bandit with logistic-Bernoulli rewards and feature vectors
- **Non-stationary**: Time-varying reward distributions

### Runners and Integration
- `runner_puffer.py`: Classical bandits with PufferLib vectorization
- `runner_puffer_advanced.py`: Contextual/adversarial algorithms
- `runner_puffer_native.py`: Native PufferLib environments with Rich TUI
- All runners support CPU/CUDA/MPS devices and multiprocessing

### Key Dependencies
- **PyTorch**: Core tensor operations and GPU acceleration
- **PufferLib**: Vectorized environment execution
- **uv**: Package management and script execution
- **Rich**: Terminal UI for interactive runs

### Configuration
- Device selection: `--device cpu|cuda|mps`
- Vectorization: `--vector serial|mp` with `--num-workers`
- Output: `--save-csv` for data export, `--tui` for interactive mode
- All hyperparameters configurable via CLI flags

## Development Notes

### Testing Strategy
Tests cover algorithm correctness, numerical stability, device compatibility, and integration smoke tests. Use pytest for running individual test files or the full suite.

### Script Execution
Always use `uv run` with the provided script aliases. Direct Python execution is not supported due to the uv-based project structure.

### Plotting and Analysis
Results automatically generate confidence intervals and export to CSV. Plotting utilities support parameter sweeps and algorithm comparisons.
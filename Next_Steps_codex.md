# Next Steps (Codex) — MAB_GPU Refactor and Hardening Plan

Date: 2025‑09‑07
Author: Codex

This plan validates the external review (Code_review_claude.md), notes where it is accurate vs. overstated, and enumerates concrete, prioritized actions to bring the codebase to a robust, production‑ready state. Items are grouped by priority to enable incremental, verifiable progress.

## Validation Summary

- Contextual env logits logic: Partly valid. The current code computes per‑arm logits correctly via a fallback einsum. The conditional branch is unnecessary and confusing; simplify to an explicit per‑arm dot (einsum) with assertions.
- Cholesky stability: Valid. We add jitter, but do not catch failures; wrap with `cholesky_ex`/fallback and instrument failure counts.
- Test coverage: Valid. Core ops covered; contextual/NN agents and integration paths need tests.
- Memory management: Partly valid. Most loops run under `no_grad()` with bounded state; leaks unlikely, but we can add optional periodic cache clears and profiling hooks.
- Input validation: Valid. Add action bounds in envs; add tensor/device/shape checks in agents.
- Performance anti‑patterns: Partly valid. Some `.to(device)` occur in hot loops by design (obs comes from NumPy each step). We can reduce transfers with preallocated buffers and non‑blocking H2D copies.
- Numerical stability: Partly valid. We clamp and add eps in several places; formalize constants and guard divisions.
- Style/architecture/docs/deps/CI: Valid areas to improve.

## Immediate (Blockers)

1) Simplify contextual env logits and validate shapes
   - File: `contextual_env.py`
   - Replace conditional/diagonal path with: `logits = np.einsum("ad,ad->a", X, self.theta)`
   - Add asserts: `assert X.shape == (k, d)` and `assert theta.shape == (k, d)`
   - Add unit test: numerical match to explicit loop and to original behavior

2) Harden Cholesky usage in agents
   - Files: `advanced_agents.py`
   - LinTS / NeuralLinearTS: wrap `torch.linalg.cholesky` with `torch.linalg.cholesky_ex` or fallback to `eigvalsh`/`clamp_min(0)` PSD repair; count/log fallbacks
   - Add tests that force near‑singular A_inv and verify path returns finite scores

3) Input validation and safe env stepping
   - Files: `bandit_env.py`, `contextual_env.py`
   - Enforce action bounds (`0 <= action < k`), raise `AssertionError` or `ValueError`
   - Contextual env: validate non‑stationary drift magnitudes; document units
   - Add tests for out‑of‑range actions

## High Priority (Next Sprint)

4) Expand tests to 80%+ on critical paths
   - Add CPU smoke/integration tests for:
     - LinUCB/LinTS one‑step correctness on synthetic linear models
     - NeuralTS/NeuralLinearTS shapes and monotonic learning on tiny problems
     - EXP3/EXP3‑IX probability invariants and update behavior
     - Native runner (Serial/MP) small runs to catch info plumbing
   - Add numerical stability tests (epsilons, clamping, division safety)

5) Optional memory management hooks and profiling
   - Add CLI flags: `--empty-cache-interval N`, `--gc-interval N` (CUDA only)
   - In runners, every N steps: `torch.cuda.empty_cache()` and `gc.collect()`
   - Document tradeoffs (usually not needed; useful for long runs/benchmarks)

6) Device transfer optimization (hot loop)
   - Preallocate torch tensors for obs and rewards; use `copy_`/`set_` instead of new allocations
   - Use pinned memory and `to(device, non_blocking=True)` when feasible
   - Cache constant tensors (e.g., identity/eps_eye) on device

7) Numerical constants and guards
   - Centralize epsilons in `utils` (e.g., `EPS=1e-6`, `EPS_KL=1e-12`)
   - Replace literal eps in `core/*` and agents with named constants
   - Guard all divisions with `clamp_min`; add comments/rationale

## Medium Priority (Following Release)

8) Error handling consistency
   - Wrap risky tensor ops (Cholesky, linear solves) and surface actionable messages
   - Add device checks and friendly messages when MPS/CUDA unavailable
   - Standardize `assert` → `ValueError`/`RuntimeError` with context

9) Code style and lint hygiene
   - Enable `ruff` (already configured) in CI; fix remaining violations
   - Keep line length ≤100, consistent naming, remove semicolons

10) Architecture refactor (incremental)
   - Split `advanced_agents.py` into modules: `lin.py`, `adversarial.py`, `neural.py`
   - Introduce `agents/base.py` with shared mixins (device, rng, shape checks)
   - Separate runner utilities (profiling, memory, TUI plumbing) into `utils/runner.py` and `ui/tui.py`

## Low Priority (Future)

11) Documentation & tutorials
   - API reference for agents/runners/envs
   - Hyperparameter guidelines (alpha, v, lam, window, discount)
   - PufferLib vectorization primer (Serial vs MP)

12) Dependencies & CI
   - Pin minimal compatible versions for PyTorch, Gymnasium, PufferLib
   - Add GitHub Actions: `uv sync`, `pytest`, `ruff`, (optional) small demo run on CPU

13) Benchmarking & monitoring
   - Add microbenchmarks for selection/update ms/step across devices
   - Optional WandB hooks for longer experiments (behind a flag)

## Concrete Task List (Checklist)

- [ ] Contextual env: replace logits path with single einsum + tests
- [ ] Agents: wrap Cholesky with `cholesky_ex` + fallback; add tests
- [ ] Env action validation (Bernoulli/Contextual) + tests
- [ ] Add LinUCB/LinTS correctness tests (tiny synthetic), EXP3 invariants, NeuralTS/NeuralLinearTS shape tests
- [ ] Runner hooks: `--empty-cache-interval/--gc-interval`; document
- [ ] Preallocation + non‑blocking H2D; cache constants in runners/agents
- [ ] Centralize EPS constants and adopt across code
- [ ] Standardize error handling for risky ops; propagate messages
- [ ] Refactor `advanced_agents.py` into smaller modules
- [ ] CI: add GitHub Actions (uv sync + pytest + ruff)
- [ ] Docs: API + tuning guidance + PufferLib usage guide

## Notes on Already Addressed Items

- KL‑UCB bisection feasibility fixed (return feasible bound)
- Deterministic RNG for tie‑breakers across agents
- Contextual resets broadcasting fixed
- Sherman‑Morrison batch outer product corrected
- Robust TUI metrics (ms/step, regret attribution, value panels)
- Removed unused GUI code and consolidated around TUI

## Acceptance Criteria

The Immediate and High‑Priority items are considered done when:
- All new tests pass locally and in CI; coverage ≥80% on agents/core
- Contextual env logits simplified with assertions; no shape‑related fallbacks
- Cholesky failures are handled without crashes in LinTS/NeuralLinearTS
- Env action bounds enforced; tests confirm exceptions
- Runners exhibit no regression in ms/step; optional cache flags work


# MAB_GPU Improvement Plan (Codex Reference)

Purpose: A precise, actionable roadmap to implement the improvements discussed in the code review. It includes priorities, file references, concrete steps, acceptance criteria, and small patch sketches to accelerate changes.

---

## Overview

Focus areas
- Reproducibility and seeding across agents and runners
- Correctness parity between Gym and native Puffer envs (contextual logits)
- Runner efficiency (buffer preallocation, device sync)
- Shape validation for clearer errors
- Minor dtype and repo hygiene fixes
- Complementary tests and perf checks

---

## High Priority

### 1) Deterministic Seeding for Agents

Problem: Agents create a fresh `torch.Generator` but don’t seed it from user config, making tie-breaking and stochastic policies non-reproducible across runs.

Targets
- Add optional `seed: int | None` to configs and plumb it through runners.
- Seed `torch.Generator` used by agents with this seed.

Files
- Configs and agents
  - MAB_GPU/agents.py:24
  - MAB_GPU/agents_ctx/base.py:16
- Runners (thread this seed)
  - MAB_GPU/runner_puffer.py:31
  - MAB_GPU/runner_puffer_advanced.py:31
  - MAB_GPU/runner_puffer_native.py:26

Steps
- Extend `AgentCfg`/`CtxAgentCfg` with `seed: int | None = None`.
- In agent `__init__`, set `self.rng = torch.Generator(device=self.device)`; if `cfg.seed is not None`: `self.rng.manual_seed(cfg.seed)`.
- Runners: add CLI flag `--agent-seed` (default: None). If None, reuse `--seed`. Pass to `AgentCfg`/`CtxAgentCfg`.

Acceptance Criteria
- Two agents constructed with same seed take identical actions under tie conditions (extend test below).
- All existing tests continue to pass.

Sketch
```python
# agents.py:24, agents_ctx/base.py:16
@dataclass
class AgentCfg:
    k: int
    num_envs: int
    device: torch.device
    seed: int | None = None

# in Agent.__init__ and CtxAgent.__init__
self.rng = rng or torch.Generator(device=self.device)
if getattr(cfg, "seed", None) is not None:
    self.rng.manual_seed(int(cfg.seed))

# runners parse_args
p.add_argument("--agent-seed", type=int, default=None)
# build_agent: seed_arg = cfg.agent_seed if cfg.agent_seed is not None else cfg.seed
AgentCfg(..., seed=seed_arg)
```

Tests
- Extend `MAB_GPU/tests/test_agents_basics.py` with a test verifying deterministic selection for same seeds.

---

### 2) Native Puffer Contextual Env: Fix Logits

Problem: Puffer native contextual env uses a diagonal-with-fallback pattern for logits. Gym env uses explicit einsum. Make them consistent and unambiguous.

Files
- MAB_GPU/puffer_envs.py:127

Steps
- Replace current branch with explicit `np.einsum("ad,ad->a", X, self.theta)` (as in `MAB_GPU/contextual_env.py:102`).

Acceptance Criteria
- A dedicated test asserts equality of logits/probabilities between Gym and Puffer envs for the same `X` and `theta`.

Sketch
```python
# puffer_envs.py:127-133
X = self._obs
logits = np.einsum("ad,ad->a", X, self.theta)
```

Tests
- New: `test_puffer_contextual_logits_consistency()` comparing `ContextualBanditEnv` vs `PufferContextualBandit` for fixed `theta`/`obs`.

---

## Medium Priority

### 3) Classic Runner: Preallocate Device Buffers and Use Sync Helper

Problem: `runner_puffer.py` re-creates tensors each step; advanced/native runners preallocate.

Files
- MAB_GPU/runner_puffer.py:170
- MAB_GPU/utils/device.py:26

Steps
- Preallocate `rewards_t = torch.empty((runs,), device=device, dtype=torch.float32)` once.
- Replace per-step `torch.from_numpy(r)...` with `rewards_t.copy_(...)`.
- Use `utils.device.sync_device(device)` when profiling for consistent timing.

Acceptance Criteria
- Functionally identical results; profile shows fewer allocations in the hot loop.

Sketch
```python
# before loop
overlay = rewards_t = torch.empty((cfg.runs,), device=device, dtype=torch.float32)
# in loop
rewards_t.copy_(torch.from_numpy(r.astype(np.float32)).to(device=device))
```

---

### 4) Shape Assertions in Agent APIs

Problem: Silent broadcasting can hide mis-wired integration.

Files
- MAB_GPU/agents.py:80
- MAB_GPU/agents_ctx/lin.py:71
- MAB_GPU/agents_ctx/adversarial.py:68

Steps
- Add `assert actions.shape == (num_envs,)` and `assert rewards.shape == (num_envs,)`.
- Contextual: `assert obs.shape == (num_envs, k, d)` in `select_actions`/`update`.

Acceptance Criteria
- Misuse fails fast with clear error; existing tests pass.

Sketch
```python
assert actions.dim() == 1 and actions.shape[0] == self.num_envs
assert rewards.dim() == 1 and rewards.shape[0] == self.num_envs
# contextual
assert obs.shape == (self.num_envs, self.k, self.d)
```

---

### 5) Counter Dtype (Optional)

Context: `KLUCB.N` uses `int32`. For very long horizons this could overflow.

Files
- MAB_GPU/agents.py:55

Change
- Switch to `torch.int64` and clamp at `torch.iinfo(dtype).max - margin` if needed.

Acceptance Criteria
- No behavior change for typical horizons; tests pass.

---

## Low Priority

### 6) Repo Hygiene: Remove Symlinked `resources`

Problem: `MAB_GPU/resources` symlink points into `.venv`, which shouldn’t be packaged.

Action
- Remove the symlink from source control and ensure `.gitignore` excludes it.

Files
- MAB_GPU/.gitignore: add `resources`

---

### 7) Logging and Diagnostics (Optional)

- Replace print-based logs in runners with `logging` gated by verbosity.
- Continue supporting `--profile` and integrate `memory_stats()` consistently.

---

## Tests to Add

1) Deterministic RNG (non-contextual)
- Path: `MAB_GPU/tests/test_agents_basics.py`
- Two agents with same `cfg.seed` pick same arms under tie.

2) Native Puffer Contextual Logits
- New test: `test_puffer_contextual_logits_consistency`
- Arrange fixed `theta` and `obs` in both envs; compare `p` vectors.

3) Shape Assertions
- Negative tests: pass wrong-shaped `obs`/`actions`/`rewards` and assert AssertionError.

4) Neural smoke (optional)
- Tiny runs for `NeuralTS` and `NeuralLinearTS` to ensure no runtime errors.

---

## Performance Validation

- Use `--profile` and print ms/step in both classic and advanced runners.
- Track select/actions_to_cpu/env_step/r2dev/update; confirm equal or reduced `r2dev` after preallocation.

---

## Rollout Plan

1) Implement high priority items, run unit tests.
2) Implement medium items, re-run tests + quick smoke (`mab-gpu-puffer` serial + mp).
3) Land low-priority cleanup.
4) Optional: add a CI job with `pytest -q` and a couple of `uv run` smoke invocations on cpu.

---

## Mini Patch Queue (for quick application)

1) Puffer contextual logits
- File: `MAB_GPU/puffer_envs.py:127`
```diff
-        logits = (X @ self.theta.T).diagonal() if X.shape[0] == self.theta.shape[0] else (X @ self.theta.T).diagonal()
-        if logits.ndim != 1:
-            logits = np.einsum("ad,ad->a", X, self.theta)
+        logits = np.einsum("ad,ad->a", X, self.theta)
```

2) Agent seeding (interface only; see full steps above)
- Files: `MAB_GPU/agents.py:17`, `MAB_GPU/agents_ctx/base.py:13`
```diff
 @dataclass
 class AgentCfg:
-    k: int
-    num_envs: int
-    device: torch.device
+    k: int
+    num_envs: int
+    device: torch.device
+    seed: int | None = None
```

3) Classic runner rewards buffer (snippet)
- File: `MAB_GPU/runner_puffer.py`
```diff
-            rewards = torch.from_numpy(r).to(device=device, dtype=torch.float32)
+            rewards = rewards_t
+            rewards.copy_(torch.from_numpy(r.astype(np.float32)).to(device=device))
```

---

## Notes

- Keep numerical safety clamps: `clamp_min(1e-12)`, jitter in Cholesky, and tiny tie-breakers.
- Avoid calling `torch.cuda.empty_cache()` in tight loops; reserve for teardown/debug scripts.
- Maintain device discipline: create tensors on target device, use `.copy_` instead of reallocation in hot paths.

---

## Checklist

- [ ] Agent configs accept `seed` and seed generators
- [ ] Runners plumb `--agent-seed` (fallback to `--seed`)
- [ ] Puffer contextual logits fixed (einsum)
- [ ] Classic runner preallocates `rewards_t` and uses `sync_device`
- [ ] Shape assertions added in agent APIs
- [ ] Optional: `KLUCB.N` -> int64
- [ ] Tests added for RNG determinism, logits consistency, shape assertions
- [ ] Remove symlinked `resources` from repo; ignore going forward

This document is designed for quick implementation. Each item includes the minimum patch context and acceptance criteria to enable safe, incremental merges.

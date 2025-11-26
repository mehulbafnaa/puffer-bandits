from __future__ import annotations

import numpy as np

from puffer_bandits.bandit_env import BernoulliBanditEnv
from puffer_bandits.contextual_env import ContextualBanditEnv


def test_bernoulli_action_bounds():
    env = BernoulliBanditEnv(k=5)
    env.reset(seed=0)
    try:
        env.step(-1)
        assert False, "expected ValueError for negative action"
    except ValueError:
        pass
    try:
        env.step(5)
        assert False, "expected ValueError for out-of-range action"
    except ValueError:
        pass


def test_contextual_logits_and_bounds():
    k, d = 4, 3
    env = ContextualBanditEnv(k=k, d=d)
    obs, _ = env.reset(seed=0)
    # Overwrite theta and obs with known values
    env.theta = np.arange(k * d, dtype=np.float32).reshape(k, d) * 0.01
    env._obs = np.arange(k * d, dtype=np.float32).reshape(k, d) * 0.02
    # Compute expected probs via einsum
    logits = np.einsum("ad,ad->a", env._obs, env.theta)
    p_exp = 1.0 / (1.0 + np.exp(-logits))
    _, r, _, _, info = env.step(0)
    p_vec = info["p"]
    assert np.allclose(p_vec, p_exp, atol=1e-6)
    # Bounds check
    try:
        env.step(k)
        assert False, "expected ValueError for out-of-range action"
    except ValueError:
        pass

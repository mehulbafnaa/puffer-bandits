from __future__ import annotations

"""Contextual bandit Gymnasium environment for vectorized experiments.

Observation: matrix of shape (k, d) containing per-arm context features x_{a,t}.
Reward model (default): Bernoulli with logistic link, p_{a,t} = sigmoid(theta_a^T x_{a,t}).
Non-stationary mode: random walk on per-arm parameters theta_a.

Returned info per step includes:
- "p": vector of arm success probabilities at decision time (length k)
- "optimal": 1 if chosen action was optimal at decision time, else 0

This pairs with GPU agents that use the observation to compute scores.
"""

from typing import Any
try:
    import matplotlib.pyplot as plt  # optional for render
except Exception:  # pragma: no cover
    plt = None  # type: ignore

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover - import guard for skeleton
    gym = None  # type: ignore
    spaces = None  # type: ignore


BaseEnv = gym.Env if gym is not None else object  # type: ignore


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class ContextualBanditEnv(BaseEnv):
    metadata = {"render_modes": []}

    def __init__(
        self,
        k: int = 10,
        d: int = 8,
        nonstationary: bool = False,
        theta_sigma: float = 0.05,
        x_sigma: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        if spaces is None:
            raise ImportError(
                "Gymnasium is required for the environment. Install via `pip install gymnasium`."
            )
        self.k = int(k)
        self.d = int(d)
        self.nonstationary = bool(nonstationary)
        self.theta_sigma = float(theta_sigma)
        self.x_sigma = float(x_sigma)
        self._rng: np.random.Generator = rng or np.random.default_rng()

        # Observations are per-arm context matrices (k, d)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.k, self.d), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.k)

        # Parameters and cached observation
        self.theta: np.ndarray | None = None  # (k, d)
        self._obs: np.ndarray | None = None   # (k, d)
        self._last_action: int | None = None
        self._last_p: np.ndarray | None = None

    def _sample_theta(self) -> np.ndarray:
        # Initialize small random parameters per arm
        return self._rng.normal(0.0, 1.0, size=(self.k, self.d)).astype(np.float32)

    def _sample_obs(self) -> np.ndarray:
        return self._rng.normal(0.0, self.x_sigma, size=(self.k, self.d)).astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.theta = self._sample_theta()
        self._obs = self._sample_obs()
        return self._obs.copy(), {}

    def step(self, action: int) -> tuple[np.ndarray, int, bool, bool, dict[str, Any]]:
        assert self.theta is not None and self._obs is not None, "reset() must be called first"

        # Non-stationary drift on theta before generating reward? Use decision-time theta.
        # Here we keep theta fixed for decision, then optionally drift for future.
        X = self._obs  # (k, d) contexts at decision time
        # Validate shapes explicitly to avoid ambiguous broadcasting paths
        assert X.shape == (self.k, self.d), f"obs shape {X.shape} != (k,d)"
        assert self.theta.shape == (self.k, self.d), f"theta shape {self.theta.shape} != (k,d)"
        # Compute per-arm logits with a clear einsum (x_a^T theta_a)
        logits = np.einsum("ad,ad->a", X, self.theta)
        p_vec = _sigmoid(logits)
        a = int(action)
        if a < 0 or a >= self.k:
            raise ValueError(f"action out of bounds: {a} not in [0,{self.k-1}]")
        p_a = float(p_vec[a])
        r = 1 if self._rng.random() < p_a else 0
        optimal = int(a == int(np.argmax(p_vec)))
        info: dict[str, Any] = {"p": p_vec.copy(), "optimal": optimal}
        self._last_action = a
        self._last_p = p_vec.copy()

        # Evolve theta if non-stationary (random walk)
        if self.nonstationary:
            self.theta = self.theta + self._rng.normal(0.0, self.theta_sigma, size=self.theta.shape).astype(np.float32)

        # Sample next observation
        self._obs = self._sample_obs()
        return self._obs.copy(), r, False, False, info

    def render(self) -> None:  # pragma: no cover - visual aid for demos
        if plt is None or self._last_p is None:
            return None
        p_vec = self._last_p
        k = self.k
        x = np.arange(k)
        plt.clf()
        plt.bar(x, p_vec, color="tab:blue", alpha=0.7)
        plt.ylim(0.0, 1.0)
        plt.xlabel("arm"); plt.ylabel("p(x)")
        if self._last_action is not None:
            la = self._last_action
            plt.bar([la], [p_vec[la]], color="tab:orange", alpha=0.9, label="chosen")
            plt.legend()
        plt.title("Contextual Bandit (current step probabilities)")
        plt.pause(0.001)
        return None

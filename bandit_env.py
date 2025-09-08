from __future__ import annotations

"""Minimal Gymnasium environment for a k-armed Bernoulli bandit.

This environment is intentionally simple to standardize evaluation in the
assignment. It exposes a discrete action space of size ``k`` and returns a
constant dummy observation of ``0`` at each step. Rewards are Bernoulli with
per-arm probabilities ``p_a`` that are sampled at reset (stationary case) or
evolve via a Gaussian random walk (non-stationary bonus case).
"""

from typing import Any
try:
    import matplotlib.pyplot as plt  # optional
except Exception:  # pragma: no cover - optional dependency
    plt = None  # type: ignore

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover - import guard for skeleton
    gym = None  # type: ignore
    spaces = None  # type: ignore


BaseEnv = gym.Env if gym is not None else object  # type: ignore


class BernoulliBanditEnv(BaseEnv):
    """Gymnasium environment for a k-armed Bernoulli bandit.

    Design targets (match assignment):
    - Action space: ``Discrete(k)``
    - Observation space: ``Discrete(1)``, return constant ``0``
    - Rewards (stationary): Bernoulli per arm with fixed ``p_a`` in ``[0,1]``
    - Rewards (non-stationary, bonus): random walk on probabilities
    - Continuing task: ``terminated=False``, ``truncated=False`` every step
    """

    metadata = {"render_modes": []}

    def __init__(self, k: int = 10, nonstationary: bool = False, sigma: float = 0.1,
                 rng: np.random.Generator | None = None) -> None:
        """Initialize a bandit with ``k`` arms.

        Args:
            k: Number of arms (size of action space).
            nonstationary: If ``True``, apply a Gaussian random walk to ``p`` each step.
            sigma: Standard deviation of random-walk noise per step (non-stationary only).
            rng: Optional numpy random generator for reproducibility.
        """
        if spaces is None:
            raise ImportError(
                "Gymnasium is required for the environment. Install via `pip install gymnasium`."
            )
        self.k = int(k)
        self.nonstationary = bool(nonstationary)
        self.sigma = float(sigma)
        self.action_space = spaces.Discrete(self.k)
        self.observation_space = spaces.Discrete(1)
        self._rng: np.random.Generator = rng or np.random.default_rng()
        self.p: np.ndarray | None = None  # true Bernoulli means per arm
        self._last_action: int | None = None
        self._last_reward: int | None = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
        """Reset the bandit instance and sample arm probabilities.

        Args:
            seed: Optional seed to re-initialize the RNG for this reset.
            options: Unused options dict (Gymnasium compatibility).

        Returns:
            A pair ``(obs, info)`` where ``obs`` is the constant integer ``0`` and
            ``info`` is an empty dict (students may add fields during experiments).
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # Sample p ~ Uniform(0,1)^k for stationary case
        self.p = self._rng.uniform(0.0, 1.0, size=self.k)
        return 0, {}

    def step(self, action: int) -> tuple[int, int, bool, bool, dict[str, Any]]:
        """Take an action and return the next transition.

        Args:
            action: Integer action index in ``[0, k-1]``.

        Returns:
            A tuple ``(obs, reward, terminated, truncated, info)`` where:
            - ``obs`` is the constant integer ``0``
            - ``reward`` is a Bernoulli sample in ``{0,1}`` for the chosen arm
            - ``terminated`` and ``truncated`` are ``False`` (continuing task)
            - ``info`` contains a copy of current probabilities under key ``"p"`` and
              an ``"optimal"`` flag (1 if ``action`` was optimal, else 0)
        """
        assert self.p is not None, "Environment must be reset before stepping"
        # Bounds check
        a = int(action)
        if a < 0 or a >= self.k:
            raise ValueError(f"action out of bounds: {a} not in [0,{self.k-1}]")
        # Evolve probabilities if non-stationary
        if self.nonstationary:
            noise = self._rng.normal(0.0, self.sigma, size=self.k)
            self.p = np.clip(self.p + noise, 0.0, 1.0)
        # Sample Bernoulli reward
        p_a = float(self.p[a])
        r = 1 if self._rng.random() < p_a else 0
        optimal = int(a == int(np.argmax(self.p)))
        self._last_action = a
        self._last_reward = r
        info: dict[str, Any] = {"p": self.p.copy(), "optimal": optimal}
        return 0, r, False, False, info

    def render(self) -> None:  # pragma: no cover - visual aid for demos
        if plt is None or self.p is None:
            return None
        k = self.k
        x = np.arange(k)
        plt.clf()
        plt.bar(x, self.p, color="tab:blue", alpha=0.7)
        plt.ylim(0.0, 1.0)
        plt.xlabel("arm"); plt.ylabel("true p")
        if self._last_action is not None:
            la = self._last_action
            plt.bar([la], [self.p[la]], color="tab:orange", alpha=0.9, label=f"chosen (r={self._last_reward})")
            plt.legend()
        plt.title("Bernoulli Bandit (true probabilities)")
        plt.pause(0.001)
        return None

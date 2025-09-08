from __future__ import annotations

import gymnasium as gym
import numpy as np
import pufferlib


class PufferBernoulliBandit(pufferlib.PufferEnv):
    """Native PufferLib environment for a k-armed Bernoulli bandit.

    Single-agent; observation is a dummy scalar 0. Action is Discrete(k).
    Returns info dict with current true probs and optimal flag.
    """

    def __init__(self, k: int = 10, nonstationary: bool = False, sigma: float = 0.1, buf=None, seed: int | None = None):
        self.k = int(k)
        self.nonstationary = bool(nonstationary)
        self.sigma = float(sigma)
        self.single_observation_space = gym.spaces.Box(low=0.0, high=0.0, shape=(1,), dtype=np.float32)
        self.single_action_space = gym.spaces.Discrete(self.k)
        self.num_agents = 1
        super().__init__(buf=buf)
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self.p: np.ndarray | None = None
        self._last_action: int | None = None
        self._last_reward: int | None = None

    def reset(self, seed: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.p = self._rng.uniform(0.0, 1.0, size=self.k).astype(np.float32)
        self.observations[0] = 0.0
        self.rewards[0] = 0.0
        self.terminals[0] = False
        self.truncations[0] = False
        info = dict(p=self.p.copy(), optimal=None)
        return self.observations, [info]

    def step(self, actions):
        assert self.p is not None
        a = int(actions[0]) if isinstance(actions, np.ndarray) else int(actions)
        if self.nonstationary:
            self.p = np.clip(self.p + self._rng.normal(0.0, self.sigma, size=self.k).astype(np.float32), 0.0, 1.0)
        p_a = float(self.p[a])
        r = 1 if self._rng.random() < p_a else 0
        self.observations[0] = 0.0
        self.rewards[0] = float(r)
        self.terminals[0] = False
        self.truncations[0] = False
        self._last_action = a
        self._last_reward = r
        info = dict(p=self.p.copy(), optimal=int(a == int(np.argmax(self.p))))
        return self.observations, self.rewards, self.terminals, self.truncations, [info]

    def render(self):  # pragma: no cover
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return None
        # Skip drawing on non-interactive backends
        try:
            if str(plt.get_backend()).lower().endswith('agg'):
                return None
        except Exception:
            pass
        if self.p is None:
            return None
        x = np.arange(self.k)
        plt.clf()
        plt.bar(x, self.p, color="tab:blue", alpha=0.7)
        if self._last_action is not None:
            la = self._last_action
            plt.bar([la], [self.p[la]], color="tab:orange", alpha=0.9, label=f"chosen (r={self._last_reward})")
            plt.legend()
        plt.ylim(0.0, 1.0)
        plt.xlabel("arm"); plt.ylabel("true p")
        plt.title("Puffer Bernoulli Bandit")
        plt.pause(0.001)
        return None

    def close(self):  # pragma: no cover
        return None


class PufferContextualBandit(pufferlib.PufferEnv):
    """Native PufferLib environment for a contextual bandit.

    Observation: (k, d) context matrix; reward is Bernoulli with logistic link.
    """

    def __init__(self, k: int = 10, d: int = 8, nonstationary: bool = False, theta_sigma: float = 0.05, x_sigma: float = 1.0, buf=None, seed: int | None = None):
        self.k = int(k)
        self.d = int(d)
        self.nonstationary = bool(nonstationary)
        self.theta_sigma = float(theta_sigma)
        self.x_sigma = float(x_sigma)
        self.single_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.k, self.d), dtype=np.float32)
        self.single_action_space = gym.spaces.Discrete(self.k)
        self.num_agents = 1
        super().__init__(buf=buf)
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self.theta: np.ndarray | None = None
        self._obs: np.ndarray | None = None
        self._last_action: int | None = None
        self._last_p: np.ndarray | None = None

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def _sample_theta(self) -> np.ndarray:
        return self._rng.normal(0.0, 1.0, size=(self.k, self.d)).astype(np.float32)

    def _sample_obs(self) -> np.ndarray:
        return self._rng.normal(0.0, self.x_sigma, size=(self.k, self.d)).astype(np.float32)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.theta = self._sample_theta()
        self._obs = self._sample_obs()
        self.observations[0] = self._obs
        self.rewards[0] = 0.0
        self.terminals[0] = False
        self.truncations[0] = False
        return self.observations, [dict()]

    def step(self, actions):
        assert self.theta is not None and self._obs is not None
        a = int(actions[0]) if isinstance(actions, np.ndarray) else int(actions)
        X = self._obs
        logits = (X @ self.theta.T).diagonal() if X.shape[0] == self.theta.shape[0] else (X @ self.theta.T).diagonal()
        if logits.ndim != 1:
            logits = np.einsum("ad,ad->a", X, self.theta)
        p_vec = self._sigmoid(logits)
        r = 1 if self._rng.random() < float(p_vec[a]) else 0
        optimal = int(a == int(np.argmax(p_vec)))
        info = dict(p=p_vec.copy(), optimal=optimal)
        # Evolve theta if non-stationary
        if self.nonstationary:
            self.theta = self.theta + self._rng.normal(0.0, self.theta_sigma, size=self.theta.shape).astype(np.float32)
        # Next obs
        self._obs = self._sample_obs()
        self.observations[0] = self._obs
        self.rewards[0] = float(r)
        self.terminals[0] = False
        self.truncations[0] = False
        self._last_action = a
        self._last_p = p_vec.copy()
        return self.observations, self.rewards, self.terminals, self.truncations, [info]

    def render(self):  # pragma: no cover
        if plt is None or self._last_p is None:
            return None
        p_vec = self._last_p
        x = np.arange(self.k)
        plt.clf()
        plt.bar(x, p_vec, color="tab:blue", alpha=0.7)
        if self._last_action is not None:
            la = self._last_action
            plt.bar([la], [p_vec[la]], color="tab:orange", alpha=0.9, label="chosen")
            plt.legend()
        plt.ylim(0.0, 1.0)
        plt.xlabel("arm"); plt.ylabel("p(x)")
        plt.title("Puffer Contextual Bandit")
        plt.pause(0.001)
        return None

    def close(self):  # pragma: no cover
        return None


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
        a = int(actions[0])
        p_a = float(self.p[a])
        r = 1.0 if (self._rng.random() < p_a) else 0.0
        self.rewards[0] = r
        if self.nonstationary:
            noise = self._rng.normal(0.0, self.sigma, size=self.k)
            self.p = np.clip(self.p + noise, 0.0, 1.0)
        info = dict(p=self.p.copy(), optimal=int(a == int(np.argmax(self.p))))
        return self.observations, self.rewards, self.terminals, self.truncations, [info]

    def close(self):
        # No resources to release; provide method to satisfy vectorizer
        return None


class PufferContextualBandit(pufferlib.PufferEnv):
    """Native PufferLib contextual bandit with per-arm features (k,d)."""

    def __init__(self, k: int = 10, d: int = 8, nonstationary: bool = False, theta_sigma: float = 0.05, x_sigma: float = 1.0, buf=None, seed: int | None = None):
        self.k = int(k)
        self.d = int(d)
        self.nonstationary = bool(nonstationary)
        self.theta_sigma = float(theta_sigma)
        self.x_sigma = float(x_sigma)
        self.single_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(k, d), dtype=np.float32)
        self.single_action_space = gym.spaces.Discrete(self.k)
        self.num_agents = 1
        super().__init__(buf=buf)
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self.theta: np.ndarray | None = None
        self._last_obs: np.ndarray | None = None

    def reset(self, seed: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.theta = self._rng.normal(0.0, 1.0, size=(self.k, self.d)).astype(np.float32)
        self._last_obs = self._rng.normal(0.0, self.x_sigma, size=(self.k, self.d)).astype(np.float32)
        self.observations[0] = 0.0  # unused by vectorizer; obs returned separately
        self.rewards[0] = 0.0
        self.terminals[0] = False
        self.truncations[0] = False
        info = dict(p=None, optimal=None)
        return self._last_obs.copy(), [info]

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def step(self, actions):
        assert self.theta is not None and self._last_obs is not None
        a = int(actions[0])
        X = self._last_obs
        logits = np.einsum("ad,ad->a", X, self.theta)
        p_vec = self._sigmoid(logits)
        p_a = float(p_vec[a])
        r = 1.0 if (self._rng.random() < p_a) else 0.0
        self.rewards[0] = r
        optimal = int(a == int(np.argmax(p_vec)))
        if self.nonstationary:
            self.theta = self.theta + self._rng.normal(0.0, self.theta_sigma, size=self.theta.shape).astype(np.float32)
        # next obs
        self._last_obs = self._rng.normal(0.0, self.x_sigma, size=(self.k, self.d)).astype(np.float32)
        info = dict(p=p_vec.copy(), optimal=optimal)
        return self._last_obs.copy(), self.rewards, self.terminals, self.truncations, [info]

    def close(self):
        return None


class PufferDatasetContextualBandit(pufferlib.PufferEnv):
    """Contextual bandit with pre-generated arrays in an NPZ.

    NPZ keys:
      - X: (N, k, d) contexts
      - P: (N, k) probabilities OR Y: (N, k) binary outcomes
    """

    def __init__(self, k: int, d: int, data_path: str, x_key: str = "X", p_key: str = "P", y_key: str = "Y", buf=None):
        self.k = int(k)
        self.d = int(d)
        self.single_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(k, d), dtype=np.float32)
        self.single_action_space = gym.spaces.Discrete(self.k)
        self.num_agents = 1
        super().__init__(buf=buf)
        data = np.load(data_path)
        self.X = data[x_key].astype(np.float32)
        if p_key in data:
            self.P = data[p_key].astype(np.float32)
            self.Y = None
        elif y_key in data:
            self.P = None
            self.Y = data[y_key].astype(np.float32)
        else:
            raise KeyError("NPZ must contain 'P' or 'Y'")
        self.i = 0

    def reset(self, seed: int | None = None):
        self.i = 0
        obs = self.X[self.i]
        info = dict(p=(self.P[self.i] if self.P is not None else None), optimal=None)
        return obs.copy(), [info]

    def step(self, actions):
        a = int(actions[0])
        self.i = (self.i + 1) % len(self.X)
        obs = self.X[self.i]
        if self.P is not None:
            p_vec = self.P[self.i]
            p_a = float(p_vec[a])
            r = 1.0 if (np.random.random() < p_a) else 0.0
            optimal = int(a == int(np.argmax(p_vec)))
            info = dict(p=p_vec.copy(), optimal=optimal)
        else:
            y_vec = self.Y[self.i]
            r = float(y_vec[a])
            info = dict(p=None, optimal=None)
        self.rewards[0] = r
        return obs.copy(), self.rewards, self.terminals, self.truncations, [info]

    def close(self):
        return None

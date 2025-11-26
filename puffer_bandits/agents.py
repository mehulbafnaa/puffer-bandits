from __future__ import annotations

from dataclasses import dataclass

import torch

from .core.bernoulli import klucb_upper_bound
from .core.nonstationary import discounted_ucb_update, sw_add_remove, ucb_scores
from .utils.device import pick_device as _pick_device
from .utils.constants import TINY


def pick_device(preferred: str | None = None) -> torch.device:  # re-export shim
    return _pick_device(preferred)


@dataclass
class AgentCfg:
    k: int
    num_envs: int
    device: torch.device


class Agent:
    def __init__(self, cfg: AgentCfg, rng: torch.Generator | None = None):
        self.cfg = cfg
        self.device = cfg.device
        self.k = int(cfg.k)
        self.num_envs = int(cfg.num_envs)
        self.rng = rng or torch.Generator(device=self.device)

    def reset(self) -> None:
        raise NotImplementedError

    @torch.no_grad()
    def select_actions(self, t: int) -> torch.LongTensor:
        raise NotImplementedError

    @torch.no_grad()
    def update(self, actions: torch.LongTensor, rewards: torch.Tensor) -> None:
        raise NotImplementedError


class KLUCB(Agent):
    """KL-UCB for Bernoulli rewards (Garivier & Cappé, 2011).

    Upper bound u solves: KL(\hat{p}_a, u) <= (ln t + alpha ln ln t) / N_a
    We compute u via batched bisection.
    """

    def __init__(self, cfg: AgentCfg, alpha: float = 3.0):
        super().__init__(cfg)
        self.alpha = float(alpha)
        self.Q = torch.zeros((self.num_envs, self.k), device=self.device, dtype=torch.float32)
        self.N = torch.zeros((self.num_envs, self.k), device=self.device, dtype=torch.int32)

    def reset(self) -> None:
        self.Q.zero_()
        self.N.zero_()

    @torch.no_grad()
    def _ucb(self, t: int) -> torch.Tensor:
        n = self.N.to(torch.float32)
        return klucb_upper_bound(self.Q, n, t, alpha=self.alpha, iters=25)

    @torch.no_grad()
    def select_actions(self, t: int) -> torch.LongTensor:
        zero_mask = self.N == 0
        any_zero = zero_mask.any(dim=1)
        # Choose among zeros uniformly at random (seeded)
        rand_scores = torch.rand((self.num_envs, self.k), device=self.device, generator=self.rng) * TINY
        zero_scores = torch.where(zero_mask, rand_scores, torch.full_like(rand_scores, -1e9))
        zero_choice = zero_scores.argmax(dim=1)

        u = self._ucb(t)
        choice = u.argmax(dim=1)
        return torch.where(any_zero, zero_choice, choice).long()

    @torch.no_grad()
    def update(self, actions: torch.LongTensor, rewards: torch.Tensor) -> None:
        actions = actions.long()
        rewards = rewards.to(self.device, dtype=torch.float32)
        idx = actions.view(-1, 1)
        self.N.scatter_add_(1, idx, torch.ones((self.num_envs, 1), device=self.device, dtype=self.N.dtype))
        q_sel = self.Q.gather(1, idx)
        n_sel = self.N.gather(1, idx).to(torch.float32)
        alpha = 1.0 / n_sel
        q_new = q_sel + alpha * (rewards.view(-1, 1) - q_sel)
        self.Q.scatter_(1, idx, q_new)


class DiscountedUCB(Agent):
    """Discounted UCB for non-stationary bandits.

    Maintains discounted counts and sums with factor `discount` in (0,1).
    """

    def __init__(self, cfg: AgentCfg, c: float = 2.0, discount: float = 0.99):
        super().__init__(cfg)
        if not (0.0 < discount < 1.0):
            raise ValueError("discount must be in (0,1)")
        if c < 0.0:
            raise ValueError("c must be non-negative")
        self.c = float(c)
        self.discount = float(discount)
        self.S = torch.zeros((self.num_envs, self.k), device=self.device, dtype=torch.float32)
        self.Neff = torch.zeros((self.num_envs, self.k), device=self.device, dtype=torch.float32)

    def reset(self) -> None:
        self.S.zero_()
        self.Neff.zero_()

    @torch.no_grad()
    def select_actions(self, t: int) -> torch.LongTensor:
        # Prefer unpulled arms (Neff==0)
        zero_mask = self.Neff <= 1e-8
        any_zero = zero_mask.any(dim=1)
        rand_scores = torch.rand((self.num_envs, self.k), device=self.device, generator=self.rng) * TINY
        zero_scores = torch.where(zero_mask, rand_scores, torch.full_like(rand_scores, -1e9))
        zero_choice = zero_scores.argmax(dim=1)

        Q = torch.where(self.Neff > 0, self.S / self.Neff.clamp_min(1e-8), torch.zeros_like(self.S))
        scores = ucb_scores(Q, self.Neff, self.c, t)
        choice = scores.argmax(dim=1)
        return torch.where(any_zero, zero_choice, choice).long()

    @torch.no_grad()
    def update(self, actions: torch.LongTensor, rewards: torch.Tensor) -> None:
        a = actions.long()
        r = rewards.to(self.device, dtype=torch.float32)
        self.S, self.Neff = discounted_ucb_update(self.S, self.Neff, a, r, self.discount)


class SlidingWindowUCB(Agent):
    """Sliding-Window UCB for non-stationary bandits.

    Maintains window-limited counts and sums over last `window` steps.
    Uses scores: Q_w + c * sqrt(ln(min(t, window)) / N_w)
    """

    def __init__(self, cfg: AgentCfg, c: float = 2.0, window: int = 200):
        super().__init__(cfg)
        if window <= 0:
            raise ValueError("window must be > 0")
        if c < 0.0:
            raise ValueError("c must be non-negative")
        self.c = float(c)
        self.window = int(window)
        self.Sw = torch.zeros((self.num_envs, self.k), device=self.device, dtype=torch.float32)
        self.Nw = torch.zeros((self.num_envs, self.k), device=self.device, dtype=torch.float32)
        # Buffers of last `window` actions and rewards per env
        self._act_buf = torch.full((self.window, self.num_envs), -1, device=self.device, dtype=torch.long)
        self._rew_buf = torch.zeros((self.window, self.num_envs), device=self.device, dtype=torch.float32)
        self._t = 0

    def reset(self) -> None:
        self.Sw.zero_()
        self.Nw.zero_()
        self._act_buf.fill_(-1)
        self._rew_buf.zero_()
        self._t = 0

    @torch.no_grad()
    def select_actions(self, t: int) -> torch.LongTensor:
        # Prefer arms with Nw==0
        zero_mask = self.Nw <= 1e-8
        any_zero = zero_mask.any(dim=1)
        rand_scores = torch.rand((self.num_envs, self.k), device=self.device, generator=self.rng) * TINY
        zero_scores = torch.where(zero_mask, rand_scores, torch.full_like(rand_scores, -1e9))
        zero_choice = zero_scores.argmax(dim=1)

        Q = torch.where(self.Nw > 0, self.Sw / self.Nw.clamp_min(1e-8), torch.zeros_like(self.Sw))
        # Use UCB with effective time limited by window
        eff_t = min(t, self.window)
        scores = ucb_scores(Q, self.Nw, self.c, eff_t)
        choice = scores.argmax(dim=1)
        return torch.where(any_zero, zero_choice, choice).long()

    @torch.no_grad()
    def update(self, actions: torch.LongTensor, rewards: torch.Tensor) -> None:
        self._t += 1
        pos = (self._t - 1) % self.window
        actions = actions.long()
        rewards = rewards.to(self.device, dtype=torch.float32)

        # Compute leaving entries when buffer is filled else mark invalid
        if self._t > self.window:
            leaving_actions = self._act_buf[pos]
            leaving_rewards = self._rew_buf[pos]
        else:
            leaving_actions = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
            leaving_rewards = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)

        # Sliding-window pure update
        self.Sw, self.Nw = sw_add_remove(
            self.Sw, self.Nw,
            leaving_actions, leaving_rewards,
            actions, rewards,
        )
        # Update buffers
        self._act_buf[pos] = actions
        self._rew_buf[pos] = rewards


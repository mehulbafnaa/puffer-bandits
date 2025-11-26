from __future__ import annotations

import torch
from ..core.adversarial import exp3_probs, exp3_update_factor, exp3ix_probs
from .base import CtxAgent, CtxAgentCfg


class EXP3(CtxAgent):
    requires_obs: bool = False

    def __init__(self, cfg: CtxAgentCfg, gamma: float = 0.07, eta: float | None = None):
        super().__init__(cfg)
        if not (0.0 <= gamma <= 1.0):
            raise ValueError("gamma must be in [0,1]")
        self.gamma = float(gamma)
        self.eta = float(eta) if eta is not None else (self.gamma / max(1, self.k))
        self.w = torch.ones((self.num_envs, self.k), device=self.device, dtype=torch.float32)
        self._last_p: torch.Tensor | None = None

    def reset(self) -> None:
        self.w.fill_(1.0)
        self._last_p = None

    @torch.no_grad()
    def _probs(self) -> torch.Tensor:
        return exp3_probs(self.w, self.gamma)

    @torch.no_grad()
    def select_actions(self, t: int, obs: torch.Tensor | None = None) -> torch.LongTensor:
        p = self._probs(); self._last_p = p
        a = torch.multinomial(p, num_samples=1, replacement=True, generator=self.rng).squeeze(-1)
        return a.long()

    @torch.no_grad()
    def update(self, actions: torch.LongTensor, rewards: torch.Tensor, obs: torch.Tensor | None = None) -> None:
        assert self._last_p is not None, "select_actions must be called before update"
        a = actions.long().view(-1)
        r = rewards.to(self.device, dtype=torch.float32).view(-1)
        p = self._last_p
        batch = torch.arange(self.num_envs, device=self.device)
        p_sel = p[batch, a].clamp_min(1e-12)
        inc = exp3_update_factor(p_sel, r, eta=self.eta, k=self.k)
        self.w[batch, a] = self.w[batch, a] * inc


class EXP3IX(CtxAgent):
    requires_obs: bool = False

    def __init__(self, cfg: CtxAgentCfg, gamma: float = 0.05, eta: float | None = None):
        super().__init__(cfg)
        if gamma <= 0.0 or gamma >= 1.0:
            raise ValueError("gamma must be in (0,1)")
        self.gamma = float(gamma)
        self.eta = float(eta) if eta is not None else min(1.0, (self.gamma / max(1, self.k)))
        self.w = torch.ones((self.num_envs, self.k), device=self.device, dtype=torch.float32)
        self._last_p: torch.Tensor | None = None

    @torch.no_grad()
    def _probs(self) -> torch.Tensor:
        return exp3ix_probs(self.w)

    @torch.no_grad()
    def select_actions(self, t: int, obs: torch.Tensor | None = None) -> torch.LongTensor:
        p = self._probs(); self._last_p = p
        a = torch.multinomial(p, num_samples=1, replacement=True, generator=self.rng).squeeze(-1)
        return a.long()

    @torch.no_grad()
    def update(self, actions: torch.LongTensor, rewards: torch.Tensor, obs: torch.Tensor | None = None) -> None:
        assert self._last_p is not None, "select_actions must be called before update"
        a = actions.long().view(-1)
        r = rewards.to(self.device, dtype=torch.float32).view(-1)
        p = self._last_p
        batch = torch.arange(self.num_envs, device=self.device)
        p_sel = (p[batch, a] + (self.gamma / max(1, self.k))).clamp_min(1e-12)
        inc = exp3_update_factor(p_sel, r, eta=self.eta, k=self.k)
        self.w[batch, a] = self.w[batch, a] * inc

    def reset(self) -> None:
        self.w.fill_(1.0)
        self._last_p = None


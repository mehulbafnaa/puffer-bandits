from __future__ import annotations

from dataclasses import dataclass

import torch

from ..utils.device import pick_device as _pick_device


def pick_device(preferred: str | None = None) -> torch.device:
    return _pick_device(preferred)


@dataclass
class CtxAgentCfg:
    k: int
    d: int
    num_envs: int
    device: torch.device


class CtxAgent:
    requires_obs: bool = True

    def __init__(self, cfg: CtxAgentCfg, rng: torch.Generator | None = None):
        self.cfg = cfg
        self.k = int(cfg.k)
        self.d = int(cfg.d)
        self.num_envs = int(cfg.num_envs)
        self.device = cfg.device
        self.rng = rng or torch.Generator(device=self.device)

    def reset(self) -> None:
        raise NotImplementedError

    @torch.no_grad()
    def select_actions(self, t: int, obs: torch.Tensor | None = None) -> torch.LongTensor:
        raise NotImplementedError

    @torch.no_grad()
    def update(self, actions: torch.LongTensor, rewards: torch.Tensor, obs: torch.Tensor | None = None) -> None:
        raise NotImplementedError


from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BernoulliState:
    Q: torch.Tensor  # (n, k)
    N: torch.Tensor  # (n, k)


@dataclass(frozen=True)
class BetaState:
    alpha: torch.Tensor  # (n, k)
    beta: torch.Tensor   # (n, k)


@dataclass(frozen=True)
class DiscountedState:
    S: torch.Tensor     # (n, k)
    Neff: torch.Tensor  # (n, k)


@dataclass(frozen=True)
class SlidingWindowState:
    Sw: torch.Tensor          # (n, k)
    Nw: torch.Tensor          # (n, k)
    act_buf: torch.Tensor     # (window, n)
    rew_buf: torch.Tensor     # (window, n)
    t: int


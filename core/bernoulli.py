from __future__ import annotations

import math

import torch


def kl_div_bernoulli(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Bernoulli KL divergence ``KL(p || q)`` computed elementwise.

    :param p: First Bernoulli parameter(s).
    :type p: torch.Tensor with values in ``[0, 1]``
    :param q: Second Bernoulli parameter(s).
    :type q: torch.Tensor with values in ``[0, 1]``
    :param eps: Numerical clamp for stability; parameters are clipped to ``[eps, 1-eps]``.
    :type eps: float
    :return: Elementwise KL divergence with broadcasting compatible shape.
    :rtype: torch.Tensor
    """
    p = p.clamp(eps, 1 - eps)
    q = q.clamp(eps, 1 - eps)
    return p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))


@torch.no_grad()
def klucb_upper_bound(
    p_hat: torch.Tensor,
    n: torch.Tensor,
    t: int,
    alpha: float = 3.0,
    iters: int = 25,
) -> torch.Tensor:
    """KL‑UCB upper confidence bound via batched bisection.

    Solves per cell for ``u`` such that ``KL(\hat{p}, u) <= (ln t + alpha ln ln t)/n``.

    :param p_hat: Empirical Bernoulli means ``\hat{p}``.
    :type p_hat: torch.Tensor with values in ``[0, 1]``
    :param n: Counts per cell.
    :type n: torch.Tensor (same broadcastable shape as ``p_hat``)
    :param t: Current timestep (``>= 1``).
    :type t: int
    :param alpha: KL‑UCB parameter multiplying the ``ln ln t`` term.
    :type alpha: float
    :param iters: Number of bisection iterations.
    :type iters: int
    :return: Upper bound ``u`` with the same shape as ``p_hat``.
    :rtype: torch.Tensor
    """
    # Target divergence per cell
    t_ = max(int(t), 2)
    logt = math.log(t_)
    loglog = math.log(max(1.0, math.log(t_))) if t_ >= 3 else 0.0
    target = (logt + alpha * loglog) / n.to(p_hat.dtype).clamp_min(1.0)

    low = p_hat.clamp(0.0, 1.0)
    high = torch.ones_like(low)
    for _ in range(iters):
        mid = (low + high) / 2.0
        kl = kl_div_bernoulli(p_hat, mid)
        cond = kl <= target
        low = torch.where(cond, mid, low)
        high = torch.where(cond, high, mid)
    # low tracks the greatest feasible point (max u s.t. KL <= target)
    return low


import math

import torch


@torch.no_grad()
def discounted_ucb_update(
    S: torch.Tensor,
    Neff: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    discount: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One‑step discounted update for sums and effective counts.

    :param S: Running (discounted) reward sums per arm.
    :type S: torch.Tensor with shape ``(n, k)``
    :param Neff: Running (discounted) effective counts per arm.
    :type Neff: torch.Tensor with shape ``(n, k)``
    :param actions: Chosen arms per environment.
    :type actions: torch.Tensor with shape ``(n,)``
    :param rewards: Received rewards per environment.
    :type rewards: torch.Tensor with shape ``(n,)``
    :param discount: Discount factor in ``(0, 1)``.
    :type discount: float
    :return: Updated ``(S, Neff)`` (new tensors; inputs are not mutated).
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    device = S.device
    n, k = S.shape
    S_new = S * discount
    Neff_new = Neff * discount
    idx = actions.long().view(-1, 1)
    S_new = S_new.clone()
    Neff_new = Neff_new.clone()
    Neff_new.scatter_add_(1, idx, torch.ones((n, 1), device=device, dtype=Neff_new.dtype))
    S_new.scatter_add_(1, idx, rewards.to(S_new.dtype).view(-1, 1))
    return S_new, Neff_new


@torch.no_grad()
def sw_add_remove(
    Sw: torch.Tensor,
    Nw: torch.Tensor,
    leaving_actions: torch.Tensor,
    leaving_rewards: torch.Tensor,
    entering_actions: torch.Tensor,
    entering_rewards: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sliding‑window update: remove leaving sample then add entering sample.

    :param Sw: Windowed reward sums per arm.
    :type Sw: torch.Tensor with shape ``(n, k)``
    :param Nw: Windowed counts per arm.
    :type Nw: torch.Tensor with shape ``(n, k)``
    :param leaving_actions: Indices of arms leaving the window; may be ``-1`` for “no row”.
    :type leaving_actions: torch.Tensor with shape ``(n,)``
    :param leaving_rewards: Rewards associated with the leaving actions.
    :type leaving_rewards: torch.Tensor with shape ``(n,)``
    :param entering_actions: Indices of arms entering the window.
    :type entering_actions: torch.Tensor with shape ``(n,)``
    :param entering_rewards: Rewards associated with the entering actions.
    :type entering_rewards: torch.Tensor with shape ``(n,)``
    :return: Updated ``(Sw, Nw)`` (new tensors; inputs are not mutated).
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    device = Sw.device
    n, k = Sw.shape
    Sw_new = Sw.clone()
    Nw_new = Nw.clone()
    # Remove leaving
    valid = leaving_actions.long() >= 0
    if valid.any():
        b = torch.arange(n, device=device)[valid]
        cols = leaving_actions.long()[valid]
        Nw_new[b, cols] = Nw_new[b, cols] - 1.0
        Sw_new[b, cols] = Sw_new[b, cols] - leaving_rewards.to(Sw_new.dtype)[valid]
    # Add entering
    idx = entering_actions.long().view(-1, 1)
    Nw_new.scatter_add_(1, idx, torch.ones((n, 1), device=device, dtype=Nw_new.dtype))
    Sw_new.scatter_add_(1, idx, entering_rewards.to(Sw_new.dtype).view(-1, 1))
    return Sw_new, Nw_new


@torch.no_grad()
def ucb_scores(Q: torch.Tensor, N: torch.Tensor, c: float, t: int) -> torch.Tensor:
    """Compute UCB1 scores ``Q + c * sqrt(ln t / N)``.

    :param Q: Empirical means per arm.
    :type Q: torch.Tensor with shape ``(n, k)``
    :param N: Counts per arm (``>= 0``).
    :type N: torch.Tensor with shape ``(n, k)``
    :param c: Exploration coefficient (``>= 0``).
    :type c: float
    :param t: Current timestep.
    :type t: int
    :return: UCB scores per arm.
    :rtype: torch.Tensor with shape ``(n, k)``
    """
    logt = math.log(max(int(t), 1))
    return Q + c * torch.sqrt(torch.tensor(logt, device=Q.device, dtype=Q.dtype) / N.clamp_min(1.0))

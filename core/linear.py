from __future__ import annotations

import torch


@torch.no_grad()
def sherman_morrison_update(A_inv: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Rank‑1 Sherman–Morrison update for a batch of inverse matrices.

    :param A_inv: Current inverse matrices (one per batch item).
    :type A_inv: torch.Tensor with shape ``(..., d, d)``
    :param x: Update vectors (one per batch item). The outer product ``x x^T`` is applied.
    :type x: torch.Tensor with shape ``(..., d)``
    :return: Updated inverse matrices ``(A + x x^T)^{-1}`` with the same leading batch dims.
    :rtype: torch.Tensor with shape ``(..., d, d)``
    """
    Ainv_x = torch.matmul(A_inv, x.unsqueeze(-1)).squeeze(-1)
    denom = (1.0 + (x * Ainv_x).sum(dim=-1, keepdim=True)).clamp_min(1e-12)
    # Outer product over the feature dim: (..., d, 1) @ (..., 1, d) -> (..., d, d)
    outer = torch.matmul(Ainv_x.unsqueeze(-1), Ainv_x.unsqueeze(-2))
    return A_inv - outer / denom.unsqueeze(-1)

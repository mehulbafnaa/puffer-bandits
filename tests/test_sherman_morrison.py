from __future__ import annotations

import torch
from MAB_GPU.core.linear import sherman_morrison_update


def test_sherman_morrison_identity():
    torch.manual_seed(0)
    n, k, d = 3, 4, 5
    lam = 1.0
    A_inv = torch.eye(d).expand(n, k, d, d).clone() / lam  # inverse of I
    x = torch.randn((n, k, d))

    # Apply per row update for a selected arm per env (simulate batch of (n,) selections)
    # Here we just update all rows independently
    Ainv_new = sherman_morrison_update(A_inv, x)
    # Check (A + x x^T)^{-1} @ (A + x x^T) ≈ I
    A = torch.eye(d).expand(n, k, d, d).clone() * lam
    xxT = torch.einsum("...i,...j->...ij", x, x)
    A_new = A + xxT
    I = torch.eye(d)
    prod = torch.matmul(Ainv_new, A_new)
    err = (prod - I).abs().max().item()
    assert err < 1e-5


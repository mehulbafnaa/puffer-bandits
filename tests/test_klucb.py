from __future__ import annotations

import math

import torch
from MAB_GPU.core.bernoulli import kl_div_bernoulli, klucb_upper_bound


def test_klucb_feasible_and_monotone():
    device = torch.device("cpu")
    torch.manual_seed(0)

    p_hat = torch.rand((8, 7), device=device) * 0.9 + 0.05  # avoid extremes
    n = torch.randint(low=1, high=50, size=(8, 7), device=device).to(torch.float32)
    t = 100
    alpha = 3.0

    u = klucb_upper_bound(p_hat, n, t, alpha=alpha, iters=50)
    target = (math.log(max(t, 2)) + (alpha * (math.log(math.log(max(t, 3))) if t >= 3 else 0.0))) / n.clamp_min(1.0)
    kl = kl_div_bernoulli(p_hat, u)

    # Feasibility: KL(p_hat, u) <= target with small numerical slack
    assert torch.all(kl <= target + 1e-6)

    # Upper confidence: u >= p_hat (monotone upwards)
    assert torch.all(u >= p_hat - 1e-7)


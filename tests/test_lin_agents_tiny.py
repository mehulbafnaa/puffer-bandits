from __future__ import annotations

import torch
import numpy as np

from puffer_bandits.agents_ctx import CtxAgentCfg, LinUCB, LinTS


def test_linucb_scores_match_manual():
    device = torch.device("cpu")
    cfg = CtxAgentCfg(k=3, d=2, num_envs=1, device=device)
    ag = LinUCB(cfg, alpha=0.5, lam=1.0)
    # Manually set A_inv and b
    ag.A_inv[0, 0] = torch.eye(2)
    ag.A_inv[0, 1] = 2.0 * torch.eye(2)
    ag.A_inv[0, 2] = 0.5 * torch.eye(2)
    ag.b[0, 0] = torch.tensor([1.0, 0.0])
    ag.b[0, 1] = torch.tensor([0.0, 1.0])
    ag.b[0, 2] = torch.tensor([0.5, 0.5])
    X = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])  # (1,3,2)
    # Compute scores manually
    theta = torch.matmul(ag.A_inv, ag.b.unsqueeze(-1)).squeeze(-1)
    mean = (X * theta).sum(dim=-1)
    Ainv_x = torch.matmul(ag.A_inv, X.unsqueeze(-1)).squeeze(-1)
    conf2 = (X * Ainv_x).sum(dim=-1).clamp_min(1e-12)
    manual = mean + 0.5 * torch.sqrt(conf2)
    # Agent
    with torch.random.fork_rng():
        ag.rng.manual_seed(0)
        a_sel = ag.select_actions(1, X).item()
    # Verify argmax equals manual argmax
    assert torch.argmax(manual, dim=1).item() == a_sel


def test_lints_v_zero_reduces_to_mean_argmax():
    device = torch.device("cpu")
    cfg = CtxAgentCfg(k=3, d=2, num_envs=1, device=device)
    ag = LinTS(cfg, v=0.0, lam=1.0)
    # Set A_inv=I, b per arm
    ag.A_inv[0] = torch.eye(2).repeat(3, 1, 1)
    ag.b[0, 0] = torch.tensor([2.0, 0.0])
    ag.b[0, 1] = torch.tensor([0.0, 1.0])
    ag.b[0, 2] = torch.tensor([0.5, 0.5])
    X = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
    # mu = b; scores are x*mu
    mu = torch.matmul(ag.A_inv, ag.b.unsqueeze(-1)).squeeze(-1)
    mean = (X * mu).sum(dim=-1)
    a_exp = torch.argmax(mean, dim=1).item()
    a_sel = ag.select_actions(1, X).item()
    assert a_exp == a_sel

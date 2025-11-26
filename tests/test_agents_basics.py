from __future__ import annotations

import torch
from puffer_bandits.advanced_agents import CtxAgentCfg, LinUCB
from puffer_bandits.agents import KLUCB, AgentCfg


def test_klucb_deterministic_tie_break():
    device = torch.device("cpu")
    cfg = AgentCfg(k=5, num_envs=4, device=device)
    g = torch.Generator(device=device)
    g.manual_seed(123)
    a1 = KLUCB(cfg)
    a1.rng = g
    a1.reset()
    a2 = KLUCB(cfg)
    g2 = torch.Generator(device=device)
    g2.manual_seed(123)
    a2.rng = g2
    a2.reset()

    # At t=1 with all-zero counts, both should pick identical arms
    act1 = a1.select_actions(1)
    act2 = a2.select_actions(1)
    assert torch.equal(act1, act2)


def test_linucb_reset_broadcast_shapes():
    device = torch.device("cpu")
    cfg = CtxAgentCfg(k=3, d=4, num_envs=2, device=device)
    agent = LinUCB(cfg, alpha=1.0, lam=2.0)
    agent.reset()
    assert agent.A_inv.shape == (cfg.num_envs, cfg.k, cfg.d, cfg.d)
    # Diagonal entries should be 1/lam
    diag = agent.A_inv[0, 0].diag()
    assert torch.allclose(diag, torch.full((cfg.d,), 1.0 / 2.0))

from __future__ import annotations

import torch

from MAB_GPU.advanced_agents import CtxAgentCfg, LinTS, NeuralLinearTS


def test_lints_safe_cholesky_no_crash():
    device = torch.device("cpu")
    cfg = CtxAgentCfg(k=2, d=2, num_envs=2, device=device)
    agent = LinTS(cfg, v=0.1, lam=1.0)
    agent.reset()
    # Make one block of A_inv indefinite to trigger fallback
    bad = torch.tensor([[1.0, 2.0], [2.0, -0.1]], device=device)
    agent.A_inv[0, 0, :, :] = bad
    obs = torch.randn((cfg.num_envs, cfg.k, cfg.d), device=device)
    # Should not raise
    _ = agent.select_actions(1, obs)


def test_neurallinear_safe_cholesky_no_crash():
    device = torch.device("cpu")
    cfg = CtxAgentCfg(k=2, d=3, num_envs=2, device=device)
    agent = NeuralLinearTS(cfg, m=4, hidden=8, depth=1)
    agent.reset()
    # Corrupt one A_inv slice
    bad = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                        [0.0, -0.01, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]], device=device)
    agent.A_inv[0, 0, :, :] = bad
    obs = torch.randn((cfg.num_envs, cfg.k, cfg.d), device=device)
    _ = agent.select_actions(1, obs)


from __future__ import annotations

import torch
from puffer_bandits.advanced_agents import EXP3IX, CtxAgentCfg


def test_exp3ix_update_changes_selected_only_and_probs_sum1():
    device = torch.device("cpu")
    cfg = CtxAgentCfg(k=7, d=1, num_envs=3, device=device)
    ag = EXP3IX(cfg, gamma=0.1)
    ag.reset()
    # Start with uniform probs
    p = ag._probs()
    assert torch.allclose(p.sum(dim=1), torch.ones(cfg.num_envs))

    # Choose a fixed action per env
    actions = torch.tensor([0, 1, 2], dtype=torch.long)
    rewards = torch.ones(cfg.num_envs, dtype=torch.float32)
    # Store current weights
    w_before = ag.w.clone()
    ag._last_p = p
    ag.update(actions, rewards, None)

    # Only selected columns should change and be strictly larger
    for i in range(cfg.num_envs):
        sel = actions[i].item()
        assert ag.w[i, sel] > w_before[i, sel]
        mask = torch.ones(cfg.k, dtype=torch.bool)
        mask[sel] = False
        assert torch.allclose(ag.w[i, mask], w_before[i, mask])

    # Probs after renormalization still sum to 1
    p2 = ag._probs()
    assert torch.allclose(p2.sum(dim=1), torch.ones(cfg.num_envs))

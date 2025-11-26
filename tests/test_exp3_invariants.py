from __future__ import annotations

import torch
from puffer_bandits.agents_ctx import CtxAgentCfg, EXP3


def test_exp3_probs_sum_to_one_and_update_increases_selected_weight():
    device = torch.device("cpu")
    cfg = CtxAgentCfg(k=5, d=1, num_envs=3, device=device)
    ag = EXP3(cfg, gamma=0.07)
    ag.reset()
    p = ag._probs()
    assert torch.allclose(p.sum(dim=1), torch.ones(cfg.num_envs))
    actions = torch.tensor([0, 1, 2], dtype=torch.long)
    rewards = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    w_before = ag.w.clone()
    ag._last_p = p
    ag.update(actions, rewards, None)
    for i in range(cfg.num_envs):
        a = actions[i].item()
        assert ag.w[i, a] > w_before[i, a]

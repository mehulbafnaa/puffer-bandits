from __future__ import annotations

import torch
from MAB_GPU.core.nonstationary import discounted_ucb_update, sw_add_remove


def test_sliding_window_sentinel_noop():
    n, k = 4, 6
    Sw = torch.zeros((n, k))
    Nw = torch.zeros((n, k))
    leaving_actions = torch.full((n,), -1)
    leaving_rewards = torch.zeros((n,))
    entering_actions = torch.zeros((n,), dtype=torch.long)
    entering_rewards = torch.ones((n,))

    Sw2, Nw2 = sw_add_remove(Sw, Nw, leaving_actions, leaving_rewards, entering_actions, entering_rewards)
    # One reward added to arm 0 per row
    assert torch.allclose(Nw2[:, 0], torch.ones(n))
    assert torch.allclose(Sw2[:, 0], torch.ones(n))
    # Others remain zero
    assert torch.all(Nw2[:, 1:] == 0)
    assert torch.all(Sw2[:, 1:] == 0)


def test_discounted_ucb_update_shapes_and_values():
    n, k = 3, 5
    S = torch.zeros((n, k))
    Neff = torch.zeros((n, k))
    actions = torch.tensor([0, 1, 2])
    rewards = torch.tensor([1.0, 0.0, 1.0])
    disc = 0.9

    S2, N2 = discounted_ucb_update(S, Neff, actions, rewards, disc)
    assert S2.shape == (n, k) and N2.shape == (n, k)
    # One step: no prior values to decay; the selected entries should be 1, others 0
    assert torch.allclose(S2[0, 0], torch.tensor(1.0))
    assert torch.allclose(N2[0, 0], torch.tensor(1.0))
    assert torch.all(S2[0, 1:] == 0)
    assert torch.all(N2[0, 1:] == 0)


from __future__ import annotations

import numpy as np
import torch

import MAB_GPU.runner_puffer_native as rpn


def test_native_runner_smoke_bernoulli_serial():
    cfg = rpn.Config(env="bernoulli", algo="klucb", k=5, d=1, T=10, runs=4, seed=0, vector="serial", device="cpu", log_every=100)
    device = torch.device("cpu")
    vec = rpn.build_envs(cfg)
    obs, infos = vec.reset(seed=cfg.seed)
    agent = rpn.build_agent(cfg, device)
    agent.reset()
    for t in range(1, cfg.T + 1):
        actions = agent.select_actions(t)
        atn_np = actions.detach().cpu().numpy()
        next_obs, r, _, _, infos = vec.step(atn_np)
        rewards = torch.from_numpy(r.astype(np.float32)).to(device=device)
        agent.update(actions, rewards)
        obs = next_obs
    vec.close()


def test_native_runner_smoke_contextual_serial():
    cfg = rpn.Config(env="contextual", algo="linucb", k=5, d=3, T=10, runs=4, seed=0, vector="serial", device="cpu", log_every=100)
    device = torch.device("cpu")
    vec = rpn.build_envs(cfg)
    obs, infos = vec.reset(seed=cfg.seed)
    agent = rpn.build_agent(cfg, device)
    agent.reset()
    for t in range(1, cfg.T + 1):
        obs_t = torch.from_numpy(obs).to(device=device, dtype=torch.float32)
        actions = agent.select_actions(t, obs_t)
        atn_np = actions.detach().cpu().numpy()
        next_obs, r, _, _, infos = vec.step(atn_np)
        rewards = torch.from_numpy(r.astype(np.float32)).to(device=device)
        agent.update(actions, rewards, obs_t)
        obs = next_obs
    vec.close()


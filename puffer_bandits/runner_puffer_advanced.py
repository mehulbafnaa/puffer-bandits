from __future__ import annotations

import argparse
import os
import time
import csv
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

CORES = os.cpu_count() or 1

from pufferlib.emulation import GymnasiumPufferEnv
from pufferlib.vector import Multiprocessing

from .agents_ctx import (
    EXP3,
    EXP3IX,
    CtxAgentCfg,
    LinTS,
    LinUCB,
    NeuralLinearTS,
    NeuralTS,
)
from .agents_ctx import pick_device
from .bandit_env import BernoulliBanditEnv
from .contextual_env import ContextualBanditEnv
from .utils.device import memory_stats, sync_device
try:
    import matplotlib.pyplot as plt  # optional plotting
except Exception:
    plt = None  # type: ignore


@dataclass
class Config:
    # Problem/Env
    algo: Literal["linucb", "lints", "exp3", "exp3ix", "neuralts", "neurallinear"] = "linucb"
    env: Literal["contextual", "bernoulli"] = "contextual"
    k: int = 10
    d: int = 8
    T: int = 1000
    runs: int = 4096
    seed: int = 0
    nonstationary: bool = False
    sigma: float = 0.1
    theta_sigma: float = 0.05
    x_sigma: float = 1.0
    # Agents
    alpha: float = 1.0   # LinUCB
    lam: float = 1.0     # LinUCB/LinTS
    v: float = 0.1       # LinTS
    gamma: float = 0.07  # EXP3
    eta: float | None = None
    hidden: int = 128
    depth: int = 2
    ensembles: int = 5
    dropout: float = 0.1
    lr: float = 1e-3
    # NeuralLinear
    features: int = 64
    linlam: float = 1.0
    linv: float = 0.1
    # Vectorization
    num_workers: int | None = None
    batch_size: int | None = None
    device: str | None = None
    log_every: int = 100
    force_device: bool = False
    amp: bool = True
    compile: bool = False
    # Output
    outdir: str = "plots_gpu"
    save_csv: bool = False
    debug_devices: bool = False
    profile: bool = False
    # Logging (optional)
    wandb: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_tags: str | None = None
    wandb_offline: bool = False
    run_name: str | None = None


def parse_args() -> Config:
    p = argparse.ArgumentParser("Advanced contextual/adversarial bandits (PufferLib)")
    p.add_argument("--algo", type=str, choices=["linucb", "lints", "exp3", "exp3ix", "neuralts", "neurallinear"], default="linucb")
    p.add_argument("--env", type=str, choices=["contextual", "bernoulli"], default="contextual")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--d", type=int, default=8)
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--runs", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--nonstationary", action="store_true")
    p.add_argument("--sigma", type=float, default=0.1, help="non-stationary reward sigma (bernoulli)")
    p.add_argument("--theta-sigma", type=float, default=0.05, help="non-stationary theta sigma (contextual)")
    p.add_argument("--x-sigma", type=float, default=1.0)
    # Agent params
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--v", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.07)
    p.add_argument("--eta", type=float, default=None)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--ensembles", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--features", type=int, default=64, help="NeuralLinear feature dim m")
    p.add_argument("--linlam", type=float, default=1.0, help="NeuralLinear ridge lam")
    p.add_argument("--linv", type=float, default=0.1, help="NeuralLinear TS scale v")
    # Vectorization
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--log-every", type=int, default=100)
    # Output
    p.add_argument("--outdir", type=str, default="plots_gpu")
    p.add_argument("--save-csv", action="store_true")
    p.add_argument("--debug-devices", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--force-device", action="store_true")
    p.add_argument("--no-amp", action="store_false", dest="amp")
    p.add_argument("--compile", action="store_true")
    # Weights & Biases (optional)
    p.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases")
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-tags", type=str, default=None, help=",-separated tags")
    p.add_argument("--wandb-offline", action="store_true")
    p.add_argument("--run-name", type=str, default=None)
    args = p.parse_args()
    return Config(
        algo=args.algo, env=args.env, k=args.k, d=args.d, T=args.T, runs=args.runs, seed=args.seed,
        nonstationary=args.nonstationary, sigma=args.sigma, theta_sigma=args.theta_sigma, x_sigma=args.x_sigma,
        alpha=args.alpha, lam=args.lam, v=args.v, gamma=args.gamma, eta=args.eta,
        hidden=args.hidden, depth=args.depth, ensembles=args.ensembles, dropout=args.dropout, lr=args.lr,
        features=args.features, linlam=args.linlam, linv=args.linv,
        num_workers=args.num_workers, batch_size=args.batch_size, device=args.device, log_every=args.log_every, force_device=bool(args.force_device), amp=bool(args.amp), compile=bool(args.compile),
        outdir=args.outdir, save_csv=bool(args.save_csv), debug_devices=bool(args.debug_devices), profile=bool(args.profile),
        wandb=bool(args.wandb), wandb_project=args.wandb_project, wandb_entity=args.wandb_entity, wandb_tags=args.wandb_tags, wandb_offline=bool(args.wandb_offline), run_name=args.run_name,
    )


def build_envs(cfg: Config) -> Multiprocessing:
    env_creators = [GymnasiumPufferEnv for _ in range(cfg.runs)]
    env_args = [[] for _ in range(cfg.runs)]
    if cfg.env == "contextual":
        env_kwargs = [
            dict(env_creator=ContextualBanditEnv, env_args=[], env_kwargs=dict(
                k=cfg.k, d=cfg.d, nonstationary=cfg.nonstationary, theta_sigma=cfg.theta_sigma, x_sigma=cfg.x_sigma
            )) for _ in range(cfg.runs)
        ]
    else:
        env_kwargs = [
            dict(env_creator=BernoulliBanditEnv, env_args=[], env_kwargs=dict(
                k=cfg.k, nonstationary=cfg.nonstationary, sigma=cfg.sigma
            )) for _ in range(cfg.runs)
        ]

    num_workers = cfg.num_workers if cfg.num_workers is not None else min(cfg.runs, CORES)
    num_workers = max(1, min(num_workers, cfg.runs))
    return Multiprocessing(
        env_creators=env_creators,
        env_args=env_args,
        env_kwargs=env_kwargs,
        num_envs=cfg.runs,
        num_workers=num_workers,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        overwork=True,
    )


def build_agent(cfg: Config, device: torch.device):
    acfg = CtxAgentCfg(k=cfg.k, d=cfg.d, num_envs=cfg.runs, device=device)
    if cfg.algo == "linucb":
        return LinUCB(acfg, alpha=cfg.alpha, lam=cfg.lam)
    if cfg.algo == "lints":
        return LinTS(acfg, v=cfg.v, lam=cfg.lam)
    if cfg.algo == "exp3":
        return EXP3(acfg, gamma=cfg.gamma, eta=cfg.eta)
    if cfg.algo == "exp3ix":
        return EXP3IX(acfg, gamma=cfg.gamma, eta=cfg.eta)
    if cfg.algo == "neuralts":
        return NeuralTS(acfg, hidden=cfg.hidden, depth=cfg.depth, ensembles=cfg.ensembles, dropout=cfg.dropout, lr=cfg.lr, amp=cfg.amp, use_compile=cfg.compile)
    if cfg.algo == "neurallinear":
        return NeuralLinearTS(acfg, m=cfg.features, hidden=cfg.hidden, depth=cfg.depth, dropout=cfg.dropout, lam=cfg.linlam, v=cfg.linv, lr=cfg.lr, amp=cfg.amp, use_compile=cfg.compile)
    raise ValueError("unknown algo")


def main() -> None:
    cfg = parse_args()
    device = pick_device(cfg.device)
    if (not cfg.force_device
        and device.type == "mps"
        and cfg.env == "contextual"
        and cfg.algo in ("lints", "neuralts", "neurallinear")
        and (cfg.k * max(1, cfg.d) <= 128)
        and (cfg.runs <= 512)):
        print("[heuristic] Using CPU instead of MPS for small contextual workload (override with --force-device).")
        device = torch.device("cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    vec = build_envs(cfg)
    obs, infos = vec.reset(seed=cfg.seed)
    # obs is numpy: contextual -> (runs,k,d), bernoulli -> scalar (ignored by agents that don't need it)

    agent = build_agent(cfg, device)
    agent.reset()
    if cfg.debug_devices:
        print("[device] torch backend device:", device)

    T = cfg.T
    n = cfg.runs
    mean_reward = np.zeros(T, dtype=float)
    reward_lo = np.zeros(T, dtype=float)
    reward_hi = np.zeros(T, dtype=float)
    pct_opt = np.zeros(T, dtype=float)
    pct_opt_lo = np.zeros(T, dtype=float)
    pct_opt_hi = np.zeros(T, dtype=float)
    cumulative_regret = np.zeros(T, dtype=float)
    cumulative_regret_lo = np.zeros(T, dtype=float)
    cumulative_regret_hi = np.zeros(T, dtype=float)

    cumulative_reward = np.zeros(n, dtype=float)
    pstar0 = np.zeros(n, dtype=float)
    have_pstar0 = False
    cumulative_pstar = np.zeros(n, dtype=float)

    # Profiling accumulators (seconds)
    prof = {"select": 0.0, "actions_to_cpu": 0.0, "env_step": 0.0, "rewards_to_device": 0.0, "update": 0.0}

    # Preallocate buffers
    rewards_t = torch.empty((n,), device=device, dtype=torch.float32)
    obs_buf = torch.empty((n, cfg.k, max(1, cfg.d)), device=device, dtype=torch.float32)

    # Optional Weights & Biases
    wb = None
    if cfg.wandb:
        try:
            import os as _os
            import wandb as _wandb  # type: ignore
            if cfg.wandb_offline:
                _os.environ.setdefault("WANDB_MODE", "offline")
            run_name = cfg.run_name or f"advanced-{cfg.env}-{cfg.algo}-k{cfg.k}-d{cfg.d}-n{cfg.runs}-{device}-{cfg.seed}"
            wb_cfg = {
                "runner": "advanced",
                "env": cfg.env,
                "algo": cfg.algo,
                "k": cfg.k,
                "d": cfg.d,
                "T": cfg.T,
                "runs": cfg.runs,
                "device": str(device),
                "seed": cfg.seed,
            }
            wb = _wandb.init(
                project=cfg.wandb_project or "puffer-bandits",
                entity=cfg.wandb_entity,
                name=run_name,
                config=wb_cfg,
                tags=[t.strip() for t in (cfg.wandb_tags or "").split(",") if t.strip()],
                reinit=True,
            )
        except Exception:
            wb = None

    for t in range(1, T + 1):
        # Prepare obs for agent (if required). EXP3 ignores obs.
        if cfg.env == "contextual":
            obs_host = obs  # numpy
        else:
            obs_host = None

        if cfg.profile:
            sync_device(device)
            t0 = time.perf_counter()
            if cfg.env == "contextual":
                obs_buf.copy_(torch.from_numpy(obs_host).to(device=device, dtype=torch.float32))
                actions = agent.select_actions(t, obs_buf)
            else:
                actions = agent.select_actions(t)
            sync_device(device)
            prof["select"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            atn_np = actions.detach().cpu().numpy()
            prof["actions_to_cpu"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            obs, r, _, _, infos = vec.step(atn_np)
            prof["env_step"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            rewards_t.copy_(torch.from_numpy(r.astype(np.float32)).to(device=device))
            sync_device(device)
            prof["rewards_to_device"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            if cfg.env == "contextual":
                agent.update(actions, rewards_t, obs_buf)
            else:
                agent.update(actions, rewards_t)
            sync_device(device)
            prof["update"] += time.perf_counter() - t0
        else:
            if cfg.env == "contextual":
                obs_t = torch.from_numpy(obs).to(device=device, dtype=torch.float32)
                actions = agent.select_actions(t, obs_t)
            else:
                actions = agent.select_actions(t)
            atn_np = actions.detach().cpu().numpy()
            obs, r, _, _, infos = vec.step(atn_np)
            rewards = torch.from_numpy(r.astype(np.float32)).to(device=device)
            if cfg.env == "contextual":
                agent.update(actions, rewards, obs_t)
            else:
                agent.update(actions, rewards)

        if t % cfg.log_every == 0:
            mean_r = float(np.mean(r))
            print(f"t={t} mean_reward={mean_r:.4f}")
            if wb is not None:
                try:
                    import wandb as _wandb  # type: ignore
                    _wandb.log({
                        "t": t,
                        "mean_reward": mean_r,
                    }, step=t)
                except Exception:
                    pass
            if cfg.profile:
                steps_done = t
                ms = {k: 1000.0 * v / steps_done for k, v in prof.items()}
                print(
                    "profile(ms/step):",
                    f"select={ms['select']:.3f}",
                    f"a2cpu={ms['actions_to_cpu']:.3f}",
                    f"step={ms['env_step']:.3f}",
                    f"r2dev={ms['rewards_to_device']:.3f}",
                    f"update={ms['update']:.3f}",
                )

    vec.close()

    # Optionally print memory stats at end
    if cfg.debug_devices:
        print(memory_stats())
    if wb is not None:
        try:
            import wandb as _wandb  # type: ignore
            _wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()

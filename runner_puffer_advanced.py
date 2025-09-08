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

from MAB_GPU.agents_ctx import (
    EXP3,
    EXP3IX,
    CtxAgentCfg,
    LinTS,
    LinUCB,
    NeuralLinearTS,
    NeuralTS,
)
from MAB_GPU.agents_ctx import pick_device
from MAB_GPU.bandit_env import BernoulliBanditEnv
from MAB_GPU.contextual_env import ContextualBanditEnv
from MAB_GPU.utils.device import memory_stats, sync_device
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


def parse_args() -> Config:
    p = argparse.ArgumentParser("Advanced GPU MAB (contextual/adversarial) with PufferLib")
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
    args = p.parse_args()
    return Config(
        algo=args.algo, env=args.env, k=args.k, d=args.d, T=args.T, runs=args.runs, seed=args.seed,
        nonstationary=args.nonstationary, sigma=args.sigma, theta_sigma=args.theta_sigma, x_sigma=args.x_sigma,
        alpha=args.alpha, lam=args.lam, v=args.v, gamma=args.gamma, eta=args.eta,
        hidden=args.hidden, depth=args.depth, ensembles=args.ensembles, dropout=args.dropout, lr=args.lr,
        features=args.features, linlam=args.linlam, linv=args.linv,
        num_workers=args.num_workers, batch_size=args.batch_size, device=args.device, log_every=args.log_every, force_device=bool(args.force_device), amp=bool(args.amp), compile=bool(args.compile),
        outdir=args.outdir, save_csv=bool(args.save_csv), debug_devices=bool(args.debug_devices), profile=bool(args.profile),
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

    for t in range(1, T + 1):
        # Prepare obs for agent (if required). EXP3 ignores obs.
        if cfg.env == "contextual":
            obs_host = obs  # numpy
        else:
            obs_host = None

        if cfg.profile:
            sync_device(device)
            t0 = time.perf_counter()
            if obs_host is not None:
                obs_t = obs_buf
                obs_t.copy_(torch.from_numpy(obs_host).to(device=device, dtype=torch.float32))
            else:
                obs_t = obs_buf
                obs_t.zero_()
            actions = agent.select_actions(t, obs_t)
            sync_device(device)
            prof["select"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            atn_np = actions.detach().cpu().numpy()
            prof["actions_to_cpu"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            next_obs, r, _, _, infos = vec.step(atn_np)
            prof["env_step"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            rewards = rewards_t
            rewards.copy_(torch.from_numpy(r.astype(np.float32)).to(device=device))
            sync_device(device)
            prof["rewards_to_device"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            agent.update(actions, rewards, obs_t)
            sync_device(device)
            prof["update"] += time.perf_counter() - t0
        else:
            if obs_host is not None:
                obs_t = obs_buf
                obs_t.copy_(torch.from_numpy(obs_host).to(device=device, dtype=torch.float32))
            else:
                obs_t = obs_buf
                obs_t.zero_()
            actions = agent.select_actions(t, obs_t)
            atn_np = actions.detach().cpu().numpy()
            next_obs, r, _, _, infos = vec.step(atn_np)
            rewards = rewards_t
            rewards.copy_(torch.from_numpy(r.astype(np.float32)).to(device=device))
            agent.update(actions, rewards, obs_t)

        # Metrics and bookkeeping
        cumulative_reward += r.astype(float)
        if isinstance(infos, list) and infos:
            opt_flags = np.array([int(i.get("optimal", 0)) for i in infos], dtype=float)
            pstars = np.array([float(np.nanmax(i.get("p", [np.nan]))) for i in infos], dtype=float)
        else:
            opt_flags = np.zeros(n, dtype=float)
            pstars = np.full(n, np.nan, dtype=float)

        if not have_pstar0:
            pstar0 = np.where(np.isnan(pstars), pstar0, pstars)
            have_pstar0 = True

        if cfg.nonstationary or cfg.env == "contextual":
            cumulative_pstar += np.nan_to_num(pstars, nan=0.0)

        # Reward CI
        m = float(r.mean())
        se = float(r.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        mean_reward[t - 1] = m
        reward_lo[t - 1] = m - 1.96 * se
        reward_hi[t - 1] = m + 1.96 * se

        # % optimal CI
        pct = float(opt_flags.mean() * 100.0)
        pct_se = float(opt_flags.std(ddof=1) / np.sqrt(n) * 100.0) if n > 1 else 0.0
        pct_opt[t - 1] = pct
        pct_opt_lo[t - 1] = pct - 1.96 * pct_se
        pct_opt_hi[t - 1] = pct + 1.96 * pct_se

        # Regret CI
        if cfg.nonstationary or cfg.env == "contextual":
            regrets = cumulative_pstar - cumulative_reward
        else:
            regrets = pstar0 * t - cumulative_reward
        rg_m = float(regrets.mean())
        rg_se = float(regrets.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        cumulative_regret[t - 1] = rg_m
        cumulative_regret_lo[t - 1] = rg_m - 1.96 * rg_se
        cumulative_regret_hi[t - 1] = rg_m + 1.96 * rg_se

        if t % cfg.log_every == 0:
            print(f"t={t} mean_reward={m:.4f} %optimal={pct:.2f} regret={rg_m:.4f}")
            if cfg.profile:
                steps_done = t
                ms = {k: 1000.0 * v / steps_done for k, v in prof.items()}
                mem = memory_stats(device)
                print(
                    "profile(ms/step):",
                    f"select={ms['select']:.3f}",
                    f"a2cpu={ms['actions_to_cpu']:.3f}",
                    f"step={ms['env_step']:.3f}",
                    f"r2dev={ms['rewards_to_device']:.3f}",
                    f"update={ms['update']:.3f}",
                    "mem=", mem,
                )

        # Advance obs
        obs = next_obs

    vec.close()

    # Save plots
    if plt is not None:
        os.makedirs(cfg.outdir, exist_ok=True)
        def tag() -> str:
            parts = [f"algo={cfg.algo}", f"env={cfg.env}", f"k={cfg.k}", f"d={cfg.d}", f"T={cfg.T}", f"runs={cfg.runs}"]
            if cfg.algo == "linucb":
                parts += [f"alpha={cfg.alpha}", f"lam={cfg.lam}"]
            if cfg.algo == "lints":
                parts += [f"v={cfg.v}", f"lam={cfg.lam}"]
            if cfg.algo == "exp3":
                parts += [f"gamma={cfg.gamma}"]
            return "_".join(parts)
        x = np.arange(1, T + 1)
        plt.figure(figsize=(6,4))
        plt.plot(x, mean_reward, label="mean reward")
        plt.fill_between(x, reward_lo, reward_hi, alpha=0.2, label="95% CI")
        plt.xlabel("t"); plt.ylabel("Mean reward"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(cfg.outdir, f"adv_reward_{tag()}.png")); plt.close()

        plt.figure(figsize=(6,4))
        plt.plot(x, pct_opt, label="% optimal")
        plt.fill_between(x, pct_opt_lo, pct_opt_hi, alpha=0.2, label="95% CI")
        plt.xlabel("t"); plt.ylabel("% optimal"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(cfg.outdir, f"adv_pct_optimal_{tag()}.png")); plt.close()

        plt.figure(figsize=(6,4))
        plt.plot(x, cumulative_regret, label="cumulative regret")
        plt.fill_between(x, cumulative_regret_lo, cumulative_regret_hi, alpha=0.2, label="95% CI")
        plt.xlabel("t"); plt.ylabel("Cumulative regret"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(cfg.outdir, f"adv_regret_{tag()}.png")); plt.close()
    else:
        print("Warning: matplotlib not available; skipping plots")

    if cfg.save_csv:
        os.makedirs(cfg.outdir, exist_ok=True)
        path = os.path.join(cfg.outdir, "advanced_summary.csv")
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if f.tell() == 0:
                w.writerow(["algo","env","k","d","T","runs","seed","alpha","lam","v","gamma","final_mean_reward","final_%_optimal","final_cumulative_regret"])
            w.writerow([
                cfg.algo, cfg.env, cfg.k, cfg.d, T, cfg.runs, cfg.seed, cfg.alpha, cfg.lam, cfg.v, cfg.gamma,
                float(mean_reward[-1]), float(pct_opt[-1]), float(cumulative_regret[-1])
            ])

    print(f"Mean reward (first 5): {np.round(mean_reward[:5], 3)}")
    print(f"% optimal (first 5): {np.round(pct_opt[:5], 1)}")
    if cfg.profile:
        ms = {k: 1000.0 * v / T for k, v in prof.items()}
        mem = memory_stats(device)
        print(
            "Final profile(ms/step):",
            f"select={ms['select']:.3f}",
            f"a2cpu={ms['actions_to_cpu']:.3f}",
            f"step={ms['env_step']:.3f}",
            f"r2dev={ms['rewards_to_device']:.3f}",
            f"update={ms['update']:.3f}",
            "mem=", mem,
        )


if __name__ == "__main__":
    main()

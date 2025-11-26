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

from .agents import (
    KLUCB,
    AgentCfg,
    DiscountedUCB,
    SlidingWindowUCB,
)
from .bandit_env import BernoulliBanditEnv
from .utils.device import pick_device
try:
    import matplotlib.pyplot as plt  # optional plotting
except Exception:
    plt = None  # type: ignore


@dataclass
class Config:
    # Problem
    algo: Literal["klucb", "ducb", "swucb"] = "klucb"
    k: int = 10
    T: int = 1000
    runs: int = 4096  # number of parallel envs (aka runs)
    seed: int = 0
    # Agent hyperparameters
    c: float | None = None
    kl_alpha: float | None = None
    discount: float | None = None
    window: int | None = None
    # Environment
    nonstationary: bool = False
    sigma: float = 0.1
    # Vectorization
    num_workers: int | None = None
    batch_size: int | None = None
    device: str | None = None
    log_every: int = 100
    # Output (like MAB_solutions)
    outdir: str = "plots_gpu"
    save_csv: bool = False
    debug_devices: bool = False
    profile: bool = False


def parse_args() -> Config:
    p = argparse.ArgumentParser("Bandit experiments (PufferLib)")
    # Problem
    p.add_argument("--algo", type=str, choices=["klucb", "ducb", "swucb"], default="klucb")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--runs", type=int, default=4096, help="number of parallel envs")
    p.add_argument("--seed", type=int, default=0)
    # Agent hyperparameters
    p.add_argument("--c", type=float, default=None)
    p.add_argument("--kl-alpha", type=float, default=None, dest="kl_alpha")
    p.add_argument("--discount", type=float, default=None)
    p.add_argument("--window", type=int, default=None)
    # Environment
    p.add_argument("--nonstationary", action="store_true")
    p.add_argument("--sigma", type=float, default=0.1)
    # Vectorization
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--log-every", type=int, default=100)
    # Output
    p.add_argument("--outdir", type=str, default="plots_gpu")
    p.add_argument("--save-csv", action="store_true")
    p.add_argument("--debug-devices", action="store_true")
    p.add_argument("--profile", action="store_true", help="print per-step timing breakdowns")
    # Backward-compatible aliases
    p.add_argument("--agent", type=str, choices=["klucb", "ducb", "swucb"], default=None)
    p.add_argument("--num-envs", type=int, default=None)

    args = p.parse_args()
    algo = args.algo if args.agent is None else args.agent
    runs = args.runs if args.num_envs is None else args.num_envs
    return Config(
        algo=algo,  # type: ignore
        k=args.k,
        T=args.T,
        runs=runs,
        seed=args.seed,
        c=args.c,
        kl_alpha=args.kl_alpha,
        discount=args.discount,
        window=args.window,
        nonstationary=args.nonstationary,
        sigma=args.sigma,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        device=args.device,
        log_every=args.log_every,
        outdir=args.outdir,
        save_csv=bool(args.save_csv),
        debug_devices=bool(args.debug_devices),
        profile=bool(args.profile),
    )


def build_envs(cfg: Config) -> Multiprocessing:
    # Use top-level class GymnasiumPufferEnv as creator; feed bandit args via env_kwargs
    env_creators = [GymnasiumPufferEnv for _ in range(cfg.runs)]
    env_args = [[] for _ in range(cfg.runs)]
    env_kwargs = [
        dict(env_creator=BernoulliBanditEnv, env_args=[], env_kwargs=dict(
            k=cfg.k, nonstationary=cfg.nonstationary, sigma=cfg.sigma
        ))
        for _ in range(cfg.runs)
    ]

    # Workers: default to min(num_envs, physical cores);
    # caller can override with --num-workers
    num_workers = cfg.num_workers
    if num_workers is None:
        num_workers = min(cfg.runs, CORES)
    # Clamp workers to [1, num_envs] to avoid division by zero in vectorizer
    num_workers = max(1, min(num_workers, cfg.runs))

    # Note: PufferLib enforces num_workers <= physical cores by default.
    # To avoid APIUsageError across platforms (e.g., os.cpu_count may exceed
    # physical cores), permit over-subscription unless the user overrides.
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
    acfg = AgentCfg(k=cfg.k, num_envs=cfg.runs, device=device)
    if cfg.algo == "klucb":
        return KLUCB(acfg, alpha=cfg.kl_alpha or 3.0)
    if cfg.algo == "ducb":
        return DiscountedUCB(acfg, c=cfg.c or 2.0, discount=cfg.discount or 0.99)
    if cfg.algo == "swucb":
        return SlidingWindowUCB(acfg, c=cfg.c or 2.0, window=cfg.window or 200)
    raise ValueError("unknown agent")


def main() -> None:
    cfg = parse_args()
    device = pick_device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    vec = build_envs(cfg)
    obs, infos = vec.reset(seed=cfg.seed)  # ignore obs

    agent = build_agent(cfg, device)
    agent.reset()
    if cfg.debug_devices:
        # Print initial devices and wrap agent for one-time logging
        from .devices_debug import (  # type: ignore
            _desc,
            print_agent_state_devices,
            wrap_agent_for_debug,
        )
        print("[device] torch backend device:", device)
        print_agent_state_devices(agent)
        # Observations are numpy on CPU via PufferLib
        print("[device] reset() -> obs:", _desc(obs))
        agent = wrap_agent_for_debug(agent)
    # Aggregators (streaming over time, per-step CI across runs)
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
    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            try:
                torch.mps.synchronize()
            except Exception:
                pass

    for t in range(1, T + 1):
        if cfg.profile:
            _sync()
            t0 = time.perf_counter()
            actions = agent.select_actions(t)
            _sync()
            prof["select"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            atn_np = actions.detach().cpu().numpy()
            prof["actions_to_cpu"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            _, r, _, _, infos = vec.step(atn_np)
            prof["env_step"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            rewards = torch.from_numpy(r).to(device=device, dtype=torch.float32)
            _sync()
            prof["rewards_to_device"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            agent.update(actions, rewards)
            _sync()
            prof["update"] += time.perf_counter() - t0
        else:
            actions = agent.select_actions(t)
            atn_np = actions.detach().cpu().numpy()
            _, r, _, _, infos = vec.step(atn_np)
            rewards = torch.from_numpy(r).to(device=device, dtype=torch.float32)
            agent.update(actions, rewards)
        if cfg.debug_devices and t == 1:
            # Show device flow at first step
            from .devices_debug import _desc  # type: ignore
            print("[device] env.step <- actions:", _desc(atn_np))
            print("[device] env.step -> rewards:", _desc(r))
            print("[device] agent.update <- rewards (torch):", _desc(rewards))
        # Map rewards directly (batch_size == runs so ordering is stable)
        r_np = r.astype(float)
        cumulative_reward += r_np

        # Extract optimal flags and p* from infos (aligned with rewards)
        if isinstance(infos, list) and infos:
            opt_flags = np.array([int(i.get("optimal", 0)) for i in infos], dtype=float)
            pstars = np.array([float(np.max(i.get("p", [np.nan]))) for i in infos], dtype=float)
        else:
            # Fallback if infos are missing
            opt_flags = np.zeros(n, dtype=float)
            pstars = np.full(n, np.nan, dtype=float)

        if not have_pstar0:
            # First-step p* snapshot for stationary regret
            pstar0 = np.where(np.isnan(pstars), pstar0, pstars)
            have_pstar0 = True

        if cfg.nonstationary:
            cumulative_pstar += np.nan_to_num(pstars, nan=0.0)

        # Reward CI
        m = float(r_np.mean())
        se = float(r_np.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
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
        if cfg.nonstationary:
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
                print(
                    "profile(ms/step):",
                    f"select={ms['select']:.3f}",
                    f"a2cpu={ms['actions_to_cpu']:.3f}",
                    f"step={ms['env_step']:.3f}",
                    f"r2dev={ms['rewards_to_device']:.3f}",
                    f"update={ms['update']:.3f}",
                )

    vec.close()

    # Plot/save like MAB_solutions
    metrics = {
        "mean_reward": mean_reward,
        "reward_lo": reward_lo,
        "reward_hi": reward_hi,
        "%_optimal": pct_opt,
        "%_optimal_lo": pct_opt_lo,
        "%_optimal_hi": pct_opt_hi,
        "cumulative_regret": cumulative_regret,
        "cumulative_regret_lo": cumulative_regret_lo,
        "cumulative_regret_hi": cumulative_regret_hi,
    }

    if plt is not None:
        os.makedirs(cfg.outdir, exist_ok=True)
        def tag() -> str:
            parts = [f"algo={cfg.algo}", f"k={cfg.k}", f"T={cfg.T}", f"runs={cfg.runs}", f"seed={cfg.seed}"]
            if cfg.c is not None:
                parts.append(f"c={cfg.c}")
            if cfg.nonstationary:
                parts.append(f"nonstat_sigma={cfg.sigma}")
            return "_".join(parts)
        x = np.arange(1, T + 1)
        plt.figure(figsize=(6,4))
        plt.plot(x, mean_reward, label="mean reward")
        plt.fill_between(x, reward_lo, reward_hi, alpha=0.2, label="95% CI")
        plt.xlabel("t"); plt.ylabel("Mean reward"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(cfg.outdir, f"reward_{tag()}.png")); plt.close()

        plt.figure(figsize=(6,4))
        plt.plot(x, pct_opt, label="% optimal")
        plt.fill_between(x, pct_opt_lo, pct_opt_hi, alpha=0.2, label="95% CI")
        plt.xlabel("t"); plt.ylabel("% optimal"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(cfg.outdir, f"pct_optimal_{tag()}.png")); plt.close()

        plt.figure(figsize=(6,4))
        plt.plot(x, cumulative_regret, label="cumulative regret")
        plt.fill_between(x, cumulative_regret_lo, cumulative_regret_hi, alpha=0.2, label="95% CI")
        plt.xlabel("t"); plt.ylabel("Cumulative regret"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(cfg.outdir, f"regret_{tag()}.png")); plt.close()
    else:
        print("Warning: matplotlib not available; skipping plots")

    if cfg.save_csv:
        os.makedirs(cfg.outdir, exist_ok=True)
        path = os.path.join(cfg.outdir, "summary.csv")
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if f.tell() == 0:
                w.writerow(["algo","k","T","runs","seed","c","nonstationary","sigma","final_mean_reward","final_%_optimal","final_cumulative_regret"])
            w.writerow([
                cfg.algo, cfg.k, T, cfg.runs, cfg.seed, cfg.c, cfg.nonstationary, cfg.sigma,
                float(mean_reward[-1]), float(pct_opt[-1]), float(cumulative_regret[-1])
            ])

    print(f"Mean reward (first 5): {np.round(mean_reward[:5], 3)}")
    print(f"% optimal (first 5): {np.round(pct_opt[:5], 1)}")
    if cfg.profile:
        ms = {k: 1000.0 * v / T for k, v in prof.items()}
        print(
            "Final profile(ms/step):",
            f"select={ms['select']:.3f}",
            f"a2cpu={ms['actions_to_cpu']:.3f}",
            f"step={ms['env_step']:.3f}",
            f"r2dev={ms['rewards_to_device']:.3f}",
            f"update={ms['update']:.3f}",
        )


if __name__ == "__main__":
    main()

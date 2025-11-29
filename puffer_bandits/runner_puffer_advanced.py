
import argparse
import os
import time
import csv
from dataclasses import dataclass, asdict
from typing import Literal

import numpy as np
import torch

CORES = os.cpu_count() or 1

from pufferlib.emulation import GymnasiumPufferEnv
from pufferlib.vector import Multiprocessing
from omegaconf import OmegaConf  # type: ignore

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
    # UI
    tui: bool = False
    # Output
    outdir: str = "plots"
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
    p.add_argument("--algo", type=str, choices=["linucb", "lints", "exp3", "exp3ix", "neuralts", "neurallinear"], default=None)
    p.add_argument("--env", type=str, choices=["contextual", "bernoulli"], default=None)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--d", type=int, default=None)
    p.add_argument("--T", type=int, default=None)
    p.add_argument("--runs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--nonstationary", action="store_true")
    p.add_argument("--sigma", type=float, default=None, help="non-stationary reward sigma (bernoulli)")
    p.add_argument("--theta-sigma", type=float, default=None, help="non-stationary theta sigma (contextual)")
    p.add_argument("--x-sigma", type=float, default=None)
    # Agent params
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--lam", type=float, default=None)
    p.add_argument("--v", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--eta", type=float, default=None)
    p.add_argument("--hidden", type=int, default=None)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--ensembles", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--features", type=int, default=None, help="NeuralLinear feature dim m")
    p.add_argument("--linlam", type=float, default=None, help="NeuralLinear ridge lam")
    p.add_argument("--linv", type=float, default=None, help="NeuralLinear TS scale v")
    # Vectorization
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--log-every", type=int, default=None)
    # Output
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--save-csv", action="store_true")
    p.add_argument("--debug-devices", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--force-device", action="store_true")
    p.add_argument("--no-amp", action="store_false", dest="amp")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--tui", action="store_true", help="Rich TUI console dashboard (no matplotlib)")
    # Weights & Biases (optional)
    p.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases")
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-tags", type=str, default=None, help=",-separated tags")
    p.add_argument("--wandb-offline", action="store_true")
    p.add_argument("--run-name", type=str, default=None)
    # Config-driven
    p.add_argument("--config", type=str, default=None, help="Path to YAML/TOML config")
    p.add_argument("--set", action="append", default=None, help="Override config with dotlist, e.g., runs=1024")
    args = p.parse_args()
    base = asdict(Config())
    conf = OmegaConf.create(base)
    if args.config:
        conf = OmegaConf.merge(conf, OmegaConf.load(args.config))
    if args.set:
        conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(args.set))
    for key, val in vars(args).items():
        if key in {"config", "set"}:
            continue
        if isinstance(val, bool):
            if val:
                conf[key] = val
        elif val is not None:
            conf[key] = val
    # Enforce: TUI is CLI-only (ignore config file value)
    conf["tui"] = bool(getattr(args, "tui", False))
    cfg_dict = OmegaConf.to_container(conf, resolve=True)  # type: ignore
    return Config(**cfg_dict)  # type: ignore[arg-type]


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

    if cfg.num_workers is None:
        candidate = min(cfg.runs, CORES)
        while candidate > 1 and (cfg.runs % candidate) != 0:
            candidate -= 1
        num_workers = max(1, candidate)
    else:
        num_workers = max(1, min(cfg.num_workers, cfg.runs))
        if (cfg.runs % num_workers) != 0:
            print(f"[vector] runs={cfg.runs} not divisible by num_workers={num_workers}; using 1 worker.")
            num_workers = 1
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


def run_with_config(cfg: Config) -> None:
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
    # Explicit per-agent RNG seed for determinism across devices
    try:
        agent.rng = torch.Generator(device=device)
        agent.rng.manual_seed(int(cfg.seed))
    except Exception:
        pass
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

    # Optional TUI accumulators
    tui = None
    cum_counts = np.zeros((cfg.k,), dtype=int)
    cum_regret_by_arm = np.zeros((cfg.k,), dtype=float)
    cum_regret_total = 0.0
    ewma_ms: float | None = None
    last_ms: float | None = None
    if cfg.tui:
        try:
            from .ui.tui import RichTUI  # type: ignore
            tui = RichTUI(cfg.k, cfg.T, f"{device}")
        except Exception:
            tui = None

    # Optional Weights & Biases
    from .utils.wandb import wb_init, wb_log, wb_finish
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
    wb = wb_init(
        enabled=cfg.wandb,
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        tags=cfg.wandb_tags,
        run_name=run_name,
        offline=cfg.wandb_offline,
        config=wb_cfg,
    )

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
            t0_np = time.perf_counter()
            obs, r, _, _, infos = vec.step(atn_np)
            rewards = torch.from_numpy(r.astype(np.float32)).to(device=device)
            if cfg.env == "contextual":
                agent.update(actions, rewards, obs_t)
            else:
                agent.update(actions, rewards)
            last_ms = (time.perf_counter() - t0_np) * 1000.0
            if ewma_ms is None:
                ewma_ms = last_ms
            else:
                ewma_ms = 0.9 * ewma_ms + 0.1 * last_ms

        if t % cfg.log_every == 0:
            mean_r = float(np.mean(r))
            print(f"t={t} mean_reward={mean_r:.4f}")
            if wb is not None:
                wb_log(wb, {"t": t, "mean_reward": mean_r}, step=t)
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
            if tui is not None:
                # % optimal and regret, cum counts
                pct = 0.0
                regret_step = 0.0
                if isinstance(infos, list) and infos:
                    optimal_flags = []
                    for i, inf in enumerate(infos):
                        opt = int(inf.get("optimal", 0)) if isinstance(inf, dict) else 0
                        optimal_flags.append(opt)
                        p_vec = inf.get("p") if isinstance(inf, dict) else None
                        if p_vec is not None:
                            try:
                                pstar = float(np.max(p_vec))
                                a_i = int(atn_np[i])
                                p_sel = float(p_vec[a_i])
                                regret_i = max(0.0, pstar - p_sel)
                                regret_step += regret_i
                                if 0 <= a_i < cfg.k:
                                    cum_regret_by_arm[a_i] += regret_i
                            except Exception:
                                pass
                    if len(optimal_flags) > 0:
                        pct = float(np.mean(optimal_flags) * 100.0)
                cum_regret_total += regret_step
                cum_counts += np.bincount(atn_np, minlength=cfg.k)
                try:
                    speed_sps = None if (last_ms is None) else (1000.0 / last_ms if last_ms > 0 else None)
                    mem = memory_stats(device)
                    tui.update(
                        t=t,
                        mean_r=mean_r,
                        pct_opt=pct,
                        regret=float(cum_regret_total),
                        actions=atn_np.tolist(),
                        mem=mem,
                        values=None,
                        speed_sps=speed_sps,
                        cum_counts=cum_counts.tolist(),
                        cum_regret_by_arm=cum_regret_by_arm.tolist(),
                        last_ms=last_ms,
                        ewma_ms=ewma_ms,
                        values_labels=None,
                    )
                except Exception:
                    pass

    vec.close()

    # Optionally print memory stats at end
    if cfg.debug_devices:
        print(memory_stats())
    if tui is not None:
        try:
            tui.stop()
        except Exception:
            pass
    if wb is not None:
        wb_finish(wb)


def main() -> None:
    cfg = parse_args()
    run_with_config(cfg)


if __name__ == "__main__":
    main()

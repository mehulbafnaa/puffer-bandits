
import argparse
import os
import time
import csv
from dataclasses import dataclass, asdict
from typing import Literal

import numpy as np
import torch
from omegaconf import OmegaConf  # type: ignore

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
from .utils.device import pick_device, memory_stats
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
    outdir: str = "plots"
    save_csv: bool = False
    debug_devices: bool = False
    profile: bool = False
    tui: bool = False
    # Logging (optional)
    wandb: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_tags: str | None = None
    wandb_offline: bool = False
    run_name: str | None = None


def parse_args() -> Config:
    p = argparse.ArgumentParser("Bandit experiments (PufferLib)")
    # Problem
    p.add_argument("--algo", type=str, choices=["klucb", "ducb", "swucb"], default=None)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--T", type=int, default=None)
    p.add_argument("--runs", type=int, default=None, help="number of parallel envs")
    p.add_argument("--seed", type=int, default=None)
    # Agent hyperparameters
    p.add_argument("--c", type=float, default=None)
    p.add_argument("--kl-alpha", type=float, default=None, dest="kl_alpha")
    p.add_argument("--discount", type=float, default=None)
    p.add_argument("--window", type=int, default=None)
    # Environment
    p.add_argument("--nonstationary", action="store_true")
    p.add_argument("--sigma", type=float, default=None)
    # Vectorization
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--log-every", type=int, default=None)
    # Output
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--save-csv", action="store_true")
    p.add_argument("--debug-devices", action="store_true")
    p.add_argument("--profile", action="store_true", help="print per-step timing breakdowns")
    p.add_argument("--tui", action="store_true", help="Rich TUI console dashboard (no matplotlib)")
    # Weights & Biases (optional)
    p.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases")
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-tags", type=str, default=None, help=",-separated tags")
    p.add_argument("--wandb-offline", action="store_true")
    p.add_argument("--run-name", type=str, default=None)
    # Backward-compatible aliases
    p.add_argument("--agent", type=str, choices=["klucb", "ducb", "swucb"], default=None)
    p.add_argument("--num-envs", type=int, default=None)
    # Config-driven
    p.add_argument("--config", type=str, default=None, help="Path to YAML/TOML config")
    p.add_argument("--set", action="append", default=None, help="Override config with dotlist, e.g., runs=1024")

    args = p.parse_args()

    # Build config: defaults -> file -> dotlist -> CLI explicit overrides (incl. aliases)
    conf = OmegaConf.create(asdict(Config()))
    if args.config:
        conf = OmegaConf.merge(conf, OmegaConf.load(args.config))
    if args.set:
        conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(args.set))

    # Map aliases into the overlay dict
    overlay = dict(vars(args))
    if overlay.get("agent") is not None and overlay.get("algo") is None:
        overlay["algo"] = overlay["agent"]
    if overlay.get("num_envs") is not None and overlay.get("runs") is None:
        overlay["runs"] = overlay["num_envs"]

    for key, val in overlay.items():
        if key in {"config", "set", "agent", "num_envs"}:
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
        # Choose a default that divides runs if possible
        candidate = min(cfg.runs, CORES)
        while candidate > 1 and (cfg.runs % candidate) != 0:
            candidate -= 1
        num_workers = max(1, candidate)
    # Clamp and guard divisibility
    num_workers = max(1, min(num_workers, cfg.runs))
    if (cfg.runs % num_workers) != 0:
        print(f"[vector] runs={cfg.runs} not divisible by num_workers={num_workers}; using 1 worker.")
        num_workers = 1

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


def run_with_config(cfg: Config) -> None:
    device = pick_device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    vec = build_envs(cfg)
    obs, infos = vec.reset(seed=cfg.seed)  # ignore obs

    agent = build_agent(cfg, device)
    # Explicit per-agent RNG seed for determinism across devices
    try:
        agent.rng = torch.Generator(device=device)
        agent.rng.manual_seed(int(cfg.seed))
    except Exception:
        pass
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

    # Optional Weights & Biases
    from .utils.wandb import wb_init, wb_log, wb_finish
    run_name = cfg.run_name or f"classic-{cfg.algo}-k{cfg.k}-T{cfg.T}-n{cfg.runs}-{device}-{cfg.seed}"
    wb_cfg = {
        "runner": "classic",
        "algo": cfg.algo,
        "k": cfg.k,
        "T": cfg.T,
        "runs": cfg.runs,
        "device": str(device),
        "seed": cfg.seed,
        "nonstationary": cfg.nonstationary,
        "sigma": cfg.sigma,
        "outdir": cfg.outdir,
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

    # Optional TUI accumulators
    tui = None
    if cfg.tui:
        try:
            from .ui.tui import RichTUI  # type: ignore
            tui = RichTUI(cfg.k, cfg.T, f"{device}")
        except Exception:
            tui = None
    cum_counts = np.zeros((cfg.k,), dtype=int)
    cum_regret_by_arm = np.zeros((cfg.k,), dtype=float)
    cum_regret_total = 0.0
    ewma_ms: float | None = None
    last_ms: float | None = None

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
            t0_np = time.perf_counter()
            actions = agent.select_actions(t)
            atn_np = actions.detach().cpu().numpy()
            _, r, _, _, infos = vec.step(atn_np)
            rewards = torch.from_numpy(r).to(device=device, dtype=torch.float32)
            agent.update(actions, rewards)
            last_ms = (time.perf_counter() - t0_np) * 1000.0
            if ewma_ms is None:
                ewma_ms = last_ms
            else:
                ewma_ms = 0.9 * ewma_ms + 0.1 * last_ms
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
            if wb is not None:
                wb_log(wb, {"t": t, "mean_reward": m, "%_optimal": pct, "cumulative_regret": rg_m}, step=t)
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
                # Per-step regret breakdown for TUI
                regret_step = 0.0
                if isinstance(infos, list) and infos:
                    for i, inf in enumerate(infos):
                        p_vec = inf.get("p") if isinstance(inf, dict) else None
                        if p_vec is not None:
                            try:
                                pstar = float(np.max(p_vec))
                                a_i = int(atn_np[i])
                                p_sel = float(p_vec[a_i])
                                r_i = max(0.0, pstar - p_sel)
                                regret_step += r_i
                                if 0 <= a_i < cfg.k:
                                    cum_regret_by_arm[a_i] += r_i
                            except Exception:
                                pass
                cum_regret_total += regret_step
                cum_counts += np.bincount(atn_np, minlength=cfg.k)
                try:
                    mem = memory_stats(device)
                    speed_sps = None if last_ms is None else (1000.0 / last_ms if last_ms > 0 else None)
                    tui.update(
                        t=t,
                        mean_r=m,
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
        reward_path = os.path.join(cfg.outdir, f"reward_{tag()}.png")
        plt.savefig(reward_path); plt.close()

        plt.figure(figsize=(6,4))
        plt.plot(x, pct_opt, label="% optimal")
        plt.fill_between(x, pct_opt_lo, pct_opt_hi, alpha=0.2, label="95% CI")
        plt.xlabel("t"); plt.ylabel("% optimal"); plt.legend(); plt.tight_layout()
        pct_path = os.path.join(cfg.outdir, f"pct_optimal_{tag()}.png")
        plt.savefig(pct_path); plt.close()

        plt.figure(figsize=(6,4))
        plt.plot(x, cumulative_regret, label="cumulative regret")
        plt.fill_between(x, cumulative_regret_lo, cumulative_regret_hi, alpha=0.2, label="95% CI")
        plt.xlabel("t"); plt.ylabel("Cumulative regret"); plt.legend(); plt.tight_layout()
        regret_path = os.path.join(cfg.outdir, f"regret_{tag()}.png")
        plt.savefig(regret_path); plt.close()

        # Log plots to W&B if enabled
        if wb is not None:
            try:
                import wandb as _wandb  # type: ignore
                wb_log(wb, {
                    "plots/reward": _wandb.Image(reward_path),
                    "plots/pct_optimal": _wandb.Image(pct_path),
                    "plots/regret": _wandb.Image(regret_path),
                })
            except Exception:
                pass
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
        if wb is not None:
            try:
                import wandb as _wandb  # type: ignore
                art_name = f"summary-{cfg.algo}-k{cfg.k}-T{T}-n{cfg.runs}-s{cfg.seed}"
                art = _wandb.Artifact(art_name, type="metrics")
                art.add_file(path)
                _wandb.log_artifact(art)
            except Exception:
                pass

    print(f"Mean reward (first 5): {np.round(mean_reward[:5], 3)}")
    print(f"% optimal (first 5): {np.round(pct_opt[:5], 1)}")
    if tui is not None:
        try:
            tui.stop()
        except Exception:
            pass
    if wb is not None:
        wb_finish(wb)
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


def main() -> None:
    cfg = parse_args()
    run_with_config(cfg)


if __name__ == "__main__":
    main()

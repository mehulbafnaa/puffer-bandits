
import argparse
import os
import sys
from dataclasses import dataclass, asdict
from typing import Literal
import time

import numpy as np
import torch
from pufferlib.vector import Multiprocessing, Serial
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
from .agents import KLUCB, AgentCfg, DiscountedUCB, SlidingWindowUCB
from .puffer_envs import PufferBernoulliBandit, PufferContextualBandit, PufferDatasetContextualBandit
from .utils.device import pick_device, memory_stats


# Preset configurations with smart defaults (MPS + MP + TUI)
PRESETS = {
    "smoke": {
        # Fast validation - CPU friendly but still use MPS
        "k": 5, "T": 100, "runs": 64, "d": 4,
        "device": "mps", "vector": "mp", "num_workers": 4, 
        "tui": True, "log_every": 20
    },
    "experiment": {  
        # Standard research - optimized for Mac M-series
        "k": 10, "T": 2000, "runs": 1024, "d": 8,
        "device": "mps", "vector": "mp", "num_workers": 8,
        "tui": True, "log_every": 100
    },
    "benchmark": {
        # Heavy computational - max throughput
        "k": 20, "T": 5000, "runs": 2000, "d": 16, 
        "device": "mps", "vector": "mp", "num_workers": 10,
        "tui": True, "log_every": 200
    },
    "neural": {
        # Neural-optimized - fewer runs, more memory per model
        "k": 10, "T": 1000, "runs": 256, "d": 8,
        "device": "mps", "vector": "mp", "num_workers": 8,
        "tui": True, "log_every": 50,
        "ensembles": 5, "hidden": 256, "depth": 3
    }
}


@dataclass
class Config:
    env: Literal["contextual", "bernoulli"] = "contextual"
    # New dataset option behaves like contextual with data-backed obs
    # (use --env dataset and provide --data-path)
    # NOTE: keep k,d consistent with dataset shape
    # Added flags below
    
    algo: str = "linucb"
    k: int = 10
    d: int = 8
    T: int = 1000
    runs: int = 512
    seed: int = 0
    # Contextual params
    theta_sigma: float = 0.05
    x_sigma: float = 1.0
    # Bernoulli params
    nonstationary: bool = False
    sigma: float = 0.1
    # Agents
    alpha: float = 1.0
    lam: float = 1.0
    v: float = 0.1
    gamma: float = 0.07
    eta: float | None = None
    hidden: int = 128
    depth: int = 2
    ensembles: int = 3
    dropout: float = 0.1
    lr: float = 1e-3
    features: int = 64
    linlam: float = 1.0
    linv: float = 0.1
    # Vectorization
    vector: Literal["mp", "serial"] = "mp"
    num_workers: int | None = None
    batch_size: int | None = None
    device: str | None = None
    force_device: bool = False
    amp: bool = True
    log_every: int = 100
    tui: bool = False
    # Dataset-backed env
    data_path: str | None = None
    x_key: str = "X"
    p_key: str = "P"
    y_key: str = "Y"
    # Logging (optional)
    wandb: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_tags: str | None = None
    wandb_offline: bool = False
    run_name: str | None = None


def parse_args() -> Config:
    p = argparse.ArgumentParser("Native PufferLib runner for puffer-bandits")
    p.add_argument("--preset", type=str, choices=list(PRESETS.keys()), default=None, 
                   help="Use preset configuration (smoke, experiment, benchmark, neural)")
    p.add_argument("--env", type=str, choices=["contextual", "bernoulli", "dataset"], default=None)
    p.add_argument("--algo", type=str, default=None)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--d", type=int, default=None)
    p.add_argument("--T", type=int, default=None)
    p.add_argument("--runs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--theta-sigma", type=float, default=None)
    p.add_argument("--x-sigma", type=float, default=None)
    p.add_argument("--nonstationary", action="store_true")
    p.add_argument("--sigma", type=float, default=None)
    # Agents
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
    p.add_argument("--features", type=int, default=None)
    p.add_argument("--linlam", type=float, default=None)
    p.add_argument("--linv", type=float, default=None)
    # Vectorization
    p.add_argument("--vector", type=str, choices=["mp", "serial"], default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--force-device", action="store_true")
    p.add_argument("--no-amp", action="store_false", dest="amp")
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--tui", action="store_true", help="Rich TUI console dashboard (no matplotlib)")
    # Dataset args (used when --env dataset)
    p.add_argument("--data-path", type=str, default=None, help="Path to NPZ with keys X and (P or Y)")
    p.add_argument("--x-key", type=str, default=None)
    p.add_argument("--p-key", type=str, default=None)
    p.add_argument("--y-key", type=str, default=None)
    # Weights & Biases (optional)
    p.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases")
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-tags", type=str, default=None, help=",-separated tags")
    p.add_argument("--wandb-offline", action="store_true")
    p.add_argument("--run-name", type=str, default=None)

    # Config-driven
    p.add_argument("--config", type=str, default=None, help="Path to YAML/TOML config")
    p.add_argument("--set", action="append", default=None, help="Override config with dotlist, e.g., sizes.runs=1024")
    args = p.parse_args()
    
    # Build config: defaults -> preset -> file -> dotlist -> CLI (explicit only)
    base = asdict(Config())
    conf = OmegaConf.create(base)
    if args.preset:
        if args.preset not in PRESETS:
            raise ValueError(f"Unknown preset: {args.preset}")
        conf = OmegaConf.merge(conf, OmegaConf.create(PRESETS[args.preset]))
        print(f"[preset] Using '{args.preset}' preset with smart defaults")
    if args.config:
        conf = OmegaConf.merge(conf, OmegaConf.load(args.config))
    if args.set:
        conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(args.set))

    # Overlay explicit CLI arguments
    for key, val in vars(args).items():
        if key in {"preset", "config", "set"}:
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


def build_envs(cfg: Config):
    # Choose vectorizer
    if cfg.vector == "serial":
        Vec = Serial
    else:
        Vec = Multiprocessing
    # Build env kwargs
    env_args = [[] for _ in range(cfg.runs)]
    if cfg.env == "contextual":
        envs = [PufferContextualBandit for _ in range(cfg.runs)]
        env_kwargs = [dict(k=cfg.k, d=cfg.d, nonstationary=cfg.nonstationary, theta_sigma=cfg.theta_sigma, x_sigma=cfg.x_sigma) for _ in range(cfg.runs)]
    elif cfg.env == "bernoulli":
        envs = [PufferBernoulliBandit for _ in range(cfg.runs)]
        env_kwargs = [dict(k=cfg.k, nonstationary=cfg.nonstationary, sigma=cfg.sigma) for _ in range(cfg.runs)]
    else:
        # dataset-backed contextual
        envs = [PufferDatasetContextualBandit for _ in range(cfg.runs)]
        if not cfg.data_path or not os.path.exists(cfg.data_path):
            raise FileNotFoundError(f"Dataset NPZ not found: {cfg.data_path}")
        env_kwargs = [dict(k=cfg.k, d=cfg.d, data_path=cfg.data_path, x_key=cfg.x_key, p_key=cfg.p_key, y_key=cfg.y_key) for _ in range(cfg.runs)]

    if cfg.num_workers is None:
        cores = os.cpu_count() or 1
        candidate = min(cfg.runs, cores)
        while candidate > 1 and (cfg.runs % candidate) != 0:
            candidate -= 1
        num_workers = max(1, candidate)
    else:
        num_workers = max(1, min(cfg.num_workers, cfg.runs))
        if (cfg.runs % num_workers) != 0:
            print(f"[vector] runs={cfg.runs} not divisible by num_workers={num_workers}; using 1 worker.")
            num_workers = 1

    return Vec(
        env_creators=envs,
        env_args=env_args,
        env_kwargs=env_kwargs,
        num_envs=cfg.runs,
        num_workers=num_workers,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        overwork=True,
    )


def build_agent(cfg: Config, device: torch.device):
    if cfg.env == "contextual":
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
            return NeuralTS(acfg, hidden=cfg.hidden, depth=cfg.depth, ensembles=cfg.ensembles, dropout=cfg.dropout, lr=cfg.lr, amp=cfg.amp)
        if cfg.algo == "neurallinear":
            return NeuralLinearTS(acfg, m=cfg.features, hidden=cfg.hidden, depth=cfg.depth, dropout=cfg.dropout, lam=cfg.linlam, v=cfg.linv, lr=cfg.lr, amp=cfg.amp)
    else:
        acfg = AgentCfg(k=cfg.k, num_envs=cfg.runs, device=device)
        if cfg.algo == "klucb":
            return KLUCB(acfg, alpha=3.0)
        if cfg.algo == "ducb":
            return DiscountedUCB(acfg, c=2.0, discount=0.99)
        if cfg.algo == "swucb":
            return SlidingWindowUCB(acfg, c=2.0, window=200)
    raise ValueError("Unknown env/algo combination")


def run_with_config(cfg: Config) -> None:
    device = pick_device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    vec = build_envs(cfg)
    obs, infos = vec.reset(seed=cfg.seed)

    agent = build_agent(cfg, device)
    # Explicit per-agent RNG seed for determinism across devices
    try:
        agent.rng = torch.Generator(device=device)
        agent.rng.manual_seed(int(cfg.seed))
    except Exception:
        pass
    agent.reset()

    T = cfg.T
    n = cfg.runs
    # Preallocate small buffers to avoid per-step allocations/copies
    rewards_t = torch.empty((n,), device=device, dtype=torch.float32)
    obs_buf = None
    if cfg.env == "contextual":
        obs_buf = torch.empty((n, cfg.k, max(1, cfg.d)), device=device, dtype=torch.float32)

    # Optional Rich TUI
    tui = None
    if cfg.tui:
        try:
            from .ui.tui import RichTUI
            tui = RichTUI(cfg.k, cfg.T, f"{device}")
        except Exception:
            tui = None

    # Stats for TUI
    cum_counts = np.zeros((cfg.k,), dtype=int)
    cum_regret_by_arm = np.zeros((cfg.k,), dtype=float)
    cum_regret_total = 0.0
    ewma_ms: float | None = None

    # Optional Weights & Biases
    from .utils.wandb import wb_init, wb_log, wb_finish
    run_name = cfg.run_name or f"native-{cfg.env}-{cfg.algo}-k{cfg.k}-d{cfg.d}-n{cfg.runs}-{device}-{cfg.seed}"
    wb_cfg = {
        "runner": "native",
        "env": cfg.env,
        "algo": cfg.algo,
        "k": cfg.k,
        "d": cfg.d,
        "T": cfg.T,
        "runs": cfg.runs,
        "device": str(device),
        "seed": cfg.seed,
        "vector": cfg.vector,
        "num_workers": cfg.num_workers,
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
        if cfg.env == "contextual":
            if isinstance(obs, np.ndarray):
                # Copy into preallocated buffer
                assert obs_buf is not None
                obs_buf.copy_(torch.from_numpy(obs).to(device=device, dtype=torch.float32))
                actions = agent.select_actions(t, obs_buf)
            else:
                actions = agent.select_actions(t)
        else:
            actions = agent.select_actions(t)
        t0 = time.perf_counter()
        atn_np = actions.detach().cpu().numpy()
        obs, r, _, _, infos = vec.step(atn_np)
        # Copy rewards into preallocated buffer
        rewards_t.copy_(torch.from_numpy(r.astype(np.float32)).to(device=device))
        if cfg.env == "contextual":
            assert obs_buf is not None
            agent.update(actions, rewards_t, obs_buf)  # type: ignore[arg-type]
        else:
            agent.update(actions, rewards_t)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        last_ms = dt_ms
        if ewma_ms is None:
            ewma_ms = last_ms
        else:
            ewma_ms = 0.9 * ewma_ms + 0.1 * last_ms

        if tui is not None and (t % cfg.log_every == 0):
            mean_r = float(np.mean(r))
            # % optimal and regret from infos if available
            pct_opt = 0.0
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
                            cum_regret_by_arm[a_i] += regret_i
                        except Exception:
                            pass
                if len(optimal_flags) > 0:
                    pct_opt = float(np.mean(optimal_flags) * 100.0)
            cum_regret_total += regret_step
            cum_counts += np.bincount(atn_np, minlength=cfg.k)

            mem = memory_stats(device)
            speed_sps = 1000.0 / last_ms if last_ms > 0 else None
            # Extra diagnostics (cheap)
            se_reward = float(np.std(r, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            conf_width = None
            entropy = None
            explore_ratio = None
            # Contextual confidence width (LinUCB/LinTS)
            try:
                if cfg.env == "contextual" and isinstance(agent, LinUCB):
                    assert obs_buf is not None
                    batch = torch.arange(n, device=device)
                    x_sel = obs_buf[batch, actions.long(), :]
                    Ainv_sel = agent.A_inv[batch, actions.long(), :, :]
                    x = x_sel.unsqueeze(-1)  # (n,d,1)
                    Ax = torch.matmul(Ainv_sel, x)  # (n,d,1)
                    quad = torch.matmul(x.transpose(1, 2), Ax).squeeze(-1).squeeze(-1)  # (n,)
                    conf_width = float((cfg.alpha * torch.sqrt(quad.clamp_min(1e-12))).mean().item())
                elif cfg.env == "contextual" and isinstance(agent, LinTS):
                    assert obs_buf is not None
                    batch = torch.arange(n, device=device)
                    x_sel = obs_buf[batch, actions.long(), :]
                    Ainv_sel = agent.A_inv[batch, actions.long(), :, :]
                    x = x_sel.unsqueeze(-1)
                    Ax = torch.matmul(Ainv_sel, x)
                    quad = torch.matmul(x.transpose(1, 2), Ax).squeeze(-1).squeeze(-1)
                    conf_width = float((cfg.v * torch.sqrt(quad.clamp_min(1e-12))).mean().item())
            except Exception:
                conf_width = None
            # Adversarial policy entropy
            try:
                if isinstance(agent, EXP3) or isinstance(agent, EXP3IX):
                    p = getattr(agent, "_last_p", None)
                    if p is not None:
                        entropy = float((-(p * (p.clamp_min(1e-12)).log()).sum(dim=1)).mean().item())
            except Exception:
                entropy = None
            # Non-contextual "cold arms" ratio
            try:
                if isinstance(agent, KLUCB):
                    explore_ratio = float((agent.N == 0).float().mean().item())
                elif isinstance(agent, DiscountedUCB):
                    explore_ratio = float((agent.Neff <= 1e-8).float().mean().item())
                elif isinstance(agent, SlidingWindowUCB):
                    explore_ratio = float((agent.Nw <= 1e-8).float().mean().item())
            except Exception:
                explore_ratio = None

            vector_info = {
                "runs": cfg.runs,
                "workers": cfg.num_workers if cfg.num_workers is not None else "auto",
                "vector": cfg.vector,
            }
            algo_info = {}
            for key in ("alpha", "lam", "v", "gamma", "eta", "features", "hidden", "depth", "ensembles"):
                val = getattr(cfg, key, None)
                if val is not None:
                    algo_info[key] = val
            # Build per-arm value estimates for env 0 when available
            values = None
            values_labels = None
            try:
                values_rows = []
                true_p0 = None
                if isinstance(infos, list) and len(infos) > 0 and isinstance(infos[0], dict):
                    p_vec0 = infos[0].get("p")
                    if p_vec0 is not None:
                        true_p0 = np.asarray(p_vec0, dtype=float)
                if cfg.env == "contextual" and (isinstance(agent, LinUCB) or isinstance(agent, LinTS)):
                    assert obs_buf is not None
                    mu = torch.matmul(agent.A_inv, agent.b.unsqueeze(-1)).squeeze(-1)  # (n,k,d)
                    mu0 = mu[0]
                    x0 = obs_buf[0]
                    est = torch.sigmoid((x0 * mu0).sum(dim=-1))  # (k,)
                    # Confidence per arm for readability
                    Ainv0 = agent.A_inv[0]
                    x = x0.unsqueeze(-1)  # (k,d,1)
                    Ax = torch.matmul(Ainv0, x).squeeze(-1)
                    quad = (x0 * Ax).sum(dim=-1).clamp_min(1e-12)
                    conf = (cfg.alpha if isinstance(agent, LinUCB) else cfg.v) * torch.sqrt(quad)
                    est_np = est.detach().cpu().numpy()
                    conf_np = conf.detach().cpu().numpy()
                    order = np.argsort(-est_np)
                    for i in order[:6]:
                        tp = None if true_p0 is None else float(true_p0[int(i)])
                        values_rows.append((int(i), float(est_np[int(i)]), tp, float(conf_np[int(i)])))
                    values = values_rows
                    values_labels = {"est": "est p", "true": "true p", "extra": "conf"}
                elif isinstance(agent, EXP3) or isinstance(agent, EXP3IX):
                    p = getattr(agent, "_last_p", None)
                    if p is not None:
                        p0 = p[0].detach().cpu().numpy()
                        order = np.argsort(-p0)
                        for i in order[:6]:
                            tp = None if true_p0 is None else float(true_p0[int(i)])
                            values_rows.append((int(i), float(p0[int(i)]), tp))
                        values = values_rows
                        values_labels = {"est": "policy p", "true": "true p"}
                elif cfg.env == "bernoulli":
                    if isinstance(agent, KLUCB):
                        Q0 = agent.Q[0].detach().cpu().numpy()
                        u0 = agent._ucb(t)[0].detach().cpu().numpy()
                        w0 = (u0 - Q0)
                        order = np.argsort(-Q0)
                        for i in order[:6]:
                            tp = None if true_p0 is None else float(true_p0[int(i)])
                            values_rows.append((int(i), float(Q0[int(i)]), tp, float(w0[int(i)])))
                        values = values_rows
                        values_labels = {"est": "Q", "true": "true p", "extra": "width"}
                    elif isinstance(agent, DiscountedUCB):
                        S0 = agent.S[0].detach().cpu().numpy()
                        N0 = agent.Neff[0].detach().cpu().numpy()
                        Q0 = np.divide(S0, np.maximum(N0, 1e-8))
                        width = agent.c * np.sqrt(np.log(max(1, t)) / np.maximum(N0, 1e-8))
                        order = np.argsort(-Q0)
                        for i in order[:6]:
                            tp = None if true_p0 is None else float(true_p0[int(i)])
                            values_rows.append((int(i), float(Q0[int(i)]), tp, float(width[int(i)])))
                        values = values_rows
                        values_labels = {"est": "Qd", "true": "true p", "extra": "width"}
                    elif isinstance(agent, SlidingWindowUCB):
                        S0 = agent.Sw[0].detach().cpu().numpy()
                        N0 = agent.Nw[0].detach().cpu().numpy()
                        Q0 = np.divide(S0, np.maximum(N0, 1e-8))
                        eff_t = max(1, min(t, int(agent.window)))
                        width = agent.c * np.sqrt(np.log(eff_t) / np.maximum(N0, 1e-8))
                        order = np.argsort(-Q0)
                        for i in order[:6]:
                            tp = None if true_p0 is None else float(true_p0[int(i)])
                            values_rows.append((int(i), float(Q0[int(i)]), tp, float(width[int(i)])))
                        values = values_rows
                        values_labels = {"est": "Qw", "true": "true p", "extra": "width"}
            except Exception:
                values = None
                values_labels = None
            try:
                tui.update(
                    t=t,
                    mean_r=mean_r,
                    pct_opt=pct_opt,
                    regret=float(cum_regret_total),
                    actions=atn_np.tolist(),
                    mem=mem,
                    values=values,
                    speed_sps=speed_sps,
                    cum_counts=cum_counts.tolist(),
                    cum_regret_by_arm=cum_regret_by_arm.tolist(),
                    last_ms=last_ms,
                    ewma_ms=ewma_ms,
                    values_labels=values_labels,
                    se_reward=se_reward,
                    conf_width=conf_width,
                    entropy=entropy,
                    explore_ratio=explore_ratio,
                    vector_info=vector_info,
                    algo_info=algo_info,
                )
            except Exception:
                pass
        if wb is not None and (t % cfg.log_every == 0):
            wb_log(
                wb,
                {
                    "t": t,
                    "mean_reward": float(np.mean(r)),
                    "%_optimal": pct_opt,
                    "cumulative_regret": float(cum_regret_total),
                    "last_ms": last_ms,
                    "ewma_ms": ewma_ms,
                },
                step=t,
            )

    vec.close()
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

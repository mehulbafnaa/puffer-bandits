from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Literal
import time

import numpy as np
import torch
from pufferlib.vector import Multiprocessing, Serial

from MAB_GPU.agents_ctx import (
    EXP3,
    EXP3IX,
    CtxAgentCfg,
    LinTS,
    LinUCB,
    NeuralLinearTS,
    NeuralTS,
)
from MAB_GPU.agents import KLUCB, AgentCfg, DiscountedUCB, SlidingWindowUCB
from MAB_GPU.puffer_envs import PufferBernoulliBandit, PufferContextualBandit
from MAB_GPU.utils.device import pick_device, memory_stats


@dataclass
class Config:
    env: Literal["contextual", "bernoulli"] = "contextual"
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


def parse_args() -> Config:
    p = argparse.ArgumentParser("Native PufferLib runner for MAB GPU")
    p.add_argument("--env", type=str, choices=["contextual", "bernoulli"], default="contextual")
    p.add_argument("--algo", type=str, default="linucb")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--d", type=int, default=8)
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--runs", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--theta-sigma", type=float, default=0.05)
    p.add_argument("--x-sigma", type=float, default=1.0)
    p.add_argument("--nonstationary", action="store_true")
    p.add_argument("--sigma", type=float, default=0.1)
    # Agents
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--v", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.07)
    p.add_argument("--eta", type=float, default=None)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--ensembles", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--features", type=int, default=64)
    p.add_argument("--linlam", type=float, default=1.0)
    p.add_argument("--linv", type=float, default=0.1)
    # Vectorization
    p.add_argument("--vector", type=str, choices=["mp", "serial"], default="mp")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--force-device", action="store_true")
    p.add_argument("--no-amp", action="store_false", dest="amp")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--tui", action="store_true", help="Rich TUI console dashboard (no matplotlib)")
    args = p.parse_args()
    return Config(
        env=args.env, algo=args.algo, k=args.k, d=args.d, T=args.T, runs=args.runs, seed=args.seed,
        theta_sigma=args.theta_sigma, x_sigma=args.x_sigma, nonstationary=args.nonstationary, sigma=args.sigma,
        alpha=args.alpha, lam=args.lam, v=args.v, gamma=args.gamma, eta=args.eta, hidden=args.hidden, depth=args.depth,
        ensembles=args.ensembles, dropout=args.dropout, lr=args.lr, features=args.features, linlam=args.linlam, linv=args.linv,
        vector=args.vector, num_workers=args.num_workers, batch_size=args.batch_size, device=args.device, log_every=args.log_every, tui=bool(args.tui), force_device=bool(args.force_device), amp=bool(args.amp),
    )


def build_envs(cfg: Config):
    env_creators = []
    env_args = []
    env_kwargs = []
    for _ in range(cfg.runs):
        if cfg.env == "contextual":
            env_creators.append(PufferContextualBandit)
            env_args.append([])
            env_kwargs.append(dict(k=cfg.k, d=cfg.d, nonstationary=False, theta_sigma=cfg.theta_sigma, x_sigma=cfg.x_sigma))
        else:
            env_creators.append(PufferBernoulliBandit)
            env_args.append([])
            env_kwargs.append(dict(k=cfg.k, nonstationary=cfg.nonstationary, sigma=cfg.sigma))
    if cfg.vector == "serial":
        return Serial(env_creators=env_creators, env_args=env_args, env_kwargs=env_kwargs, num_envs=cfg.runs, seed=cfg.seed)
    # multiprocessing
    num_workers = cfg.num_workers if cfg.num_workers is not None else min(cfg.runs, os.cpu_count() or 1)
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


def _extract_p_and_opt(infos, runs: int, k: int):
    opt_flags = None
    opt_mean = None
    pstars = np.full(runs, np.nan, dtype=float)
    pvec0 = None
    pvec_mean = None
    if isinstance(infos, list) and infos:
        try:
            opt_flags = np.array([int(i.get("optimal", 0)) for i in infos], dtype=float)
            pstars = np.array([float(np.nanmax(i.get("p", [ np.nan ]))) for i in infos], dtype=float)
            p0 = infos[0].get("p", None)
            if p0 is not None:
                pvec0 = np.asarray(p0, dtype=float)
        except Exception:
            pass
    elif isinstance(infos, dict):
        try:
            if "optimal" in infos:
                # already averaged across envs by PufferLib Serial
                opt_mean = float(infos.get("optimal"))
            p = infos.get("p", None)
            if p is not None:
                pvec = np.asarray(p, dtype=float)
                pvec_mean = pvec
                pvec0 = pvec
                pstars = np.full(runs, float(np.nanmax(pvec)), dtype=float)
        except Exception:
            pass
    return opt_flags, opt_mean, pstars, pvec0, pvec_mean


def build_agent(cfg: Config, device: torch.device):
    if cfg.env == "contextual":
        acfg = CtxAgentCfg(k=cfg.k, d=cfg.d, num_envs=cfg.runs, device=device)
        key = cfg.algo.lower()
        if key == "linucb":
            return LinUCB(acfg, alpha=cfg.alpha, lam=cfg.lam)
        if key == "lints":
            return LinTS(acfg, v=cfg.v, lam=cfg.lam)
        if key == "exp3":
            return EXP3(acfg, gamma=cfg.gamma, eta=cfg.eta)
        if key == "exp3ix":
            return EXP3IX(acfg, gamma=cfg.gamma, eta=cfg.eta)
        if key == "neuralts":
            return NeuralTS(acfg, hidden=cfg.hidden, depth=cfg.depth, ensembles=cfg.ensembles, dropout=cfg.dropout, lr=cfg.lr, amp=cfg.amp)
        if key == "neurallinear":
            return NeuralLinearTS(acfg, m=cfg.features, hidden=cfg.hidden, depth=cfg.depth, dropout=cfg.dropout, lam=cfg.linlam, v=cfg.linv, lr=cfg.lr, amp=cfg.amp)
        raise ValueError("unknown algo for contextual")
    else:
        key = cfg.algo.lower()
        # EXP3 algorithms need CtxAgentCfg even for Bernoulli (d=1 for compatibility)
        if key in ("exp3", "exp3ix"):
            acfg = CtxAgentCfg(k=cfg.k, d=1, num_envs=cfg.runs, device=device)
            if key == "exp3":
                return EXP3(acfg, gamma=cfg.gamma, eta=cfg.eta)
            if key == "exp3ix":
                return EXP3IX(acfg, gamma=cfg.gamma, eta=cfg.eta)
        else:
            # Classical algorithms use AgentCfg
            acfg = AgentCfg(k=cfg.k, num_envs=cfg.runs, device=device)
            if key == "klucb":
                return KLUCB(acfg, alpha=cfg.alpha)
            if key in ("ducb", "discounted_ucb"):
                return DiscountedUCB(acfg, c=2.0, discount=0.99)
            if key in ("swucb", "sliding_window_ucb"):
                return SlidingWindowUCB(acfg, c=2.0, window=200)
        raise ValueError("unknown algo for bernoulli")


def main() -> None:
    cfg = parse_args()
    # Heuristic: small contextual problems run faster on CPU than MPS for LinTS/NeuralTS
    device = pick_device(cfg.device)
    if (not cfg.force_device
        and device.type == "mps"
        and cfg.env == "contextual"
        and cfg.algo.lower() in ("lints", "neuralts", "neurallinear")
        and (cfg.k * max(1, cfg.d) <= 128)
        and (cfg.runs <= 512)):
        print("[heuristic] Using CPU instead of MPS for small contextual workload (override with --force-device).")
        device = torch.device("cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)


    vec = build_envs(cfg)
    obs, infos = vec.reset(seed=cfg.seed)
    agent = build_agent(cfg, device)
    agent.reset()

    T = cfg.T
    n = cfg.runs
    cumulative_reward = np.zeros(n, dtype=float)
    pstar0 = np.zeros(n, dtype=float)
    have_pstar0 = False
    cumulative_pstar = np.zeros(n, dtype=float)
    cum_counts = np.zeros(cfg.k, dtype=int)
    cum_regret_by_arm = np.zeros(cfg.k, dtype=float)

    # Optional TUI using rich; stays zero-cost if not requested
    tui = None
    if cfg.tui:
        try:
            from MAB_GPU.ui.tui import RichTUI
            device_desc = f"{device.type}"
            if device.type == "cuda":
                device_desc += f":{device.index or 0}"
            title = f"env={cfg.env} • algo={cfg.algo} • runs={cfg.runs} • vec={cfg.vector}"
            tui = RichTUI(k=cfg.k, T=T, device_desc=device_desc, title=title)
        except Exception:
            tui = None

    import time
    start_time = time.perf_counter()
    steps_done = 0
    ewma_ms = None
    # Preallocate buffers for device transfers
    rewards_t = torch.empty((n,), device=device, dtype=torch.float32)
    if cfg.env == "contextual":
        obs_buf = torch.empty((n, cfg.k, cfg.d), device=device, dtype=torch.float32)
    else:
        obs_buf = None

    for t in range(1, T + 1):
        loop_t0 = time.perf_counter()
        # Prepare obs for contextual only (native obs already boxed)
        if cfg.env == "contextual":
            obs_t = obs_buf
            obs_t.copy_(torch.from_numpy(obs).to(device=device, dtype=torch.float32))
            actions = agent.select_actions(t, obs_t)
        else:
            actions = agent.select_actions(t)
        atn_np = actions.detach().cpu().numpy()
        next_obs, r, _, _, infos = vec.step(atn_np)
        rewards = rewards_t
        rewards.copy_(torch.from_numpy(r.astype(np.float32)).to(device=device))
        if cfg.env == "contextual":
            agent.update(actions, rewards, obs_t)
        else:
            agent.update(actions, rewards)
        r_np = r.astype(float)
        cumulative_reward += r_np
        # Cumulative action counts per arm (every step)
        try:
            step_counts = np.bincount(atn_np.reshape(-1), minlength=cfg.k)
            cum_counts += step_counts.astype(int)
        except Exception:
            pass

        # Info-based metrics (works for Serial averaged dict or list of dict)
        opt_flags, opt_mean, pstars, pvec0, pvec_mean = _extract_p_and_opt(infos, n, cfg.k)
        if opt_flags is None:
            opt_flags = np.zeros(n, dtype=float)

        if not have_pstar0:
            pstar0 = np.where(np.isnan(pstars), pstar0, pstars)
            have_pstar0 = True
        if cfg.env == "contextual" or cfg.nonstationary:
            cumulative_pstar += np.nan_to_num(pstars, nan=0.0)

        # no GUI rendering (TUI only)
        # Aggregate and print
        if cfg.env == "contextual" or cfg.nonstationary:
            regrets = cumulative_pstar - cumulative_reward
        else:
            regrets = pstar0 * t - cumulative_reward
        mean_r = float(r_np.mean())
        pct = float(opt_flags.mean() * 100.0) if opt_flags is not None and len(opt_flags) == n and n > 0 else (float(opt_mean) * 100.0 if opt_mean is not None else 0.0)
        rg_m = float(regrets.mean())

        # Update cumulative regret attribution by chosen arm if we have p vectors
        try:
            if isinstance(infos, list) and infos:
                # MP: exact attribution using per-env infos
                for i in range(min(len(infos), len(atn_np))):
                    pvec = infos[i].get("p", None)
                    if pvec is None:
                        continue
                    a = int(atn_np[i])
                    best = float(np.nanmax(pvec))
                    chosen = float(pvec[a])
                    reg = max(0.0, best - chosen)
                    if 0 <= a < cfg.k:
                        cum_regret_by_arm[a] += reg
            elif cfg.vector == "serial":
                # Serial: read per-env p directly from native envs
                try:
                    envs = getattr(vec, 'envs', [])
                    for i, e in enumerate(envs[:len(atn_np)]):
                        pvec = None
                        if cfg.env == "bernoulli":
                            pvec = getattr(e, 'p', None)
                        else:
                            pvec = getattr(e, '_last_p', None)
                        if pvec is None:
                            continue
                        a = int(atn_np[i])
                        best = float(np.nanmax(pvec))
                        chosen = float(pvec[a])
                        reg = max(0.0, best - chosen)
                        if 0 <= a < cfg.k:
                            cum_regret_by_arm[a] += reg
                except Exception:
                    pass
            elif pvec_mean is not None:
                # Fallback: approximate using mean p over envs and step action counts
                step_counts = np.bincount(atn_np.reshape(-1), minlength=cfg.k).astype(float)
                best = float(np.nanmax(pvec_mean))
                diffs = np.maximum(0.0, best - pvec_mean)
                cum_regret_by_arm += diffs * step_counts
            else:
                # Last-resort fallback: use agent's current Q estimates to approximate regret
                try:
                    from MAB_GPU.agents import KLUCB as _KLUCB, DiscountedUCB as _DUCB, SlidingWindowUCB as _SW
                    step_counts = np.bincount(atn_np.reshape(-1), minlength=cfg.k).astype(float)
                    q = None
                    if isinstance(agent, _KLUCB):
                        q = agent.Q.detach().mean(dim=0).cpu().numpy()
                    elif isinstance(agent, _DUCB):
                        q = (agent.S / agent.Neff.clamp_min(1e-8)).detach().mean(dim=0).cpu().numpy()
                    elif isinstance(agent, _SW):
                        q = (agent.Sw / agent.Nw.clamp_min(1e-8)).detach().mean(dim=0).cpu().numpy()
                    if q is not None:
                        best = float(np.nanmax(q))
                        diffs = np.maximum(0.0, best - q)
                        cum_regret_by_arm += diffs * step_counts
                except Exception:
                    pass
        except Exception:
            pass

        if tui is not None and (t % cfg.log_every == 0 or t == 1):
            try:
                acts = atn_np.reshape(-1).tolist()
                mem = memory_stats(device)
                values = None
                values_labels = None
                if cfg.env == "contextual":
                    try:
                        if isinstance(agent, LinUCB):
                            with torch.no_grad():
                                X0 = obs_t[0]
                                theta = torch.matmul(agent.A_inv, agent.b.unsqueeze(-1)).squeeze(-1)[0]
                                logits = (X0 * theta).sum(dim=-1)
                                est = torch.sigmoid(logits).detach().cpu().numpy()
                                truep = None
                                if isinstance(infos, list) and infos and isinstance(infos[0], dict):
                                    p0 = infos[0].get("p", None)
                                    if p0 is not None:
                                        p0 = np.asarray(p0, dtype=float)
                                    truep = p0
                                Ainv0 = agent.A_inv[0]
                                Ainv_x0 = torch.matmul(Ainv0, X0.unsqueeze(-1)).squeeze(-1)
                                conf2 = (X0 * Ainv_x0).sum(dim=-1)
                                conf = torch.sqrt(conf2.clamp_min(1e-12)).detach().cpu().numpy()
                                idx = np.argsort(-est)
                                values = [(int(i), float(est[i]), None if truep is None else float(truep[i]), float(conf[i])) for i in idx[:6]]
                                values_labels = {"est": "est p", "true": "true p", "extra": "conf"}
                        elif isinstance(agent, LinTS):
                            with torch.no_grad():
                                X0 = obs_t[0]
                                mu0 = torch.matmul(agent.A_inv, agent.b.unsqueeze(-1)).squeeze(-1)[0]
                                mean_logits = (X0 * mu0).sum(dim=-1)
                                d = agent.d
                                eps_eye = 1e-6 * torch.eye(d, device=device, dtype=torch.float32)
                                Ainv0 = agent.A_inv[0] + eps_eye
                                L0 = torch.linalg.cholesky(Ainv0)
                                z = torch.randn((cfg.k, d), device=device)
                                theta_samp = mu0 + agent.v * torch.matmul(L0, z.unsqueeze(-1)).squeeze(-1)
                                samp_logits = (X0 * theta_samp).sum(dim=-1)
                                est_sample = torch.sigmoid(samp_logits).detach().cpu().numpy()
                                est_mean = torch.sigmoid(mean_logits).detach().cpu().numpy()
                                truep = None
                                if isinstance(infos, list) and infos and isinstance(infos[0], dict):
                                    p0 = infos[0].get("p", None)
                                    if p0 is not None:
                                        p0 = np.asarray(p0, dtype=float)
                                    truep = p0
                                idx = np.argsort(-est_sample)
                                values = [(int(i), float(est_sample[i]), None if truep is None else float(truep[i]), float(est_mean[i])) for i in idx[:6]]
                                values_labels = {"est": "sample p", "true": "true p", "extra": "mean p"}
                        elif isinstance(agent, NeuralLinearTS):
                            with torch.no_grad():
                                n, k, d = obs_t.shape
                                X0 = obs_t[0]
                                z = agent.encoder(X0)
                                Ainv0 = agent.A_inv[0]
                                b0 = agent.b[0]
                                mu = torch.matmul(Ainv0, b0.unsqueeze(-1)).squeeze(-1)
                                logits = (z * mu).sum(dim=-1)
                                est = torch.sigmoid(logits).detach().cpu().numpy()
                                conf2 = (z * torch.matmul(Ainv0, z.unsqueeze(-1)).squeeze(-1)).sum(dim=-1)
                                conf = torch.sqrt(conf2.clamp_min(1e-12)).detach().cpu().numpy()
                                truep = None
                                if isinstance(infos, list) and infos and isinstance(infos[0], dict):
                                    p0 = infos[0].get("p", None)
                                    if p0 is not None:
                                        p0 = np.asarray(p0, dtype=float)
                                    truep = p0
                                idx = np.argsort(-est)
                                values = [(int(i), float(est[i]), None if truep is None else float(truep[i]), float(conf[i])) for i in idx[:6]]
                                values_labels = {"est": "est p", "true": "true p", "extra": "conf"}
                        elif isinstance(agent, NeuralTS):
                            with torch.no_grad():
                                X0 = obs_t[0]
                                probs = []
                                for m in agent.models:
                                    logits = m(X0).squeeze(-1)
                                    p = torch.sigmoid(logits)
                                    probs.append(p)
                                P = torch.stack(probs, dim=0)
                                mean_p = P.mean(dim=0).detach().cpu().numpy()
                                std_p = P.std(dim=0).detach().cpu().numpy()
                                truep = None
                                if isinstance(infos, list) and infos and isinstance(infos[0], dict):
                                    p0 = infos[0].get("p", None)
                                    if p0 is not None:
                                        p0 = np.asarray(p0, dtype=float)
                                    truep = p0
                                idx = np.argsort(-mean_p)
                                values = [(int(i), float(mean_p[i]), None if truep is None else float(truep[i]), float(std_p[i])) for i in idx[:6]]
                                values_labels = {"est": "mean p", "true": "true p", "extra": "std"}
                    except Exception:
                        values = None
                else:
                    # Bernoulli agents: show empirical Q and true p (mean across envs if aggregated)
                    try:
                        # use top-level imports
                        truep = None
                        if pvec0 is not None:
                            truep = pvec0
                        if isinstance(agent, _KLUCB):
                            q = agent.Q.detach().mean(dim=0).cpu().numpy()
                            import numpy as _np
                            idx = _np.argsort(-q)
                            values = [(int(i), float(q[i]), None if truep is None else float(truep[i])) for i in idx[:6]]
                            values_labels = {"est": "Q", "true": "true p"}
                        elif isinstance(agent, _DUCB):
                            q = (agent.S / agent.Neff.clamp_min(1e-8)).detach().mean(dim=0).cpu().numpy()
                            import numpy as _np
                            idx = _np.argsort(-q)
                            values = [(int(i), float(q[i]), None if truep is None else float(truep[i])) for i in idx[:6]]
                            values_labels = {"est": "Q", "true": "true p"}
                        elif isinstance(agent, _SW):
                            q = (agent.Sw / agent.Nw.clamp_min(1e-8)).detach().mean(dim=0).cpu().numpy()
                            import numpy as _np
                            idx = _np.argsort(-q)
                            values = [(int(i), float(q[i]), None if truep is None else float(truep[i])) for i in idx[:6]]
                            values_labels = {"est": "Q", "true": "true p"}
                    except Exception:
                        pass
                steps_done = t
                now = time.perf_counter()
                elapsed = now - start_time
                sps = (steps_done / elapsed) if elapsed > 0 else None
                last_ms = (now - loop_t0) * 1000.0
                if ewma_ms is None:
                    ewma_ms = last_ms
                else:
                    ewma_ms = 0.8 * ewma_ms + 0.2 * last_ms
                # If regret attribution somehow stayed zero, backfill using mean p or Q and cumulative counts
                try:
                    if float(np.sum(cum_regret_by_arm)) == 0.0:
                        diffs = None
                        if pvec_mean is not None:
                            best = float(np.nanmax(pvec_mean))
                            diffs = np.maximum(0.0, best - pvec_mean)
                        elif pvec0 is not None:
                            best = float(np.nanmax(pvec0))
                            diffs = np.maximum(0.0, best - pvec0)
                        if diffs is None:
                            # fallback to agent estimates
                            q = None
                            if isinstance(agent, KLUCB):
                                q = agent.Q.detach().mean(dim=0).cpu().numpy()
                            elif isinstance(agent, DiscountedUCB):
                                q = (agent.S / agent.Neff.clamp_min(1e-8)).detach().mean(dim=0).cpu().numpy()
                            elif isinstance(agent, SlidingWindowUCB):
                                q = (agent.Sw / agent.Nw.clamp_min(1e-8)).detach().mean(dim=0).cpu().numpy()
                            if q is not None:
                                best = float(np.nanmax(q))
                                diffs = np.maximum(0.0, best - q)
                        if diffs is not None:
                            cum_regret_by_arm = diffs * cum_counts.astype(float)
                except Exception:
                    pass
                tui.update(t=t, mean_r=mean_r, pct_opt=pct, regret=rg_m, actions=acts, mem=mem, values=values, speed_sps=sps, cum_counts=cum_counts, cum_regret_by_arm=cum_regret_by_arm, last_ms=last_ms, ewma_ms=ewma_ms, values_labels=values_labels)
                # Fallback textual summary to ensure visibility in constrained terminals
                try:
                    r = np.asarray(cum_regret_by_arm, dtype=float)
                    top_idx = np.argsort(-r)[:5]
                    total_r = float(r.sum())
                    summary = ", ".join([f"{int(i)}:{r[i]:.3f}({(100.0*r[i]/total_r if total_r>0 else 0.0):.1f}%)" for i in top_idx])
                    print(f"regret_top: {summary}")
                except Exception:
                    pass
            except Exception:
                pass
        elif t % cfg.log_every == 0:
            print(f"t={t} mean_reward={mean_r:.4f} %optimal={pct:.2f} regret={rg_m:.4f}")

        obs = next_obs

    vec.close()
    if tui is not None:
        tui.stop()


if __name__ == "__main__":
    main()

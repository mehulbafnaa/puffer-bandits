import argparse
import time
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from pufferlib.emulation import GymnasiumPufferEnv
from pufferlib.vector import Multiprocessing, Serial
import pufferlib.models as pl_models

from .bandit_env import BernoulliBanditEnv
from .contextual_env import ContextualBanditEnv
from .utils.device import pick_device, memory_stats
from .utils.wandb import wb_init, wb_log, wb_finish


def parse_args():
    p = argparse.ArgumentParser("PPO baseline (one-step) using PufferLib")
    # Problem
    p.add_argument("--env", choices=["contextual", "bernoulli"], default="contextual")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--d", type=int, default=8)
    p.add_argument("--runs", type=int, default=256)
    p.add_argument("--T", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    # Vector/device
    p.add_argument("--vector", choices=["mp", "serial"], default="mp")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    # PPO
    p.add_argument("--policy", choices=["mlp", "lstm"], default="mlp")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--episodes-per-update", type=int, default=64)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=8192)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--vf-coef", type=float, default=0.5)
    # Logging
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-tags", type=str, default=None)
    p.add_argument("--wandb-offline", action="store_true")
    p.add_argument("--run-name", type=str, default=None)
    return p.parse_args()


def build_vecenv(env_kind: str, k: int, d: int, runs: int, vector: str, num_workers: Optional[int], seed: int):
    Vec = Serial if vector == "serial" else Multiprocessing
    env_creators = [GymnasiumPufferEnv for _ in range(runs)]
    env_args = [[] for _ in range(runs)]
    if env_kind == "contextual":
        env_kwargs = [dict(env_creator=ContextualBanditEnv, env_args=[], env_kwargs=dict(
            k=k, d=d
        )) for _ in range(runs)]
    else:
        env_kwargs = [dict(env_creator=BernoulliBanditEnv, env_args=[], env_kwargs=dict(
            k=k
        )) for _ in range(runs)]

    # Workers: choose a divisor of runs when auto
    if num_workers is None:
        cand = min(runs, (torch.get_num_threads() or 1))
        while cand > 1 and (runs % cand) != 0:
            cand -= 1
        num_workers = max(1, cand)
    else:
        num_workers = max(1, min(num_workers, runs))
        if (runs % num_workers) != 0:
            print(f"[vector] runs={runs} not divisible by num_workers={num_workers}; using 1 worker.")
            num_workers = 1

    return Vec(
        env_creators=env_creators,
        env_args=env_args,
        env_kwargs=env_kwargs,
        num_envs=runs,
        num_workers=num_workers,
        seed=seed,
        overwork=True,
    )


def _build_policy(vec, device: torch.device):
    # Minimal: MLP Default policy from PufferLib
    model = pl_models.Default(vec, hidden_size=128).to(device)
    return model


def run_with_config(args) -> None:
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    vec = build_vecenv(args.env, args.k, args.d, args.runs, args.vector, args.num_workers, args.seed)
    obs, infos = vec.reset(seed=args.seed)

    # Build policy via PufferLib (Default MLP or LSTMWrapper(Default))
    if args.policy != "mlp":
        print("[ppo] Only 'mlp' policy supported in this minimal baseline; ignoring --policy")
    model = _build_policy(vec, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Logging (W&B)
    run_name = args.run_name or f"ppo-{args.env}-{args.policy}-k{args.k}-d{args.d}-n{args.runs}-{device}-{args.seed}"
    wb = wb_init(args.wandb, args.wandb_project, args.wandb_entity, args.wandb_tags, run_name, args.wandb_offline, dict(
        runner="ppo", env=args.env, k=args.k, d=args.d, runs=args.runs, device=str(device), seed=args.seed
    ))

    total_updates = max(1, args.T // max(1, args.episodes_per_update))
    steps_done = 0

    for update in range(1, total_updates + 1):
        # Collect transitions over episodes_per_update vector steps
        buf_obs = []
        buf_actions = []
        buf_logprobs = []
        buf_values = []
        buf_rewards = []

        t_collect0 = time.perf_counter()
        for _ in range(args.episodes_per_update):
            # to tensor
            if isinstance(obs, np.ndarray):
                obs_t = torch.from_numpy(obs).to(device=device, dtype=torch.float32)
            else:
                obs_t = torch.tensor(obs, device=device, dtype=torch.float32)

            # Forward through policy, sample action from logits
            logits, value = model(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logprob = dist.log_prob(action)
            value = value.squeeze(-1)

            atn_np = action.detach().cpu().numpy()
            obs, r, _, _, infos = vec.step(atn_np)

            # store
            buf_obs.append(obs_t.detach())
            buf_actions.append(action.detach())
            buf_logprobs.append(logprob.detach())
            buf_values.append(value.detach())
            buf_rewards.append(torch.from_numpy(r.astype(np.float32)).to(device))

            steps_done += 1

        t_collect = (time.perf_counter() - t_collect0) * 1000.0

        # Stack buffers along batch dim: (episodes_per_update * runs, ...)
        obs_b = torch.cat(buf_obs, dim=0)
        actions_b = torch.cat(buf_actions, dim=0)
        logprobs_b = torch.cat(buf_logprobs, dim=0)
        values_b = torch.cat(buf_values, dim=0)
        rewards_b = torch.cat(buf_rewards, dim=0)

        # One-step returns/advantages (gamma=0): return = reward; adv = reward - value
        returns_b = rewards_b
        advantages_b = returns_b - values_b
        # Advantage normalization (stabilizes updates)
        adv_mean = advantages_b.mean()
        adv_std = advantages_b.std(unbiased=False) + 1e-8
        advantages_b = (advantages_b - adv_mean) / adv_std

        # PPO update
        batch_size = advantages_b.shape[0]
        mb_size = min(args.minibatch_size, batch_size)
        idx = torch.randperm(batch_size, device=device)
        epochs = max(1, args.update_epochs)
        t_update0 = time.perf_counter()
        for _ in range(epochs):
            for start in range(0, batch_size, mb_size):
                end = min(start + mb_size, batch_size)
                b = idx[start:end]
                mb_obs = obs_b[b]
                mb_actions = actions_b[b]
                mb_old_logprob = logprobs_b[b]
                mb_adv = advantages_b[b]
                mb_ret = returns_b[b]

                # Recompute logits and value from the underlying model
                logits, value = model(mb_obs)
                dist = Categorical(logits=logits)
                logprob = dist.log_prob(mb_actions)
                value = value.squeeze(-1)
                entropy = dist.entropy().mean()

                ratio = (logprob - mb_old_logprob).exp()
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(value, mb_ret)
                loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        t_update = (time.perf_counter() - t_update0) * 1000.0

        # Basic logging
        mean_r = float(rewards_b.mean().item())
        if update % max(1, args.log_every) == 0:
            print(f"upd={update}/{total_updates} steps={steps_done} mean_reward={mean_r:.4f} collect_ms={t_collect:.1f} update_ms={t_update:.1f}")
            if wb is not None:
                wb_log(wb, {
                    "update": update,
                    "steps": steps_done,
                    "mean_reward": mean_r,
                    "collect_ms": t_collect,
                    "update_ms": t_update,
                    "mem": memory_stats(device).get("allocated", 0.0),
                }, step=steps_done)

    vec.close()
    if wb is not None:
        wb_finish(wb)


def main() -> None:
    args = parse_args()
    run_with_config(args)


if __name__ == "__main__":
    main()

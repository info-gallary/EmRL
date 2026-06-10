"""
bc_warmup.py — Behavioral Cloning Pre-Training
================================================
Pre-trains the PPO network via imitation of ClassicalCGR.
This bootstraps the policy to forward bundles intelligently
before PPO fine-tunes via environment rewards.

Ensures the agent starts from a useful initial policy, preventing
the entropy collapse / local optima seen with pure RL from scratch.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.env.contact_plan import ContactPlan, generate_rrna, generate_rrnb, generate_synthetic
from src.env.dtn_env import DTNRoutingEnv
from src.env.bundle import Bundle
from src.agents.classical_cgr import ClassicalCGR
from src.agents.ppo_agent import PPOAgent

K_CONTACTS = 10


def collect_bc_data(
    contact_plans: list[ContactPlan],
    teacher: ClassicalCGR,
    n_episodes: int = 5000,
    seed: int = 42,
    k_contacts: int = K_CONTACTS,
    env_kwargs: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run ClassicalCGR in DTNRoutingEnv and collect (obs, action, mask) pairs.

    The teacher action is determined by matching the teacher's chosen Contact
    to the corresponding contact slot index in the observation.
    """
    rng = np.random.default_rng(seed)

    all_obs:     List[np.ndarray] = []
    all_actions: List[int]        = []
    all_masks:   List[np.ndarray] = []

    plan_list = contact_plans
    n_delivered = 0

    _env_kwargs = dict(env_kwargs or {})
    for ep in range(n_episodes):
        plan = plan_list[ep % len(plan_list)]
        env  = DTNRoutingEnv(contact_plan=plan, k_contacts=k_contacts, **_env_kwargs)
        obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
        mask = info["action_mask"]

        for _ in range(100):
            bundle   = env._bundle
            cp       = env.cp
            cur_node = env._current_node
            cur_time = env._current_time

            # Ask teacher for the next hop contact
            teacher_contact = teacher.route(bundle, cp, cur_node, cur_time)

            # Map teacher contact to action index in current obs
            action = env.K  # default: hold (if no contact found)
            if teacher_contact is not None:
                for k, slot in enumerate(env._episode_contacts):
                    if slot is not None and slot.contact_id == teacher_contact.contact_id:
                        action = k
                        break
                else:
                    # Teacher found a contact not in the visible window → hold
                    action = env.K

            all_obs.append(obs.copy())
            all_actions.append(action)
            all_masks.append(mask.copy())

            obs, rew, term, trunc, info = env.step(action)
            mask = info["action_mask"]

            if info.get("bdr", 0) > 0:
                n_delivered += 1

            if term or trunc:
                break

        env.close()

    print(f"  BC data: {len(all_obs)} transitions, {n_delivered}/{n_episodes} delivered ({n_delivered/n_episodes:.2%})", flush=True)
    return (
        np.array(all_obs,     dtype=np.float32),
        np.array(all_actions, dtype=np.int64),
        np.array(all_masks,   dtype=bool),
    )


def bc_pretrain(
    agent:        PPOAgent,
    contact_plans: list[ContactPlan],
    n_episodes:   int   = 5000,
    n_epochs:     int   = 20,
    batch_size:   int   = 256,
    lr:           float = 1e-3,
    seed:         int   = 42,
    k_contacts:   int   = K_CONTACTS,
    env_kwargs:   dict | None = None,
) -> dict:
    """
    Pre-train PPO actor via behavioral cloning.

    Returns dict with training stats.
    """
    teacher = ClassicalCGR()

    print(f"Collecting BC rollouts ({n_episodes} episodes, K={k_contacts})...", flush=True)
    obs_arr, act_arr, mask_arr = collect_bc_data(
        contact_plans, teacher, n_episodes=n_episodes, seed=seed,
        k_contacts=k_contacts, env_kwargs=env_kwargs,
    )

    N = len(obs_arr)
    print(f"  Total transitions: {N}")

    obs_t  = torch.as_tensor(obs_arr,  dtype=torch.float32,  device=agent.device)
    act_t  = torch.as_tensor(act_arr,  dtype=torch.long,     device=agent.device)
    mask_t = torch.as_tensor(mask_arr, dtype=torch.bool,     device=agent.device)

    bc_optimizer = Adam(agent.network.parameters(), lr=lr, weight_decay=1e-5)
    losses = []

    agent.network.train()
    for epoch in range(n_epochs):
        idxs = torch.randperm(N, device=agent.device)
        epoch_loss = 0.0
        n_batches  = 0

        for i in range(0, N, batch_size):
            batch_idx = idxs[i : i + batch_size]
            o = obs_t[batch_idx]
            a = act_t[batch_idx]
            m = mask_t[batch_idx]

            dist, _ = agent.network(o, m)
            bc_loss = F.nll_loss(dist.logits, a)

            bc_optimizer.zero_grad()
            bc_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.network.parameters(), 0.5)
            bc_optimizer.step()

            epoch_loss += bc_loss.item()
            n_batches  += 1

        mean_loss = epoch_loss / max(n_batches, 1)
        losses.append(mean_loss)
        if (epoch + 1) % 5 == 0:
            print(f"  BC epoch {epoch+1}/{n_epochs}: loss={mean_loss:.4f}", flush=True)

    return {"bc_losses": losses, "n_transitions": N}

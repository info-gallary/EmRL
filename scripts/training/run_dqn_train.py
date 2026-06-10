"""
run_dqn_train.py — Train vanilla DQN baseline for EmRL comparison.
=====================================================================
This is the "is RL alone enough, or does PPO+attention matter?" baseline.
Trains a flat-MLP DQN on the same observation/action space and curriculum
as EmRL. Same number of environment steps (1M) for fair comparison.

DQN configuration:
  - FlatQNetwork: 3-layer MLP (76 → 256 → 256 → 256 → 12)
  - Epsilon-greedy: 1.0 → 0.05 over 200k steps
  - Replay buffer: 100k transitions
  - Target network update: every 1000 train steps
  - Same reward function and environment as PPO

This is also an approximation of the RUCoP design (Segui 2022), which used
Q-learning over a flat state representation.

Saves: checkpoints/dqn/best_model_bdr*.pt
"""
import io
import sys
import os
import time
import csv
import json
import argparse
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.path.insert(0, '.')

import numpy as np

from src.env.contact_plan import generate_rrna, generate_rrnb, generate_synthetic
from src.env.dtn_env import DTNRoutingEnv
from src.agents.dqn_agent import DQNAgent


def evaluate_dqn(agent: DQNAgent, env_factory, n_episodes: int = 30) -> dict:
    """Quick training-time evaluation. Returns BDR + mean reward."""
    bdrs, rewards = [], []
    env = env_factory()
    for _ in range(n_episodes):
        obs, info = env.reset()
        action_mask = info.get('action_mask', np.ones(12, dtype=bool))
        ep_reward = 0.0
        ep_done = False
        while not ep_done:
            action = agent.select_action(obs, action_mask, deterministic=True)
            obs, r, te, tr, info = env.step(action)
            action_mask = info.get('action_mask', np.ones(12, dtype=bool))
            ep_reward += r
            ep_done = te or tr
        bdrs.append(info.get('bdr', 0.0))
        rewards.append(ep_reward)
    env.close()
    return {
        'mean_bdr': float(np.mean(bdrs)),
        'mean_reward': float(np.mean(rewards)),
        'n_episodes': n_episodes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1_000_000)
    parser.add_argument('--arch', type=str, default='flat', choices=['flat', 'attn'])
    parser.add_argument('--eval-interval', type=int, default=20000,
                        help='env steps between evals')
    parser.add_argument('--eval-episodes', type=int, default=30)
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print(f"DQN Baseline Training (arch={args.arch})", flush=True)
    print("=" * 60, flush=True)

    out_dir = Path(f'results/dqn_{args.arch}')
    ckpt_dir = Path(f'checkpoints/dqn_{args.arch}')
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Curriculum: same as PPO
    print("Generating contact plans...", flush=True)
    rrna = generate_rrna(sim_duration=86400, seed=42)
    rrnb = generate_rrnb(sim_duration=86400, seed=42)
    syn  = generate_synthetic(n_nodes=12, sim_duration=86400, seed=42)
    print(f"  RRN-A: {len(rrna.contacts)} contacts", flush=True)
    print(f"  RRN-B: {len(rrnb.contacts)} contacts", flush=True)

    plans = {'rrna': rrna, 'rrnb': rrnb, 'synthetic': syn}
    plan_keys = list(plans.keys())
    counter = [0]

    def env_factory():
        key = plan_keys[counter[0] % len(plan_keys)]
        counter[0] += 1
        return DTNRoutingEnv(contact_plan=plans[key], topology=key)

    # ── Build agent ───────────────────────────────────────────────────────────
    agent = DQNAgent(
        obs_dim=76, action_dim=12,
        arch=args.arch,
        lr=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=300_000,
        target_update_interval=1000,
        replay_capacity=100_000,
        batch_size=128,
        min_buffer_size=5000,
    )
    print(f"Network parameters: {agent.network.count_parameters():,}", flush=True)

    # ── CSV logging ──────────────────────────────────────────────────────────
    csv_path = out_dir / 'training_curves.csv'
    csv_f = open(csv_path, 'w', newline='')
    csv_w = csv.DictWriter(csv_f, fieldnames=[
        'env_step', 'epsilon', 'q_loss', 'mean_q',
        'eval_bdr', 'eval_reward',
    ])
    csv_w.writeheader()

    # ── Training loop ────────────────────────────────────────────────────────
    env = env_factory()
    obs, info = env.reset()
    mask = info.get('action_mask', np.ones(12, dtype=bool))
    ep_reward = 0.0
    ep_count = 0

    best_bdr = 0.0
    best_ckpt = None
    t0 = time.time()
    last_eval_step = 0
    last_log_step = 0

    print(f"\nTraining for {args.steps:,} environment steps...\n", flush=True)

    for step in range(args.steps):
        action = agent.select_action(obs, mask)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_mask = info.get('action_mask', np.ones(12, dtype=bool))

        agent.store(obs, action, reward, next_obs, done, mask, next_mask)
        metrics = agent.update()

        ep_reward += reward

        if done:
            ep_count += 1
            obs, info = env.reset()
            mask = info.get('action_mask', np.ones(12, dtype=bool))
            ep_reward = 0.0
        else:
            obs = next_obs
            mask = next_mask

        # Periodic logging
        if step - last_log_step >= 2048:
            elapsed = time.time() - t0
            print(f"[{step:>8d}/{args.steps}] eps={metrics['epsilon']:.3f} "
                  f"q_loss={metrics['q_loss']:.4f} mean_q={metrics['mean_q']:+.3f} "
                  f"eps_count={ep_count} elapsed={elapsed:.0f}s", flush=True)
            last_log_step = step

        # Periodic eval
        if step - last_eval_step >= args.eval_interval and step > 5000:
            eval_res = evaluate_dqn(agent, env_factory, args.eval_episodes)
            print(f"  EVAL @ {step}: BDR={eval_res['mean_bdr']:.4f} "
                  f"reward={eval_res['mean_reward']:.2f}", flush=True)

            csv_w.writerow({
                'env_step': step,
                'epsilon': metrics['epsilon'],
                'q_loss':  round(metrics['q_loss'], 5),
                'mean_q':  round(metrics['mean_q'], 5),
                'eval_bdr': round(eval_res['mean_bdr'], 5),
                'eval_reward': round(eval_res['mean_reward'], 5),
            })
            csv_f.flush()

            if eval_res['mean_bdr'] > best_bdr:
                best_bdr = eval_res['mean_bdr']
                best_ckpt = ckpt_dir / f"best_model_bdr{best_bdr:.4f}.pt"
                agent.save(best_ckpt)
                print(f"  New best DQN BDR={best_bdr:.4f} -> {best_ckpt}", flush=True)

            last_eval_step = step

    # Final save
    final_ckpt = ckpt_dir / "final_model.pt"
    agent.save(final_ckpt)
    csv_f.close()

    elapsed = time.time() - t0
    print(f"\nDQN training done in {elapsed/3600:.2f}h", flush=True)
    print(f"Best BDR: {best_bdr:.4f}", flush=True)
    print(f"Best checkpoint: {best_ckpt}", flush=True)

    summary = {
        "baseline": "DQN",
        "arch": args.arch,
        "total_timesteps": args.steps,
        "best_bdr_training_eval": best_bdr,
        "best_checkpoint": str(best_ckpt) if best_ckpt else None,
        "elapsed_hours": round(elapsed / 3600, 2),
    }
    (out_dir / 'training_summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

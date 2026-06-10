"""
run_no_bc_train.py — Train EmRL from scratch WITHOUT BC warmup.
================================================================
Proper ablation for the BC warmup contribution. Identical setup to
run_train.py but skips Phase 0 (BC warmup) entirely.

This gives the correct comparison for the paper's BC ablation claim,
replacing the proxy method that used an early checkpoint.

Saves: checkpoints/no_bc/best_model_bdr*.pt
Logs:  results/no_bc/training_curves.csv
"""
import io
import sys
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

sys.path.insert(0, '.')

from pathlib import Path
import json
import time

from src.env.contact_plan import generate_rrna, generate_rrnb, generate_synthetic
from src.env.dtn_env import DTNRoutingEnv
from src.agents.ppo_agent import PPOAgent
from src.training.trainer import PPOTrainer

Path('results/no_bc').mkdir(parents=True, exist_ok=True)
Path('checkpoints/no_bc').mkdir(parents=True, exist_ok=True)

print("=" * 60, flush=True)
print("EmRL — NoBCWarmup Ablation Training (from scratch)", flush=True)
print("=" * 60, flush=True)

print("Generating contact plans...", flush=True)
rrna = generate_rrna(sim_duration=86400, seed=42)
rrnb = generate_rrnb(sim_duration=86400, seed=42)
syn  = generate_synthetic(n_nodes=12, sim_duration=86400, seed=42)
print(f"  RRN-A: {len(rrna.contacts)} contacts, {len(rrna.node_ids)} nodes", flush=True)
print(f"  RRN-B: {len(rrnb.contacts)} contacts", flush=True)
print(f"  Synthetic: {len(syn.contacts)} contacts", flush=True)

plans = {'rrna': rrna, 'rrnb': rrnb, 'synthetic': syn}
plan_keys = list(plans.keys())
counter = [0]

def env_factory():
    key = plan_keys[counter[0] % len(plan_keys)]
    counter[0] += 1
    return DTNRoutingEnv(contact_plan=plans[key], topology=key)

# IDENTICAL hyperparams to run_train.py - only difference is NO BC warmup
agent = PPOAgent(
    obs_dim=76, action_dim=12,
    lr=3e-4,
    entropy_coef=0.15,
    clip_epsilon=0.15,
    n_epochs=4,
    batch_size=128,
)

# NOTE: No BC warmup phase. Network starts from random init.
print("\nSKIPPING BC warmup (this is the ablation).", flush=True)
print("Network initialized randomly.\n", flush=True)

print("Phase 1-3: PPO training from scratch (1M steps)...", flush=True)
t0 = time.time()

trainer = PPOTrainer(agent=agent)
result = trainer.train(
    env_factory=env_factory,
    total_timesteps=1_000_000,
    rollout_steps=2048,
    eval_interval=10,
    checkpoint_dir='checkpoints/no_bc',
    log_dir='results/no_bc/logs',
    results_dir='results/no_bc',
    n_eval_episodes=30,
    early_stop_patience=60,
    seed=42,
)

elapsed = time.time() - t0
print(f"\nNoBC training done in {elapsed/3600:.2f}h")
print(f"Best BDR (training eval): {result.best_bdr:.4f}")
print(f"Best checkpoint: {result.best_checkpoint}")

summary = {
    "ablation": "NoBCWarmup",
    "total_timesteps": result.total_timesteps,
    "total_updates": result.total_updates,
    "best_bdr_training_eval": result.best_bdr,
    "best_checkpoint": result.best_checkpoint,
    "seed": 42,
    "elapsed_hours": round(elapsed / 3600, 2),
}
Path('results/no_bc/training_summary.json').write_text(json.dumps(summary, indent=2))
print("Summary saved to results/no_bc/training_summary.json")

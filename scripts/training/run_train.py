"""Training runner — bypasses PowerShell Unicode issues."""
import io
import sys
import os

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'   # prevent oneDNN crash on Windows

sys.path.insert(0, '.')

from pathlib import Path
from src.env.contact_plan import generate_rrna, generate_rrnb, generate_synthetic
from src.env.dtn_env import DTNRoutingEnv
from src.agents.ppo_agent import PPOAgent
from src.training.trainer import PPOTrainer
from src.training.bc_warmup import bc_pretrain

Path('results').mkdir(exist_ok=True)
Path('checkpoints').mkdir(exist_ok=True)
Path('figures').mkdir(exist_ok=True)

print("Generating contact plans...", flush=True)
rrna = generate_rrna(sim_duration=86400, seed=42)
rrnb = generate_rrnb(sim_duration=86400, seed=42)
syn  = generate_synthetic(n_nodes=12, sim_duration=86400, seed=42)
print(f"  RRN-A: {len(rrna.contacts)} contacts, {len(rrna.node_ids)} nodes")
print(f"  RRN-B: {len(rrnb.contacts)} contacts")
print(f"  Synthetic: {len(syn.contacts)} contacts")

plans = {'rrna': rrna, 'rrnb': rrnb, 'synthetic': syn}
plan_keys = list(plans.keys())
counter = [0]

def env_factory():
    key = plan_keys[counter[0] % len(plan_keys)]
    counter[0] += 1
    return DTNRoutingEnv(contact_plan=plans[key], topology=key)

agent = PPOAgent(
    obs_dim=76, action_dim=12,
    lr=3e-4,
    entropy_coef=0.15,    # strong entropy regularisation — prevents collapse
    clip_epsilon=0.15,    # tighter clipping — smaller policy steps
    n_epochs=4,
    batch_size=128,
)

# ── Phase 0: Behavioral Cloning warmup ───────────────────────────────────────
import torch
bc_ckpt = Path('checkpoints/bc_warmup.pt')
if bc_ckpt.exists():
    print(f"\nPhase 0: Loading existing BC warmup from {bc_ckpt}...", flush=True)
    agent.network.load_state_dict(torch.load(bc_ckpt, map_location=agent.device, weights_only=True))
    print("BC warmup loaded. Skipping re-training.", flush=True)
else:
    print("\nPhase 0: Behavioral Cloning warmup (ClassicalCGR teacher)...", flush=True)
    bc_stats = bc_pretrain(
        agent=agent,
        contact_plans=[rrna, rrnb, syn],
        n_episodes=3000,
        n_epochs=15,
        batch_size=256,
        lr=5e-4,
        seed=42,
    )
    Path('checkpoints').mkdir(exist_ok=True)
    torch.save(agent.network.state_dict(), str(bc_ckpt))
    print(f"BC warmup done. Final loss: {bc_stats['bc_losses'][-1]:.4f}", flush=True)

print("\nPhase 1-3: PPO fine-tuning (1M steps)...", flush=True)
trainer = PPOTrainer(agent=agent)
result = trainer.train(
    env_factory=env_factory,
    total_timesteps=1_000_000,
    rollout_steps=2048,
    eval_interval=10,
    checkpoint_dir='checkpoints',
    log_dir='results/logs',
    results_dir='results',
    n_eval_episodes=50,
    early_stop_patience=40,
    seed=42,
)
print(f"TRAINING DONE. Best BDR={result.best_bdr:.4f}", flush=True)
print(f"Best checkpoint: {result.best_checkpoint}", flush=True)

import json
summary = {
    "total_timesteps": result.total_timesteps,
    "total_updates": result.total_updates,
    "best_bdr": result.best_bdr,
    "best_checkpoint": result.best_checkpoint,
    "seed": 42,
}
Path('results/training_summary.json').write_text(json.dumps(summary, indent=2))
print("Summary saved to results/training_summary.json", flush=True)

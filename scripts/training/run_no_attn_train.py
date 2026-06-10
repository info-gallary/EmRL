"""
run_no_attn_train.py — Train EmRL WITHOUT contact attention.
================================================================
Architectural ablation: PPO with the same backbone but no
ContactAttentionBlocks (n_attn_layers=0). Each contact is encoded
independently with no cross-contact information flow.

This isolates the contribution of the ContactAttentionBlock vs. the
rest of the architecture (pointer-network actor, global encoder, etc.).

Uses BC warmup (the strong contribution) for fair comparison — only
the attention is removed.

Saves: checkpoints/no_attn/best_model_bdr*.pt
"""
import io
import sys
import os
import time
import json
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

sys.path.insert(0, '.')

from src.env.contact_plan import generate_rrna, generate_rrnb, generate_synthetic
from src.env.dtn_env import DTNRoutingEnv
from src.agents.ppo_agent import PPOAgent
from src.training.trainer import PPOTrainer
from src.training.bc_warmup import bc_pretrain
import torch

Path('results/no_attn').mkdir(parents=True, exist_ok=True)
Path('checkpoints/no_attn').mkdir(parents=True, exist_ok=True)

print("=" * 60, flush=True)
print("EmRL — NoAttention Architecture Ablation", flush=True)
print("=" * 60, flush=True)

print("Generating contact plans...", flush=True)
rrna = generate_rrna(sim_duration=86400, seed=42)
rrnb = generate_rrnb(sim_duration=86400, seed=42)
syn  = generate_synthetic(n_nodes=12, sim_duration=86400, seed=42)
print(f"  RRN-A: {len(rrna.contacts)} contacts, {len(rrna.node_ids)} nodes", flush=True)

plans = {'rrna': rrna, 'rrnb': rrnb, 'synthetic': syn}
plan_keys = list(plans.keys())
counter = [0]

def env_factory():
    key = plan_keys[counter[0] % len(plan_keys)]
    counter[0] += 1
    return DTNRoutingEnv(contact_plan=plans[key], topology=key)

# IDENTICAL hyperparams to run_train.py BUT n_attn_layers=0
# This removes ALL contact self-attention. Contacts are encoded independently.
agent = PPOAgent(
    obs_dim=76, action_dim=12,
    n_attn_layers=0,       # ← THE ABLATION: no contact attention
    lr=3e-4,
    entropy_coef=0.15,
    clip_epsilon=0.15,
    n_epochs=4,
    batch_size=128,
)
print(f"\nNoAttn network parameters: {agent.network.count_parameters():,}", flush=True)

# BC warmup (same as full model — fairness)
bc_ckpt = Path('checkpoints/no_attn/bc_warmup.pt')
if bc_ckpt.exists():
    print(f"\nPhase 0: Loading existing BC warmup from {bc_ckpt}...", flush=True)
    agent.network.load_state_dict(torch.load(bc_ckpt, map_location=agent.device, weights_only=True))
else:
    print("\nPhase 0: BC warmup (ClassicalCGR teacher)...", flush=True)
    bc_stats = bc_pretrain(
        agent=agent,
        contact_plans=[rrna, rrnb, syn],
        n_episodes=3000,
        n_epochs=15,
        batch_size=256,
        lr=5e-4,
        seed=42,
    )
    torch.save(agent.network.state_dict(), str(bc_ckpt))
    print(f"BC warmup done. Final loss: {bc_stats['bc_losses'][-1]:.4f}", flush=True)

print("\nPhase 1-3: PPO training (1M steps, NO contact attention)...", flush=True)
t0 = time.time()

trainer = PPOTrainer(agent=agent)
result = trainer.train(
    env_factory=env_factory,
    total_timesteps=1_000_000,
    rollout_steps=2048,
    eval_interval=10,
    checkpoint_dir='checkpoints/no_attn',
    log_dir='results/no_attn/logs',
    results_dir='results/no_attn',
    n_eval_episodes=30,
    early_stop_patience=60,
    seed=42,
)

elapsed = time.time() - t0
print(f"\nNoAttn training done in {elapsed/3600:.2f}h", flush=True)
print(f"Best BDR (training eval): {result.best_bdr:.4f}", flush=True)
print(f"Best checkpoint: {result.best_checkpoint}", flush=True)

summary = {
    "ablation": "NoAttention",
    "n_attn_layers": 0,
    "total_timesteps": result.total_timesteps,
    "best_bdr_training_eval": result.best_bdr,
    "best_checkpoint": result.best_checkpoint,
    "elapsed_hours": round(elapsed / 3600, 2),
}
Path('results/no_attn/training_summary.json').write_text(json.dumps(summary, indent=2))
print("Summary saved to results/no_attn/training_summary.json", flush=True)

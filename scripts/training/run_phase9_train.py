"""
run_phase9_train.py — Redesigned EmRL architecture (destination-aware contacts).
==================================================================================
ARCHITECTURE CHANGE (same name, stronger model):
  Each candidate contact now carries an 8th feature — the normalized graph hop
  distance from its receiver to the bundle destination (ContactPlan.hop_distance_to).
  This injects the GLOBAL routing structure a local policy previously lacked:
  the agent can now tell which contacts actually lead toward the goal, closing
  the gap to Dijkstra.

  obs: 6 global + K*8 contacts  (K=16 -> 134).  Action: K+2 = 18.

PIPELINE:
  1. Fresh BC warm-up (8-feature obs) cloning ClassicalCGR on CLEAN contacts —
     with the destination-distance signal this clones near-optimal routing.
  2. Reliability-shaped PPO with light anomaly for AACR-aware robustness.

Saves: checkpoints/phase9/bc_warmup_k16_v2.pt, checkpoints/phase9/best_model_bdr*.pt
"""
import io, sys, os, json, time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.path.insert(0, '.')

import torch
from src.env.contact_plan import generate_rrna, generate_rrnb, generate_synthetic
from src.env.dtn_env import DTNRoutingEnv
from src.agents.ppo_agent import PPOAgent
from src.training.trainer import PPOTrainer, CurriculumPhase
from src.training.bc_warmup import bc_pretrain
import src.training.trainer as trainer_mod

K = 16
OBS_DIM, ACTION_DIM = 6 + K*8, K + 2   # 134, 18
RELIABILITY_SHAPING = 1.5
ANOMALY_SEVERITY = (0.2, 0.5)
ANOMALY_RATE = 0.20
PPO_STEPS = 5_000_000

Path('results/phase9').mkdir(parents=True, exist_ok=True)
Path('checkpoints/phase9').mkdir(parents=True, exist_ok=True)

print("=" * 74, flush=True)
print(f"EmRL Phase 9 — Destination-aware redesign (K={K}, OBS={OBS_DIM})", flush=True)
print("=" * 74, flush=True)

rrna = generate_rrna(sim_duration=86400, seed=42)
rrnb = generate_rrnb(sim_duration=86400, seed=42)
syn  = generate_synthetic(n_nodes=12, sim_duration=86400, seed=42)
print(f"RRN-A {len(rrna.contacts)} | RRN-B {len(rrnb.contacts)} | Syn {len(syn.contacts)}", flush=True)

agent = PPOAgent(obs_dim=OBS_DIM, action_dim=ACTION_DIM,
                 lr=2e-4, entropy_coef=0.03, clip_epsilon=0.15,
                 n_epochs=4, batch_size=256)
print(f"Network params: {agent.network.count_parameters():,}", flush=True)

# ── Resume from best phase9 PPO ckpt if present ───────────────────────────────
def best_ppo():
    c = list(Path('checkpoints/phase9').glob('best_model_bdr*.pt'))
    if not c: return None
    return max(c, key=lambda p: float(p.stem.replace('best_model_bdr',''))
               if p.stem.replace('best_model_bdr','').replace('.','').isdigit() else 0.0)

resume = best_ppo()
bc_ckpt = Path('checkpoints/phase9/bc_warmup_k16_v2.pt')
if resume is not None:
    print(f"\nResuming PPO from {resume}", flush=True)
    agent.load(resume)
    for pg in agent.optimizer.param_groups: pg['lr'] = 2e-4
elif bc_ckpt.exists():
    print(f"\nLoading BC warmup {bc_ckpt}", flush=True)
    agent.network.load_state_dict(torch.load(bc_ckpt, map_location=agent.device, weights_only=True))
else:
    print("\nPhase 0: BC warmup (8-feature, clone ClassicalCGR, clean)...", flush=True)
    t0 = time.time()
    stats = bc_pretrain(agent=agent, contact_plans=[rrna, rrnb, syn],
                        n_episodes=8000, n_epochs=25, batch_size=256, lr=5e-4,
                        seed=42, k_contacts=K, env_kwargs={})
    torch.save(agent.network.state_dict(), str(bc_ckpt))
    print(f"BC done {(time.time()-t0)/60:.1f}min, loss={stats['bc_losses'][-1]:.4f}", flush=True)

# ── PPO fine-tune ─────────────────────────────────────────────────────────────
plans = ['rrna', 'rrna', 'rrna', 'rrnb', 'syn']
counter = [0]; pm = {'rrna': rrna, 'rrnb': rrnb, 'syn': syn}
def env_factory():
    key = plans[counter[0] % len(plans)]; counter[0] += 1
    topo = 'synthetic' if key == 'syn' else key
    return DTNRoutingEnv(contact_plan=pm[key], topology=topo, k_contacts=K,
                         reliability_shaping=RELIABILITY_SHAPING,
                         anomaly_severity=ANOMALY_SEVERITY)

trainer_mod.CURRICULUM = [CurriculumPhase("phase9_destaware", 0, PPO_STEPS + 1, ANOMALY_RATE, True)]
print(f"\nPPO {PPO_STEPS:,} steps (anomaly={ANOMALY_RATE}, shaping={RELIABILITY_SHAPING})...\n", flush=True)
t0 = time.time()
trainer = PPOTrainer(agent=agent, obs_dim=OBS_DIM, action_dim=ACTION_DIM)
result = trainer.train(
    env_factory=env_factory, total_timesteps=PPO_STEPS, rollout_steps=4096,
    eval_interval=20, checkpoint_dir='checkpoints/phase9',
    log_dir='results/phase9/logs', results_dir='results/phase9',
    n_eval_episodes=50, early_stop_patience=120, seed=9)
elapsed = time.time() - t0
print(f"\nPhase 9 done in {elapsed/3600:.1f}h  best BDR={result.best_bdr:.4f}", flush=True)
Path('results/phase9/summary.json').write_text(json.dumps({
    "phase": 9, "K": K, "obs_dim": OBS_DIM, "feature": "destination-distance",
    "anomaly_rate": ANOMALY_RATE, "reliability_shaping": RELIABILITY_SHAPING,
    "ppo_steps": PPO_STEPS, "best_bdr": result.best_bdr,
    "best_checkpoint": result.best_checkpoint, "elapsed_hours": round(elapsed/3600,2)}, indent=2))
print("Summary saved.", flush=True)

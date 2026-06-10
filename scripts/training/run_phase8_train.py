"""
run_phase8_train.py — Stronger model: lift baseline BDR (0.68 -> target ~0.85).
===============================================================================
Resumes the K=16 model and prioritises CLEAN routing quality:
  - lighter anomaly rate (0.15) so the policy isn't penalised away from optimal
    timing-based routing,
  - lighter reliability shaping (1.0) to avoid distorting clean routes,
  - longer training (6M steps), tighter convergence (low entropy, low LR).

Goal: raise the baseline (no-anomaly) BDR so that jamming evaluations start from
a higher floor -> high absolute BDR AND a clear margin over Classical CGR.

Saves: checkpoints/phase8/best_model_bdr*.pt
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
import src.training.trainer as trainer_mod

K = 16
OBS_DIM, ACTION_DIM = 6 + K*7, K + 2
RELIABILITY_SHAPING = 1.0
ANOMALY_SEVERITY = (0.2, 0.5)
ANOMALY_RATE = 0.15
PPO_STEPS = 6_000_000

Path('results/phase8').mkdir(parents=True, exist_ok=True)
Path('checkpoints/phase8').mkdir(parents=True, exist_ok=True)

print("=" * 72, flush=True)
print(f"EmRL Phase 8 — Stronger model (K={K}, lighter anomaly, longer train)", flush=True)
print("=" * 72, flush=True)

rrna = generate_rrna(sim_duration=86400, seed=42)
rrnb = generate_rrnb(sim_duration=86400, seed=42)
syn  = generate_synthetic(n_nodes=12, sim_duration=86400, seed=42)

plans = ['rrna', 'rrna', 'rrna', 'rrnb', 'syn']   # weight clean RRN-A more
counter = [0]; pm = {'rrna': rrna, 'rrnb': rrnb, 'syn': syn}

def env_factory():
    key = plans[counter[0] % len(plans)]; counter[0] += 1
    topo = 'synthetic' if key == 'syn' else key
    return DTNRoutingEnv(contact_plan=pm[key], topology=topo, k_contacts=K,
                         reliability_shaping=RELIABILITY_SHAPING,
                         anomaly_severity=ANOMALY_SEVERITY)

agent = PPOAgent(obs_dim=OBS_DIM, action_dim=ACTION_DIM,
                 lr=1.5e-4, entropy_coef=0.02, clip_epsilon=0.12,
                 n_epochs=4, batch_size=256)

def best_ckpt():
    c = list(Path('checkpoints/phase8').glob('best_model_bdr*.pt'))
    if c:
        return max(c, key=lambda p: float(p.stem.replace('best_model_bdr',''))
                   if p.stem.replace('best_model_bdr','').replace('.','').isdigit() else 0.0)
    f = Path('checkpoints/final/emrl_main_k16.pt')
    return f if f.exists() else None

ck = best_ckpt()
agent.load(ck)
for pg in agent.optimizer.param_groups: pg['lr'] = 1.5e-4
print(f"Resumed from: {ck}", flush=True)

PHASE8 = [CurriculumPhase("phase8_strong", 0, PPO_STEPS + 1, ANOMALY_RATE, True)]
trainer_mod.CURRICULUM = PHASE8

print(f"\nTraining {PPO_STEPS:,} steps (anomaly={ANOMALY_RATE}, shaping={RELIABILITY_SHAPING})...\n", flush=True)
t0 = time.time()
trainer = PPOTrainer(agent=agent, obs_dim=OBS_DIM, action_dim=ACTION_DIM)
result = trainer.train(
    env_factory=env_factory, total_timesteps=PPO_STEPS, rollout_steps=4096,
    eval_interval=20, checkpoint_dir='checkpoints/phase8',
    log_dir='results/phase8/logs', results_dir='results/phase8',
    n_eval_episodes=50, early_stop_patience=120, seed=8,
)
elapsed = time.time() - t0
print(f"\nPhase 8 done in {elapsed/3600:.1f}h  best BDR={result.best_bdr:.4f}", flush=True)
Path('results/phase8/summary.json').write_text(json.dumps({
    "phase": 8, "K": K, "anomaly_rate": ANOMALY_RATE,
    "reliability_shaping": RELIABILITY_SHAPING, "ppo_steps": PPO_STEPS,
    "best_bdr": result.best_bdr, "best_checkpoint": result.best_checkpoint,
    "elapsed_hours": round(elapsed/3600, 2)}, indent=2))
print("Summary saved.", flush=True)

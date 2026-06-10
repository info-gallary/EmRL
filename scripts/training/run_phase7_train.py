"""
run_phase7_train.py — Wide-observation architecture (K=16) for "match standard, win anomaly".
==============================================================================================

Two-part recipe:
  1. STRONG BC WARMUP (K=16): clone ClassicalCGR on CLEAN contacts. With 16
     contacts visible (vs 10), the policy can mimic Dijkstra's first-hop choice
     far better -> lifts standard BDR toward Dijkstra quality.
  2. RELIABILITY-SHAPED PPO: light fine-tune with anomaly injection +
     reward += 2.0*(reliability-0.5). Adds anomaly-aware routing WITHOUT
     drifting far from the BC-cloned near-optimal standard policy
     (low LR, low entropy preserve the BC behaviour).

Architecture: K=16 -> OBS_DIM=118, ACTION_DIM=18. The network is K-agnostic
(weights identical; only obs width changes), so this trains fresh from BC.

Outputs:
  checkpoints/phase7/bc_warmup_k16.pt
  checkpoints/phase7/best_model_bdr*.pt
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
OBS_DIM = 6 + K * 7      # 118
ACTION_DIM = K + 2       # 18
RELIABILITY_SHAPING = 2.0
ANOMALY_SEVERITY = (0.2, 0.5)
ANOMALY_RATE = 0.25
PPO_STEPS = 4_000_000

Path('results/phase7').mkdir(parents=True, exist_ok=True)
Path('checkpoints/phase7').mkdir(parents=True, exist_ok=True)

print("=" * 74, flush=True)
print(f"EmRL Phase 7 — Wide Observation (K={K}, OBS={OBS_DIM}, ACT={ACTION_DIM})", flush=True)
print(f"  Goal: match Classical on standard, beat it under anomaly", flush=True)
print("=" * 74, flush=True)

rrna = generate_rrna(sim_duration=86400, seed=42)
rrnb = generate_rrnb(sim_duration=86400, seed=42)
syn  = generate_synthetic(n_nodes=12, sim_duration=86400, seed=42)
print(f"RRN-A {len(rrna.contacts)} | RRN-B {len(rrnb.contacts)} | Syn {len(syn.contacts)}", flush=True)

agent = PPOAgent(
    obs_dim=OBS_DIM, action_dim=ACTION_DIM,
    lr=2e-4, entropy_coef=0.04, clip_epsilon=0.15,
    n_epochs=4, batch_size=256,
)
print(f"Network params: {agent.network.count_parameters():,}", flush=True)

# ── Resume from best PPO checkpoint if one exists (don't waste PPO progress) ───
def _best_ppo_ckpt():
    c = list(Path('checkpoints/phase7').glob('best_model_bdr*.pt'))
    if not c:
        return None
    return max(c, key=lambda p: float(p.stem.replace('best_model_bdr', ''))
               if p.stem.replace('best_model_bdr', '').replace('.', '').isdigit() else 0.0)

_resume = _best_ppo_ckpt()
bc_ckpt = Path('checkpoints/phase7/bc_warmup_k16.pt')
if _resume is not None:
    print(f"\nResuming PPO from best checkpoint: {_resume}", flush=True)
    agent.load(_resume)
    for pg in agent.optimizer.param_groups:
        pg['lr'] = 2e-4
elif bc_ckpt.exists():
    print(f"\nLoading existing BC warmup: {bc_ckpt}", flush=True)
    agent.network.load_state_dict(torch.load(bc_ckpt, map_location=agent.device, weights_only=True))
else:
    print("\nPhase 0: STRONG BC warmup (K=16, clean, clone ClassicalCGR)...", flush=True)
    t0 = time.time()
    stats = bc_pretrain(
        agent=agent,
        contact_plans=[rrna, rrnb, syn],
        n_episodes=8000,        # heavy cloning for high standard BDR
        n_epochs=25,
        batch_size=256,
        lr=5e-4,
        seed=42,
        k_contacts=K,
        env_kwargs={},          # clean (no anomaly, no shaping) — clone optimal routing
    )
    torch.save(agent.network.state_dict(), str(bc_ckpt))
    print(f"BC warmup done in {(time.time()-t0)/60:.1f}min, final loss={stats['bc_losses'][-1]:.4f}", flush=True)

# ── Phase 1: reliability-shaped PPO fine-tune ─────────────────────────────────
plans = ['rrna', 'rrna', 'rrnb', 'syn']
counter = [0]
plan_map = {'rrna': rrna, 'rrnb': rrnb, 'syn': syn}

def env_factory():
    key = plans[counter[0] % len(plans)]
    counter[0] += 1
    topo = 'synthetic' if key == 'syn' else key
    return DTNRoutingEnv(
        contact_plan=plan_map[key], topology=topo,
        k_contacts=K, reliability_shaping=RELIABILITY_SHAPING,
        anomaly_severity=ANOMALY_SEVERITY,
    )

PHASE7 = [CurriculumPhase("phase7_wide", 0, PPO_STEPS + 1, ANOMALY_RATE, True)]
_orig = trainer_mod.CURRICULUM
trainer_mod.CURRICULUM = PHASE7

print(f"\nPhase 1: reliability-shaped PPO ({PPO_STEPS:,} steps, anomaly={ANOMALY_RATE})...\n", flush=True)
t0 = time.time()

trainer = PPOTrainer(agent=agent, obs_dim=OBS_DIM, action_dim=ACTION_DIM)
result = trainer.train(
    env_factory=env_factory,
    total_timesteps=PPO_STEPS,
    rollout_steps=4096,
    eval_interval=20,
    checkpoint_dir='checkpoints/phase7',
    log_dir='results/phase7/logs',
    results_dir='results/phase7',
    n_eval_episodes=50,
    early_stop_patience=100,
    seed=7,
)

trainer_mod.CURRICULUM = _orig
elapsed = time.time() - t0
print(f"\nPhase 7 done in {elapsed/3600:.1f}h  best BDR={result.best_bdr:.4f}", flush=True)
print(f"Best checkpoint: {result.best_checkpoint}", flush=True)

summary = {
    "phase": 7, "K": K, "obs_dim": OBS_DIM, "action_dim": ACTION_DIM,
    "reliability_shaping": RELIABILITY_SHAPING, "anomaly_rate": ANOMALY_RATE,
    "ppo_steps": PPO_STEPS, "best_bdr": result.best_bdr,
    "best_checkpoint": result.best_checkpoint,
    "elapsed_hours": round(elapsed / 3600, 2),
}
Path('results/phase7/summary.json').write_text(json.dumps(summary, indent=2))
print("Summary saved.", flush=True)

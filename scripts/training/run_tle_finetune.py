"""
run_tle_finetune.py — Phase 12: fine-tune EmRL on REAL TLE-derived geometry.
============================================================================
The synthetic-trained EmRL-PPO does not transfer zero-shot to real orbital
geometry (§6.13: ~0.5 BDR vs CGR 0.76-0.96). This fine-tunes the SAME 9-feature
architecture on contact plans built from real NORAD TLEs (SGP4), mixed with the
original synthetic topologies to avoid catastrophic forgetting.

Pipeline (warm-start from checkpoints/final/emrl_main_k16.pt):
  1. BC fine-tune cloning ClassicalCGR on the TLE+synthetic pool (CGR is strong
     on real geometry, so this transfers a competent prior).
  2. Reliability-shaped PPO with anomaly injection over a plan-swapping env.

Saves: checkpoints/phase12_tle/best_model_bdr*.pt
Promotes best -> checkpoints/final/emrl_tle_ft_k16.pt   (does NOT touch emrl_main_k16.pt)
"""
import io, sys, os, json, time, random, shutil
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.path.insert(0, '.')

import torch
from src.env.contact_plan import generate_rrna, generate_rrnb, generate_walker
from src.env.tle_contact_plan import generate_from_tle
from src.env.dtn_env import DTNRoutingEnv, _QueueTracker
from src.agents.ppo_agent import PPOAgent
from src.training.trainer import PPOTrainer, CurriculumPhase
from src.training.bc_warmup import bc_pretrain
import src.training.trainer as trainer_mod

K = 16
OBS_DIM, ACTION_DIM = 6 + K * 9, K + 2     # 150, 18
SIM_DUR = 86400.0
RELIABILITY_SHAPING = 1.5
ANOMALY_SEVERITY = (0.2, 0.5)
ANOMALY_RATE = 0.15
PPO_STEPS = int(os.environ.get('TLE_PPO_STEPS', 800_000))
BC_EPISODES = int(os.environ.get('TLE_BC_EPISODES', 8000))
PATIENCE = int(os.environ.get('TLE_PATIENCE', 40))
# Leave-one-constellation-out: exclude this group from TRAINING (held-out for eval).
HOLDOUT = os.environ.get('TLE_HOLDOUT', '').strip()
TAG = f"_holdout_{HOLDOUT}" if HOLDOUT else ""
CKPT_DIR = f"checkpoints/phase12_tle{TAG}"
RESULTS_DIR = f"results/phase12_tle{TAG}"
PROMOTED = f"checkpoints/final/emrl_tle_ft{TAG}_k16.pt"

Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
Path(CKPT_DIR).mkdir(parents=True, exist_ok=True)

print("=" * 76, flush=True)
print(f"EmRL Phase 12 — fine-tune on REAL TLE geometry (K={K}, OBS={OBS_DIM})", flush=True)
print(f"PPO_STEPS={PPO_STEPS:,}  BC_EPISODES={BC_EPISODES:,}  patience={PATIENCE}", flush=True)
if HOLDOUT:
    print(f"HELD-OUT (excluded from training): {HOLDOUT}  ->  promote to {PROMOTED}", flush=True)
print("=" * 76, flush=True)


# --- plan-swapping env: trainer holds ONE env, so swap the plan on reset() ---
class MultiPlanDTNEnv(DTNRoutingEnv):
    def __init__(self, plans, plan_seed=0, **kwargs):
        self._plans = list(plans)
        self._plan_rng = random.Random(plan_seed)
        super().__init__(contact_plan=self._plans[0], **kwargs)

    def reset(self, *, seed=None, options=None):
        plan = self._plan_rng.choice(self._plans)
        self.cp = plan
        self.n_nodes = plan.n_nodes
        self._queues = _QueueTracker(self.n_nodes)
        return super().reset(seed=seed, options=options)


# --- build the plan pool: real TLE (weighted) + synthetic (anti-forgetting) ---
print("\nBuilding TLE plans from real catalogs (cached after first fetch)...", flush=True)
tle_specs = [("iridium-NEXT", 24), ("orbcomm", 20), ("globalstar", 20), ("oneweb", 24)]
if HOLDOUT:
    tle_specs = [(g, n) for g, n in tle_specs if g != HOLDOUT]
    print(f"  (excluding held-out group '{HOLDOUT}' from training pool)", flush=True)
tle_plans = []
for grp, ns in tle_specs:
    try:
        p = generate_from_tle(group=grp, n_sats=ns, sim_duration=SIM_DUR,
                              step_s=60.0, seed=0, verbose=True)
        tle_plans.append(p)
    except Exception as exc:  # noqa: BLE001
        print(f"  [skip] {grp}: {exc}", flush=True)

print("\nBuilding synthetic plans (anti-forgetting)...", flush=True)
syn_plans = [
    generate_rrna(sim_duration=SIM_DUR, seed=42),
    generate_rrnb(sim_duration=SIM_DUR, seed=42),
    generate_walker(6, 8, 4, seed=42),
]
for p in syn_plans:
    print(f"  synthetic: nodes={p.n_nodes} contacts={len(p.contacts)}", flush=True)

# Weight the pool ~70% TLE / 30% synthetic by repetition.
pool = tle_plans * 2 + syn_plans
bc_pool = tle_plans * 2 + syn_plans     # BC sees the same distribution
print(f"\nPool: {len(tle_plans)} TLE × 2 + {len(syn_plans)} synthetic = {len(pool)} entries", flush=True)

# --- agent: warm-start from the promoted final model ---
agent = PPOAgent(obs_dim=OBS_DIM, action_dim=ACTION_DIM,
                 lr=1e-4, entropy_coef=0.02, clip_epsilon=0.15,
                 n_epochs=4, batch_size=256)
print(f"Network params: {agent.network.count_parameters():,}", flush=True)

resume = Path(CKPT_DIR)
ft_ckpts = list(resume.glob('best_model_bdr*.pt'))
warm = Path('checkpoints/final/emrl_main_k16.pt')
bc_ckpt = Path(CKPT_DIR) / 'bc_tle.pt'
if ft_ckpts:
    latest = max(ft_ckpts, key=lambda p: p.stat().st_mtime)
    print(f"\nResuming PPO from {latest}", flush=True)
    agent.load(latest)
    for pg in agent.optimizer.param_groups: pg['lr'] = 1e-4
elif bc_ckpt.exists():
    print(f"\nLoading BC-TLE warmup {bc_ckpt}", flush=True)
    agent.network.load_state_dict(torch.load(bc_ckpt, map_location=agent.device, weights_only=True))
else:
    print(f"\nWarm-starting from {warm}", flush=True)
    agent.load(warm)
    print("\nBC fine-tune: clone ClassicalCGR on TLE+synthetic pool...", flush=True)
    t0 = time.time()
    stats = bc_pretrain(agent=agent, contact_plans=bc_pool,
                        n_episodes=BC_EPISODES, n_epochs=20, batch_size=256, lr=3e-4,
                        seed=42, k_contacts=K, env_kwargs={})
    torch.save(agent.network.state_dict(), str(bc_ckpt))
    print(f"BC done {(time.time()-t0)/60:.1f}min, loss={stats['bc_losses'][-1]:.4f}", flush=True)

# --- PPO fine-tune over the plan-swapping env ---
def env_factory():
    return MultiPlanDTNEnv(pool, plan_seed=random.randint(0, 1_000_000),
                           topology='synthetic', k_contacts=K, sim_duration=SIM_DUR,
                           reliability_shaping=RELIABILITY_SHAPING,
                           anomaly_severity=ANOMALY_SEVERITY)

trainer_mod.CURRICULUM = [CurriculumPhase("phase12_tle", 0, PPO_STEPS + 1, ANOMALY_RATE, True)]
print(f"\nPPO {PPO_STEPS:,} steps (anomaly={ANOMALY_RATE}, shaping={RELIABILITY_SHAPING})...\n", flush=True)
t0 = time.time()
trainer = PPOTrainer(agent=agent, obs_dim=OBS_DIM, action_dim=ACTION_DIM)
result = trainer.train(
    env_factory=env_factory, total_timesteps=PPO_STEPS, rollout_steps=4096,
    eval_interval=8, checkpoint_dir=CKPT_DIR,
    log_dir=f'{RESULTS_DIR}/logs', results_dir=RESULTS_DIR,
    n_eval_episodes=60, early_stop_patience=PATIENCE, seed=12)
elapsed = time.time() - t0
print(f"\nPhase 12 done in {elapsed/3600:.2f}h  best BDR={result.best_bdr:.4f}", flush=True)

# Promote best -> a NEW final checkpoint (never overwrite emrl_main_k16.pt)
if result.best_checkpoint and Path(result.best_checkpoint).exists():
    dst = Path(PROMOTED)
    shutil.copy(result.best_checkpoint, dst)
    print(f"Promoted {result.best_checkpoint} -> {dst}", flush=True)

Path(f'{RESULTS_DIR}/summary.json').write_text(json.dumps({
    "phase": 12, "K": K, "obs_dim": OBS_DIM, "held_out": HOLDOUT or None,
    "warm_start": str(warm), "tle_groups": [g for g, _ in tle_specs],
    "n_tle_plans": len(tle_plans), "n_syn_plans": len(syn_plans),
    "anomaly_rate": ANOMALY_RATE, "reliability_shaping": RELIABILITY_SHAPING,
    "ppo_steps": PPO_STEPS, "bc_episodes": BC_EPISODES,
    "best_bdr": result.best_bdr, "best_checkpoint": result.best_checkpoint,
    "promoted": PROMOTED,
    "elapsed_hours": round(elapsed/3600, 2)}, indent=2))
print("Summary saved.", flush=True)

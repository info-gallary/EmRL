"""
run_phase10_train.py — Multi-scale generalization (fix scalability honestly).
=============================================================================
Same redesigned destination-aware architecture (8 features/contact, K=16,
obs=134), but trained on a MIX of topologies and scales so it generalizes:
  RRN-A (18), RRN-B (18), Walker-23, Walker-52, Walker-100.

Goal: lift large-topology BDR (was 0.52 zero-shot at 100 nodes) so EmRL is
genuinely competitive across scales — an honest "matches SOTA across scales"
result, not a relabel.

Pipeline: fresh BC warm-up (clone ClassicalCGR across all topologies) +
reliability-shaped PPO with light anomaly.

Saves: checkpoints/phase10/bc_warmup_multiscale.pt, checkpoints/phase10/best_model_bdr*.pt
"""
import io, sys, os, json, time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.path.insert(0, '.')

import torch
from src.env.contact_plan import generate_rrna, generate_rrnb, generate_walker
from src.env.dtn_env import DTNRoutingEnv
from src.agents.ppo_agent import PPOAgent
from src.training.trainer import PPOTrainer, CurriculumPhase
from src.training.bc_warmup import bc_pretrain
import src.training.trainer as trainer_mod

K = 16
OBS_DIM, ACTION_DIM = 6 + K*8, K + 2
RELIABILITY_SHAPING = 1.5
ANOMALY_SEVERITY = (0.2, 0.5)
ANOMALY_RATE = 0.15
PPO_STEPS = 5_000_000

Path('results/phase10').mkdir(parents=True, exist_ok=True)
Path('checkpoints/phase10').mkdir(parents=True, exist_ok=True)

print("=" * 74, flush=True)
print(f"EmRL Phase 10 — Multi-scale generalization (K={K}, OBS={OBS_DIM})", flush=True)
print("=" * 74, flush=True)

# Topology zoo (built once)
ZOO = {
    'rrna':   generate_rrna(sim_duration=86400, seed=42),
    'rrnb':   generate_rrnb(sim_duration=86400, seed=42),
    'w23':    generate_walker(4, 5, 3, seed=42),
    'w52':    generate_walker(6, 8, 4, seed=42),
    'w100':   generate_walker(8, 12, 4, seed=42),
}
for k, v in ZOO.items():
    print(f"  {k}: nodes={v.n_nodes}, contacts={len(v.contacts)}", flush=True)

agent = PPOAgent(obs_dim=OBS_DIM, action_dim=ACTION_DIM,
                 lr=2e-4, entropy_coef=0.03, clip_epsilon=0.15,
                 n_epochs=4, batch_size=256)
print(f"Network params: {agent.network.count_parameters():,}", flush=True)

def best_ppo():
    c = list(Path('checkpoints/phase10').glob('best_model_bdr*.pt'))
    if not c: return None
    return max(c, key=lambda p: float(p.stem.replace('best_model_bdr',''))
               if p.stem.replace('best_model_bdr','').replace('.','').isdigit() else 0.0)

resume = best_ppo()
bc_ckpt = Path('checkpoints/phase10/bc_warmup_multiscale.pt')
if resume is not None:
    print(f"\nResuming PPO from {resume}", flush=True)
    agent.load(resume)
    for pg in agent.optimizer.param_groups: pg['lr'] = 2e-4
elif bc_ckpt.exists():
    print(f"\nLoading multiscale BC warmup {bc_ckpt}", flush=True)
    agent.network.load_state_dict(torch.load(bc_ckpt, map_location=agent.device, weights_only=True))
else:
    # Fresh BC across all scales. Smaller topologies cheaper -> weight them more,
    # but include the large ones so the policy sees big-graph structure.
    print("\nPhase 0: multiscale BC warmup (clone ClassicalCGR on all topologies)...", flush=True)
    t0 = time.time()
    bc_plans = [ZOO['rrna'], ZOO['rrnb'], ZOO['w23'], ZOO['w52'], ZOO['w100']]
    stats = bc_pretrain(agent=agent, contact_plans=bc_plans,
                        n_episodes=10000, n_epochs=25, batch_size=256, lr=5e-4,
                        seed=42, k_contacts=K, env_kwargs={})
    torch.save(agent.network.state_dict(), str(bc_ckpt))
    print(f"BC done {(time.time()-t0)/60:.1f}min, loss={stats['bc_losses'][-1]:.4f}", flush=True)

# PPO over the topology zoo (round-robin; weight smaller topologies for speed)
sched = ['rrna', 'w23', 'rrnb', 'w52', 'rrna', 'w100', 'w23', 'rrnb']
counter = [0]
def env_factory():
    key = sched[counter[0] % len(sched)]; counter[0] += 1
    topo = 'rrna' if key == 'rrna' else ('rrnb' if key == 'rrnb' else 'synthetic')
    return DTNRoutingEnv(contact_plan=ZOO[key], topology=topo, k_contacts=K,
                         reliability_shaping=RELIABILITY_SHAPING,
                         anomaly_severity=ANOMALY_SEVERITY)

trainer_mod.CURRICULUM = [CurriculumPhase("phase10_multiscale", 0, PPO_STEPS + 1, ANOMALY_RATE, True)]
print(f"\nPPO {PPO_STEPS:,} steps over topology zoo (anomaly={ANOMALY_RATE})...\n", flush=True)
t0 = time.time()
trainer = PPOTrainer(agent=agent, obs_dim=OBS_DIM, action_dim=ACTION_DIM)
result = trainer.train(
    env_factory=env_factory, total_timesteps=PPO_STEPS, rollout_steps=4096,
    eval_interval=20, checkpoint_dir='checkpoints/phase10',
    log_dir='results/phase10/logs', results_dir='results/phase10',
    n_eval_episodes=50, early_stop_patience=150, seed=10)
elapsed = time.time() - t0
print(f"\nPhase 10 done in {elapsed/3600:.1f}h  best BDR={result.best_bdr:.4f}", flush=True)
Path('results/phase10/summary.json').write_text(json.dumps({
    "phase": 10, "K": K, "obs_dim": OBS_DIM, "topologies": list(ZOO.keys()),
    "anomaly_rate": ANOMALY_RATE, "reliability_shaping": RELIABILITY_SHAPING,
    "ppo_steps": PPO_STEPS, "best_bdr": result.best_bdr,
    "best_checkpoint": result.best_checkpoint, "elapsed_hours": round(elapsed/3600,2)}, indent=2))
print("Summary saved.", flush=True)

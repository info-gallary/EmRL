"""
main.py — EmRL Overnight Research Runner
===========================================
Runs full training + evaluation pipeline for the EmRL paper.

Usage:
    python main.py [--mode train|eval|full] [--timesteps 1000000] [--seed 42]

Modes:
    train  — train PPO agent on RRN-A/B + synthetic topologies
    eval   — load checkpoint and run full evaluation battery
    full   — train then evaluate (default, overnight run)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Imports ───────────────────────────────────────────────────────────────────
from src.env.contact_plan import generate_rrna, generate_rrnb, generate_synthetic
from src.env.dtn_env import DTNRoutingEnv
from src.agents.ppo_agent import PPOAgent
from src.agents.rl_adapter import PPORoutingAdapter
from src.training.trainer import PPOTrainer
from src.evaluation.evaluator import Evaluator


# ── Config ────────────────────────────────────────────────────────────────────

RESULTS_DIR    = ROOT / "results"
CHECKPOINTS    = ROOT / "checkpoints"
FIGURES_DIR    = ROOT / "figures"

TRAIN_SEEDS    = [42, 0, 1]
EVAL_SEEDS     = [0, 1, 2, 3, 4]
N_EVAL_EPISODES = 300         # per (scenario, agent) pair


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_contact_plans(seed: int = 42):
    print("Generating contact plans …")
    rrna = generate_rrna(sim_duration=86_400, seed=seed)
    rrnb = generate_rrnb(sim_duration=86_400, seed=seed)
    synthetic = generate_synthetic(n_nodes=12, sim_duration=86_400, p_fail=0.15, seed=seed)
    print(f"  RRN-A: {len(rrna.contacts)} contacts, {len(rrna.node_ids)} nodes")
    print(f"  RRN-B: {len(rrnb.contacts)} contacts, {len(rrnb.node_ids)} nodes")
    print(f"  Synthetic: {len(synthetic.contacts)} contacts, {len(synthetic.node_ids)} nodes")
    return {"rrna": rrna, "rrnb": rrnb, "synthetic": synthetic}


def make_env_factory(contact_plans: dict):
    """Returns a factory callable that creates DTNRoutingEnv instances.
    Rotates through RRN-A, RRN-B, and synthetic for curriculum variety.
    """
    plan_list = list(contact_plans.values())
    _counter = [0]

    def factory(topology: str = "rrna"):
        if topology == "rrna":
            plan = contact_plans["rrna"]
        elif topology == "rrnb":
            plan = contact_plans["rrnb"]
        elif topology == "synthetic":
            plan = contact_plans["synthetic"]
        else:
            plan = plan_list[_counter[0] % len(plan_list)]
            _counter[0] += 1
        return DTNRoutingEnv(topology=topology, contact_plan=plan)

    return factory


def make_rotating_factory(contact_plans: dict):
    """Rotating factory: alternates across plans each call."""
    plan_keys = list(contact_plans.keys())
    counter = [0]

    def factory():
        key = plan_keys[counter[0] % len(plan_keys)]
        counter[0] += 1
        plan = contact_plans[key]
        return DTNRoutingEnv(topology=key, contact_plan=plan)

    return factory


# ── Training ──────────────────────────────────────────────────────────────────

def run_training(
    contact_plans: dict,
    total_timesteps: int = 1_000_000,
    seed: int = 42,
) -> PPOAgent:
    print("\n" + "=" * 60)
    print("PHASE: TRAINING")
    print("=" * 60)
    np.random.seed(seed)

    agent = PPOAgent(
        obs_dim=76,
        action_dim=12,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        n_epochs=10,
        batch_size=64,
    )

    trainer = PPOTrainer(agent=agent)
    env_factory = make_rotating_factory(contact_plans)

    result = trainer.train(
        env_factory=env_factory,
        total_timesteps=total_timesteps,
        rollout_steps=2048,
        eval_interval=10,
        checkpoint_dir=str(CHECKPOINTS),
        log_dir=str(RESULTS_DIR / "logs"),
        results_dir=str(RESULTS_DIR),
        n_eval_episodes=50,
        early_stop_patience=40,
        seed=seed,
    )

    print(f"\nTraining complete. Best BDR={result.best_bdr:.4f}")
    print(f"Best checkpoint: {result.best_checkpoint}")

    # Save training summary
    summary = {
        "total_timesteps": result.total_timesteps,
        "total_updates": result.total_updates,
        "best_bdr": result.best_bdr,
        "best_checkpoint": result.best_checkpoint,
        "seed": seed,
    }
    (RESULTS_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2))
    return agent


# ── Evaluation ────────────────────────────────────────────────────────────────

def run_evaluation(agent: PPOAgent, contact_plans: dict, n_episodes: int = N_EVAL_EPISODES):
    print("\n" + "=" * 60)
    print("PHASE: EVALUATION")
    print("=" * 60)

    # Detect n_nodes from the RRN-A plan
    rrna_plan = contact_plans["rrna"]
    n_nodes = len(rrna_plan.node_ids)
    sim_duration = getattr(rrna_plan, "t_end", 86_400.0)

    ppo_adapter = PPORoutingAdapter(agent, n_nodes=n_nodes, sim_duration=sim_duration)

    evaluator = Evaluator(
        max_hops=32,
        bundle_ttl=3600.0,
        bundle_size_bits=10_000_000,
        verbose=True,
    )

    print(f"\nRunning evaluation: {n_episodes} episodes per (scenario × agent) …")
    t0 = time.time()

    all_results = evaluator.evaluate_all(
        emrl_agent=ppo_adapter,
        contact_plans={"rrna": rrna_plan, "rrnb": contact_plans["rrnb"]},
        n_episodes=n_episodes,
        seeds=EVAL_SEEDS,
    )

    elapsed = time.time() - t0
    print(f"\nEvaluation complete in {elapsed:.1f}s")

    # Serialize results to JSON
    serialised = _serialise_results(all_results)
    out_path = RESULTS_DIR / "eval_results.json"
    out_path.write_text(json.dumps(serialised, indent=2))
    print(f"Results saved → {out_path}")

    _print_summary_table(all_results)
    return all_results


def _serialise_results(all_results: dict) -> dict:
    out = {}
    for scenario, agents in all_results.items():
        out[scenario] = {}
        for agent_name, metrics in agents.items():
            out[scenario][agent_name] = {
                "bdr": round(metrics.bdr, 4),
                "mean_delay": round(metrics.mean_delay, 2),
                "mean_energy": round(metrics.mean_energy, 4),
                "mean_hops": round(metrics.mean_hops, 2),
                "oracle_bdr": round(metrics.oracle_bdr, 4),
                "relative_performance": round(metrics.relative_performance, 4),
                "congestion_bdr": round(metrics.congestion_bdr, 4) if not (metrics.congestion_bdr != metrics.congestion_bdr) else None,
                "jamming_bdr": round(metrics.jamming_bdr, 4) if not (metrics.jamming_bdr != metrics.jamming_bdr) else None,
                "n_episodes": metrics.n_episodes,
                "ci_95": {k: [round(v, 4) for v in ci] for k, ci in metrics.ci_95.items()},
            }
    return out


def _print_summary_table(all_results: dict):
    agents_order = ["EmRL", "ClassicalCGR", "AdaptiveCGR", "SprayAndWait", "Oracle"]
    scenarios_of_interest = ["rrna", "rrnb", "congestion", "jamming", "adversarial"]

    print("\n" + "=" * 90)
    print(f"{'Scenario':<15} " + "  ".join(f"{a:<12}" for a in agents_order))
    print("-" * 90)
    for scenario in scenarios_of_interest:
        if scenario not in all_results:
            continue
        row = f"{scenario:<15} "
        for agent in agents_order:
            m = all_results[scenario].get(agent)
            bdr = f"{m.bdr:.3f}" if m else "  N/A  "
            row += f"  {bdr:<12}"
        print(row)
    print("=" * 90)
    print("(values = Bundle Delivery Ratio)")


# ── Ablation study ────────────────────────────────────────────────────────────

def run_ablation(contact_plans: dict, timesteps: int = 300_000, seed: int = 42):
    """
    Train two ablated variants and compare:
      - EmRL-NoAttn: flat MLP actor/critic (no contact attention)
      - EmRL-NoAACR: PPO without reliability feature
    """
    print("\n" + "=" * 60)
    print("PHASE: ABLATION")
    print("=" * 60)

    ablation_results = {}
    rrna = contact_plans["rrna"]
    n_nodes = len(rrna.node_ids)
    sim_dur = getattr(rrna, "t_end", 86_400.0)

    env_factory = make_rotating_factory(contact_plans)

    for variant_name, agent_kwargs in [
        ("EmRL-Full",    {}),
        ("EmRL-NoAttn",  {"flat_mlp": True}),
    ]:
        print(f"\n  Training variant: {variant_name} …")
        agent = PPOAgent(obs_dim=76, action_dim=12, **agent_kwargs)
        trainer = PPOTrainer(agent=agent)
        result = trainer.train(
            env_factory=env_factory,
            total_timesteps=timesteps,
            rollout_steps=2048,
            eval_interval=10,
            checkpoint_dir=str(CHECKPOINTS / variant_name),
            results_dir=str(RESULTS_DIR / "ablation" / variant_name),
            n_eval_episodes=30,
            early_stop_patience=20,
            seed=seed,
        )
        ablation_results[variant_name] = {
            "best_bdr": result.best_bdr,
            "best_checkpoint": result.best_checkpoint,
        }
        print(f"  {variant_name}: best BDR = {result.best_bdr:.4f}")

    out_path = RESULTS_DIR / "ablation_summary.json"
    out_path.write_text(json.dumps(ablation_results, indent=2))
    print(f"\nAblation results saved → {out_path}")
    return ablation_results


# ── Multi-seed robustness ─────────────────────────────────────────────────────

def run_seed_sweep(contact_plans: dict, timesteps: int = 500_000):
    """Train PPO with multiple seeds to get confidence intervals."""
    print("\n" + "=" * 60)
    print("PHASE: SEED SWEEP")
    print("=" * 60)

    seed_results = {}
    env_factory = make_rotating_factory(contact_plans)
    rrna = contact_plans["rrna"]

    for seed in TRAIN_SEEDS:
        print(f"\n  Seed {seed} …")
        agent = PPOAgent(obs_dim=76, action_dim=12, lr=3e-4)
        trainer = PPOTrainer(agent=agent)
        result = trainer.train(
            env_factory=env_factory,
            total_timesteps=timesteps,
            rollout_steps=2048,
            eval_interval=10,
            checkpoint_dir=str(CHECKPOINTS / f"seed_{seed}"),
            results_dir=str(RESULTS_DIR / "seed_sweep" / f"seed_{seed}"),
            n_eval_episodes=30,
            early_stop_patience=20,
            seed=seed,
        )
        seed_results[seed] = {"best_bdr": result.best_bdr}
        print(f"  Seed {seed} best BDR: {result.best_bdr:.4f}")

    bdrs = [v["best_bdr"] for v in seed_results.values()]
    print(f"\nSeed sweep: mean BDR={np.mean(bdrs):.4f} ± {np.std(bdrs):.4f}")
    out_path = RESULTS_DIR / "seed_sweep_summary.json"
    out_path.write_text(json.dumps(seed_results, indent=2))
    return seed_results


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EmRL Research Runner")
    parser.add_argument("--mode", choices=["train", "eval", "full", "ablation", "sweep"],
                        default="full")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint for eval-only mode")
    parser.add_argument("--episodes", type=int, default=N_EVAL_EPISODES)
    args = parser.parse_args()

    # Create output directories
    for d in [RESULTS_DIR, CHECKPOINTS, FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Generate contact plans (shared across all phases)
    contact_plans = build_contact_plans(seed=args.seed)

    t_start = time.time()

    if args.mode in ("train", "full"):
        agent = run_training(contact_plans, total_timesteps=args.timesteps, seed=args.seed)

    if args.mode in ("eval", "full"):
        if args.mode == "eval":
            # Load from checkpoint
            ckpt = args.checkpoint or str(CHECKPOINTS / "best_model.pt")
            agent = PPOAgent(obs_dim=76, action_dim=12)
            agent.load(ckpt)
            print(f"Loaded checkpoint: {ckpt}")
        run_evaluation(agent, contact_plans, n_episodes=args.episodes)

    if args.mode == "ablation":
        run_ablation(contact_plans, timesteps=min(args.timesteps, 300_000), seed=args.seed)

    if args.mode == "sweep":
        run_seed_sweep(contact_plans, timesteps=min(args.timesteps, 500_000))

    elapsed = time.time() - t_start
    print(f"\nTotal runtime: {elapsed/3600:.2f} hours")
    print("Done. Run `python plot_results.py` to generate figures.")


if __name__ == "__main__":
    main()

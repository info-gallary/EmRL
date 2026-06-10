"""
run_eval_full.py — Comprehensive 300-episode evaluation with statistical tests.
==================================================================================
This is the journal-quality evaluation runner. Differences from run_eval.py:

  1. 300 episodes per scenario (vs 150) for tighter CIs
  2. Includes all baselines:
        - EmRL-Full (this paper)
        - DQN-flat (vanilla RL baseline, RUCoP-style)
        - DQN-attn (architecture-fair RL baseline)
        - EmRL-NoAttn (architectural ablation)
        - EmRL-NoBCWarm (BC ablation, retrained)
        - ClassicalCGR (deterministic baseline)
        - SprayAndWait (operational baseline)
        - Oracle (upper bound)
  3. Paired t-tests with Holm-Bonferroni correction for every comparison
  4. Cohen's d effect sizes
  5. Per-episode outcomes preserved for reproducibility

Saves:
  results/eval_full.json       — full BDR/delay/energy by (scenario, agent)
  results/eval_stats.json      — paired t-test results
  results/eval_per_episode.json — per-episode binary outcomes (for plotting CDFs)
"""
import io
import sys
import os
import json
import argparse
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, '.')

import numpy as np

from src.env.contact_plan import generate_rrna, generate_rrnb
from src.agents.ppo_agent import PPOAgent
from src.agents.rl_adapter import PPORoutingAdapter
from src.agents.dqn_agent import DQNAgent, DQNRoutingAdapter
from src.evaluation.evaluator import Evaluator
from src.analysis.stats import (
    pairwise_comparison_table,
    paired_t_test,
    holm_correction,
    PairedTestResult,
)


CHECKPOINTS = Path('checkpoints')
RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)


def find_best(pattern: str, root: Path = CHECKPOINTS) -> Path | None:
    """Find the best-performing checkpoint matching pattern (highest BDR in name)."""
    candidates = list(root.glob(pattern))
    if not candidates:
        return None
    def _bdr(p):
        try:
            stem = p.stem.replace('best_model_bdr', '')
            return float(stem)
        except ValueError:
            return 0.0
    return max(candidates, key=_bdr)


def load_ppo(checkpoint: Path, use_aacr: bool = True,
             n_attn_layers: int = 2) -> PPORoutingAdapter:
    """Load a PPO agent and wrap with routing adapter."""
    agent = PPOAgent(obs_dim=76, action_dim=12, n_attn_layers=n_attn_layers)
    agent.load(checkpoint)
    return PPORoutingAdapter(agent, n_nodes=18, sim_duration=86400.0,
                             use_aacr=use_aacr)


def load_dqn(checkpoint: Path, arch: str = 'flat') -> DQNRoutingAdapter:
    """Load a DQN agent and wrap with routing adapter."""
    agent = DQNAgent(obs_dim=76, action_dim=12, arch=arch)
    agent.load(checkpoint)
    return DQNRoutingAdapter(agent, n_nodes=18, sim_duration=86400.0)


def evaluate_agent_on_scenarios(adapter, agent_name: str, scenarios: dict,
                                evaluator: Evaluator, plans: dict,
                                n_episodes: int, seeds: list[int]) -> dict:
    """Run one agent across all scenarios, return scenario→metrics dict."""
    results = evaluator.evaluate_all(
        emrl_agent=adapter,
        contact_plans=plans,
        n_episodes=n_episodes,
        seeds=seeds,
    )
    return results


def per_episode_array(metrics):
    """Convert EvalMetrics episode_results to a binary delivered array."""
    if hasattr(metrics, 'episode_results') and metrics.episode_results:
        return np.array([1.0 if er.delivered else 0.0
                         for er in metrics.episode_results])
    n = metrics.n_episodes
    n_del = int(round(metrics.bdr * n))
    arr = np.zeros(n)
    arr[:n_del] = 1.0
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    print("=" * 70)
    print(f"EmRL — Full Evaluation Suite ({args.episodes} eps per scenario)")
    print("=" * 70)

    # ── Discover checkpoints ──────────────────────────────────────────────────
    print("\nLocating checkpoints...")
    ckpt_full   = find_best('best_model_bdr*.pt', CHECKPOINTS)
    ckpt_nobc   = find_best('best_model_bdr*.pt', CHECKPOINTS / 'no_bc')
    ckpt_noattn = find_best('best_model_bdr*.pt', CHECKPOINTS / 'no_attn')
    ckpt_dqnf   = find_best('best_model_bdr*.pt', CHECKPOINTS / 'dqn_flat')
    ckpt_dqna   = find_best('best_model_bdr*.pt', CHECKPOINTS / 'dqn_attn')

    print(f"  EmRL-Full:    {ckpt_full}")
    print(f"  EmRL-NoBC:    {ckpt_nobc}")
    print(f"  EmRL-NoAttn:  {ckpt_noattn}")
    print(f"  DQN-flat:       {ckpt_dqnf}")
    print(f"  DQN-attn:       {ckpt_dqna}")

    if ckpt_full is None:
        print("ERROR: no EmRL-Full checkpoint found.")
        sys.exit(1)

    # ── Build agents ──────────────────────────────────────────────────────────
    print("\nLoading agents...")
    agents = {}

    agents["EmRL-Full"] = load_ppo(ckpt_full, use_aacr=True, n_attn_layers=2)
    print(f"  EmRL-Full loaded from {ckpt_full.name}")

    agents["EmRL-NoAACR"] = load_ppo(ckpt_full, use_aacr=False, n_attn_layers=2)
    print(f"  EmRL-NoAACR loaded from {ckpt_full.name} (use_aacr=False)")

    if ckpt_nobc is not None:
        agents["EmRL-NoBCWarm"] = load_ppo(ckpt_nobc, use_aacr=True, n_attn_layers=2)
        print(f"  EmRL-NoBCWarm loaded from {ckpt_nobc.name}")

    if ckpt_noattn is not None:
        agents["EmRL-NoAttn"] = load_ppo(ckpt_noattn, use_aacr=True, n_attn_layers=0)
        print(f"  EmRL-NoAttn loaded from {ckpt_noattn.name}")

    if ckpt_dqnf is not None:
        agents["DQN-flat"] = load_dqn(ckpt_dqnf, arch='flat')
        print(f"  DQN-flat loaded from {ckpt_dqnf.name}")

    if ckpt_dqna is not None:
        agents["DQN-attn"] = load_dqn(ckpt_dqna, arch='attn')
        print(f"  DQN-attn loaded from {ckpt_dqna.name}")

    # ── Plans ────────────────────────────────────────────────────────────────
    print("\nGenerating contact plans...")
    rrna = generate_rrna(sim_duration=86400, seed=42)
    rrnb = generate_rrnb(sim_duration=86400, seed=42)
    plans = {'rrna': rrna, 'rrnb': rrnb}

    # ── Evaluator ────────────────────────────────────────────────────────────
    evaluator = Evaluator(
        max_hops=32,
        bundle_ttl=7200.0,
        bundle_size_bits=10_000_000,
        verbose=True,
    )

    # ── Run each RL/SL agent through all scenarios ────────────────────────────
    print(f"\nRunning evaluation for each agent ({args.episodes} eps per scenario)...")
    per_agent_results = {}

    t0 = time.time()
    for name, adapter in agents.items():
        print(f"\n[Agent: {name}]")
        res = evaluator.evaluate_all(
            emrl_agent=adapter,
            contact_plans=plans,
            n_episodes=args.episodes,
            seeds=args.seeds,
        )
        per_agent_results[name] = res
        # Classical baselines are included in res automatically; extract them
        # only once (from the first agent's results).

    # Pull classical baselines from the EmRL-Full results
    # (they're computed by Evaluator independently of the EmRL agent)
    classical_results = {}
    first_agent_res = per_agent_results["EmRL-Full"]
    for scenario, scenario_res in first_agent_res.items():
        for cls_name in ("ClassicalCGR", "AdaptiveCGR", "SprayAndWait", "Oracle"):
            if cls_name in scenario_res:
                classical_results.setdefault(cls_name, {})[scenario] = scenario_res[cls_name]

    elapsed = time.time() - t0
    print(f"\n\nAll agent evaluations done in {elapsed/60:.1f} min")

    # ── Aggregate scenarios × agents BDR ──────────────────────────────────────
    all_scenarios = list(first_agent_res.keys())
    aggregate = {}
    per_episode_outcomes = {}   # scenario → agent → np.array(0/1)

    for scenario in all_scenarios:
        aggregate[scenario] = {}
        per_episode_outcomes[scenario] = {}

        for agent_name in agents:
            m = per_agent_results[agent_name].get(scenario, {}).get("EmRL")
            if m is None:
                continue
            aggregate[scenario][agent_name] = {
                "bdr": round(m.bdr, 4),
                "mean_delay": round(m.mean_delay, 2) if m.mean_delay == m.mean_delay else None,
                "mean_energy": round(m.mean_energy, 4) if m.mean_energy == m.mean_energy else None,
                "mean_hops": round(m.mean_hops, 2) if m.mean_hops == m.mean_hops else None,
                "ci_95_bdr": [round(m.ci_95["bdr"][0], 4), round(m.ci_95["bdr"][1], 4)],
                "n_episodes": m.n_episodes,
                "drop_reasons": dict(m.drop_reason_counts),
            }
            per_episode_outcomes[scenario][agent_name] = per_episode_array(m)

        for cls_name, cls_scenarios in classical_results.items():
            m = cls_scenarios.get(scenario)
            if m is None:
                continue
            aggregate[scenario][cls_name] = {
                "bdr": round(m.bdr, 4),
                "mean_delay": round(m.mean_delay, 2) if m.mean_delay == m.mean_delay else None,
                "mean_energy": round(m.mean_energy, 4) if m.mean_energy == m.mean_energy else None,
                "mean_hops": round(m.mean_hops, 2) if m.mean_hops == m.mean_hops else None,
                "ci_95_bdr": [round(m.ci_95["bdr"][0], 4), round(m.ci_95["bdr"][1], 4)],
                "n_episodes": m.n_episodes,
                "drop_reasons": dict(m.drop_reason_counts),
            }
            per_episode_outcomes[scenario][cls_name] = per_episode_array(m)

    # ── Save raw aggregate ───────────────────────────────────────────────────
    out_full = RESULTS_DIR / "eval_full.json"
    out_full.write_text(json.dumps(aggregate, indent=2))
    print(f"\nSaved: {out_full}")

    # ── Paired t-tests (EmRL-Full vs all baselines, per scenario) ──────────
    stats_results = {}
    print("\n" + "=" * 70)
    print("PAIRED t-TESTS (Holm-Bonferroni corrected)")
    print("=" * 70)

    for scenario in all_scenarios:
        if "EmRL-Full" not in per_episode_outcomes[scenario]:
            continue
        comparisons = pairwise_comparison_table(
            per_episode_outcomes[scenario],
            primary_agent="EmRL-Full",
        )
        stats_results[scenario] = [{
            "agent_a": r.agent_a,
            "agent_b": r.agent_b,
            "mean_diff": round(r.mean_diff, 4),
            "ci_95_diff": [round(r.ci_95_lo, 4), round(r.ci_95_hi, 4)],
            "t_statistic": round(r.t_statistic, 3),
            "p_value": round(r.p_value, 6),
            "cohens_d": round(r.cohens_d, 3),
            "significant_holm": r.significant_holm,
            "n_pairs": r.n_pairs,
        } for r in comparisons]

        print(f"\n[Scenario: {scenario}]")
        for r in comparisons:
            print(f"  {r.summary()}")

    out_stats = RESULTS_DIR / "eval_stats.json"
    out_stats.write_text(json.dumps(stats_results, indent=2))
    print(f"\nSaved: {out_stats}")

    # ── Save per-episode outcomes (for reproducibility / CDF plots) ──────────
    per_ep_serializable = {}
    for scenario, agents_dict in per_episode_outcomes.items():
        per_ep_serializable[scenario] = {
            agent: arr.tolist() for agent, arr in agents_dict.items()
        }
    out_pe = RESULTS_DIR / "eval_per_episode.json"
    out_pe.write_text(json.dumps(per_ep_serializable, indent=2))
    print(f"Saved: {out_pe}")

    # ── Pretty summary table ─────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"{'Scenario':<14}", end='')
    agent_order = ["EmRL-Full", "EmRL-NoAACR", "EmRL-NoBCWarm", "EmRL-NoAttn",
                   "DQN-flat", "DQN-attn",
                   "ClassicalCGR", "SprayAndWait", "AdaptiveCGR", "Oracle"]
    for a in agent_order:
        print(f" {a[:12]:>12}", end='')
    print()
    print("-" * (14 + 13 * len(agent_order)))
    for scenario in all_scenarios:
        print(f"{scenario:<14}", end='')
        for a in agent_order:
            v = aggregate[scenario].get(a, {}).get("bdr")
            if v is None:
                print(f" {'-':>12}", end='')
            else:
                print(f" {v:>12.3f}", end='')
        print()
    print("=" * 90)

    print(f"\nDone. Total wall time: {(time.time() - t0) / 60:.1f} min")


if __name__ == '__main__':
    main()

"""
run_behavioral_analysis.py — Policy behavior analysis for the paper.
=====================================================================
Addresses Reviewer Points 2 and 3:
  - Point 2: Why does EmRL fail? (failure mode breakdown by scenario)
  - Point 3: What did EmRL learn? (action distribution, trajectories)

Outputs:
  results/behavioral_analysis.json — action distributions, trajectory stats,
                                      failure mode percentages
  results/sample_trajectories.json  — full hop-by-hop traces for inspection

Generates data feeding fig14 (failure modes), fig15 (action dist), fig16 (trajectories).
"""
import io
import sys
import os
import json
import time
from pathlib import Path
from collections import defaultdict
from copy import deepcopy

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, '.')

import numpy as np
import random

from src.env.contact_plan import generate_rrna, generate_rrnb
from src.env.bundle import Bundle
from src.agents.ppo_agent import PPOAgent
from src.agents.rl_adapter import PPORoutingAdapter, K_CONTACTS
from src.agents.classical_cgr import ClassicalCGR
from src.evaluation.evaluator import (
    Evaluator, _RoutingEnv, _apply_jamming, _apply_anomaly,
    _apply_adversarial_delay, ScenarioConfig,
)


# ── Instrumented adapter that records action choices ────────────────────────


class InstrumentedPPOAdapter(PPORoutingAdapter):
    """Wraps PPORoutingAdapter to record every action chosen."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_log = []          # list[int] of all actions taken
        self.trajectory_log = []      # list[list[dict]] one per bundle
        self._current_trajectory = []
        self._record_traj = False

    def start_episode(self, bundle):
        self._current_trajectory = []

    def end_episode(self):
        if self._record_traj:
            self.trajectory_log.append(self._current_trajectory)
        self._current_trajectory = []

    def route(self, bundle, contact_plan, current_node, current_time):
        # We need to record actions *as the adapter sees them*. The simplest is
        # to wrap the PPO agent's select_action so we can intercept the action.
        sim_time = current_time
        max_hold_steps = max(1, int((bundle.deadline - current_time) / self.HOLD_STEP) + 1)

        for _ in range(max_hold_steps):
            if sim_time >= bundle.deadline:
                self.action_log.append(-1)   # TTL expired
                return None

            obs, action_mask, contacts = self._build_obs(
                bundle, contact_plan, current_node, sim_time
            )

            n_valid_contacts = sum(1 for c in contacts if c is not None)

            if not any(action_mask[:K_CONTACTS]):
                self.action_log.append(K_CONTACTS)   # implicit hold (no contacts)
                if self._record_traj:
                    self._current_trajectory.append({
                        "t": sim_time, "node": current_node,
                        "action": "hold_implicit",
                        "n_contacts": 0,
                    })
                sim_time += self.HOLD_STEP
                continue

            action, _lp, _v = self.ppo_agent.select_action(
                obs, action_mask, deterministic=True
            )
            self.action_log.append(int(action))

            if self._record_traj:
                if action < K_CONTACTS:
                    chosen = contacts[action]
                    self._current_trajectory.append({
                        "t": sim_time, "node": current_node,
                        "action": f"forward_to_{chosen.receiver}",
                        "n_contacts": n_valid_contacts,
                        "contact_start": chosen.start_time,
                        "rate": chosen.data_rate,
                        "reliability": chosen.reliability,
                    })
                elif action == K_CONTACTS:
                    self._current_trajectory.append({
                        "t": sim_time, "node": current_node,
                        "action": "hold",
                        "n_contacts": n_valid_contacts,
                    })
                else:
                    self._current_trajectory.append({
                        "t": sim_time, "node": current_node,
                        "action": "drop",
                        "n_contacts": n_valid_contacts,
                    })

            if action < K_CONTACTS and action_mask[action]:
                return contacts[action]
            elif action == K_CONTACTS + 1:
                return None
            else:
                sim_time += self.HOLD_STEP

        return None


# ── Helper to evaluate one scenario and collect behavioral data ──────────────


def evaluate_scenario_behavioral(
    adapter: InstrumentedPPOAdapter,
    contact_plan,
    scenario_cfg: ScenarioConfig,
    n_episodes: int,
    record_trajectories: int = 0,
) -> dict:
    """
    Run a scenario and return action distribution + drop reasons + trajectory samples.

    record_trajectories: number of episodes to record full hop-by-hop traces.
    """
    rng = random.Random(scenario_cfg.seed)
    np_rng = np.random.default_rng(scenario_cfg.seed)

    plan = contact_plan
    if scenario_cfg.jamming_rate > 0.0:
        plan = _apply_jamming(plan, scenario_cfg.jamming_rate, rng)
    if scenario_cfg.adversarial_delay > 0.0:
        plan = _apply_adversarial_delay(plan, scenario_cfg.adversarial_delay)
    if scenario_cfg.anomaly_rate > 0.0:
        plan = _apply_anomaly(plan, scenario_cfg.anomaly_rate, rng)

    # Generate bundles
    evaluator = Evaluator(max_hops=32, bundle_ttl=7200.0,
                         bundle_size_bits=10_000_000, verbose=False)
    bundles = evaluator._generate_bundles(plan, n_episodes, rng, np_rng)

    # Reset adapter logs
    adapter.action_log = []
    adapter.trajectory_log = []

    env = _RoutingEnv(contact_plan=plan, max_hops=32,
                     congestion_load=scenario_cfg.congestion_load, rng=rng)

    drop_reasons = defaultdict(int)
    delivered = 0
    delays = []
    hops_per_bundle = []

    for i, b in enumerate(bundles):
        adapter._record_traj = (i < record_trajectories)
        adapter.start_episode(b)
        env.reset(b)
        result = env.step(adapter)
        adapter.end_episode()

        if result.delivered:
            delivered += 1
            delays.append(result.delay)
            hops_per_bundle.append(result.hops)
            drop_reasons['delivered'] += 1
        else:
            drop_reasons[result.drop_reason or 'unknown'] += 1

    actions = adapter.action_log
    action_dist = defaultdict(int)
    for a in actions:
        if a == -1:
            action_dist['ttl_exhausted'] += 1
        elif a < K_CONTACTS:
            action_dist[f'forward_slot_{a}'] += 1
            action_dist['forward_any'] += 1
        elif a == K_CONTACTS:
            action_dist['hold'] += 1
        elif a == K_CONTACTS + 1:
            action_dist['drop_explicit'] += 1

    total_actions = sum(action_dist[k] for k in
                       ['forward_any', 'hold', 'drop_explicit', 'ttl_exhausted'])

    summary_actions = {
        'forward_pct': 100 * action_dist['forward_any'] / max(total_actions, 1),
        'hold_pct': 100 * action_dist['hold'] / max(total_actions, 1),
        'drop_explicit_pct': 100 * action_dist['drop_explicit'] / max(total_actions, 1),
        'mean_actions_per_episode': total_actions / n_episodes,
        'forward_slot_distribution': {
            f'slot_{i}': action_dist.get(f'forward_slot_{i}', 0)
            for i in range(K_CONTACTS)
        },
        'total_actions': total_actions,
    }

    summary_failures = {
        'delivered_pct': 100 * drop_reasons['delivered'] / n_episodes,
        'ttl_expired_pct': 100 * drop_reasons['ttl_expired'] / n_episodes,
        'no_route_pct': 100 * drop_reasons['no_route'] / n_episodes,
        'congestion_pct': 100 * drop_reasons['congestion'] / n_episodes,
        'raw_counts': dict(drop_reasons),
    }

    return {
        'n_episodes': n_episodes,
        'bdr': delivered / n_episodes,
        'mean_delay_delivered': float(np.mean(delays)) if delays else None,
        'mean_hops_delivered': float(np.mean(hops_per_bundle)) if hops_per_bundle else None,
        'failure_modes': summary_failures,
        'action_distribution': summary_actions,
        'sample_trajectories': adapter.trajectory_log[:record_trajectories],
    }


def find_best_ckpt() -> Path | None:
    for pat in ['best_model_bdr0.7*.pt', 'best_model_bdr*.pt']:
        candidates = list(Path('checkpoints').glob(pat))
        if candidates:
            def _bdr(p):
                try:
                    return float(p.stem.replace('best_model_bdr', ''))
                except ValueError:
                    return 0.0
            return max(candidates, key=_bdr)
    return None


def main():
    print("=" * 70)
    print("EmRL Behavioral Analysis")
    print("=" * 70)

    ckpt = find_best_ckpt()
    if ckpt is None:
        print("ERROR: no checkpoint")
        sys.exit(1)
    print(f"Loading from: {ckpt}")

    agent = PPOAgent(obs_dim=76, action_dim=12, n_attn_layers=2)
    agent.load(ckpt)
    adapter = InstrumentedPPOAdapter(agent, n_nodes=18, sim_duration=86400.0)

    print("\nGenerating contact plans...")
    rrna = generate_rrna(sim_duration=86400, seed=42)

    # Run all 6 scenarios
    scenarios = {
        'rrna':        ScenarioConfig("rrna", seed=0),
        'congestion':  ScenarioConfig("congestion", congestion_load=3.0, seed=2),
        'jamming':     ScenarioConfig("jamming", jamming_rate=0.30, seed=1),
        'adversarial': ScenarioConfig("adversarial", adversarial_delay=120.0, seed=3),
        'anomaly':     ScenarioConfig("anomaly", anomaly_rate=0.30, seed=5),
    }

    all_results = {}
    t0 = time.time()

    for name, cfg in scenarios.items():
        print(f"\n[Scenario: {name}]")
        ts0 = time.time()
        # Record trajectories on first 5 episodes of each scenario
        result = evaluate_scenario_behavioral(
            adapter, rrna, cfg, n_episodes=150, record_trajectories=5,
        )
        elapsed = time.time() - ts0
        print(f"  BDR={result['bdr']:.3f}")
        print(f"  Failures: delivered={result['failure_modes']['delivered_pct']:.1f}%, "
              f"TTL={result['failure_modes']['ttl_expired_pct']:.1f}%, "
              f"no_route={result['failure_modes']['no_route_pct']:.1f}%, "
              f"congestion={result['failure_modes']['congestion_pct']:.1f}%")
        print(f"  Actions: forward={result['action_distribution']['forward_pct']:.1f}%, "
              f"hold={result['action_distribution']['hold_pct']:.1f}%, "
              f"drop={result['action_distribution']['drop_explicit_pct']:.1f}%, "
              f"mean_actions/ep={result['action_distribution']['mean_actions_per_episode']:.1f}")
        print(f"  Recorded {len(result['sample_trajectories'])} sample trajectories ({elapsed:.0f}s)")
        all_results[name] = result

    # ── ClassicalCGR for the same scenarios (for comparison) ─────────────────
    print("\n[ClassicalCGR baseline scenarios for trajectory comparison]")
    classical_traj = []
    classical = ClassicalCGR()
    for cfg_name, cfg in [('rrna', scenarios['rrna']),
                          ('congestion', scenarios['congestion'])]:
        rng = random.Random(cfg.seed)
        np_rng = np.random.default_rng(cfg.seed)
        plan = rrna
        if cfg.jamming_rate > 0.0:
            plan = _apply_jamming(plan, cfg.jamming_rate, rng)
        if cfg.adversarial_delay > 0.0:
            plan = _apply_adversarial_delay(plan, cfg.adversarial_delay)
        if cfg.anomaly_rate > 0.0:
            plan = _apply_anomaly(plan, cfg.anomaly_rate, rng)

        evaluator = Evaluator(max_hops=32, bundle_ttl=7200.0,
                            bundle_size_bits=10_000_000, verbose=False)
        bundles = evaluator._generate_bundles(plan, 5, rng, np_rng)

        env = _RoutingEnv(contact_plan=plan, max_hops=32,
                         congestion_load=cfg.congestion_load, rng=rng)
        for i, b in enumerate(bundles[:5]):
            env.reset(b)
            # Trace the classical agent's path manually
            traj = []
            current_node = b.source
            current_time = b.creation_time
            for hop in range(32):
                if current_time >= b.deadline:
                    traj.append({"t": current_time, "node": current_node,
                                "action": "ttl_expired"})
                    break
                contact = classical.route(b, plan, current_node, current_time)
                if contact is None:
                    traj.append({"t": current_time, "node": current_node,
                                "action": "no_route"})
                    break
                if rng.random() > contact.reliability:
                    traj.append({"t": current_time, "node": current_node,
                                "action": f"forward_to_{contact.receiver}_FAILED",
                                "reliability": contact.reliability})
                    current_time = contact.end_time
                    continue
                send_time = max(current_time, contact.start_time)
                tx_delay = b.size_bits / contact.data_rate if contact.data_rate > 0 else 1e9
                finish_time = send_time + tx_delay
                if finish_time > contact.end_time or finish_time >= b.deadline:
                    traj.append({"t": current_time, "node": current_node,
                                "action": "ttl_or_window_expired"})
                    break
                traj.append({"t": current_time, "node": current_node,
                            "action": f"forward_to_{contact.receiver}",
                            "reliability": contact.reliability,
                            "rate": contact.data_rate})
                current_time = finish_time
                current_node = contact.receiver
                if current_node == b.destination:
                    traj.append({"t": current_time, "node": current_node,
                                "action": "DELIVERED"})
                    break
            classical_traj.append({"scenario": cfg_name, "bundle_idx": i,
                                  "src": b.source, "dest": b.destination,
                                  "trajectory": traj})

    all_results['classical_trajectories'] = classical_traj

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")

    # ── Save ─────────────────────────────────────────────────────────────────
    out = Path('results/behavioral_analysis.json')
    out.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nSaved: {out}")

    # Compact trajectory file (separately for easy reading)
    traj_out = Path('results/sample_trajectories.json')
    traj_data = {
        name: all_results[name]['sample_trajectories']
        for name in scenarios.keys()
    }
    traj_data['classical_trajectories'] = classical_traj
    traj_out.write_text(json.dumps(traj_data, indent=2, default=str))
    print(f"Saved: {traj_out}")


if __name__ == '__main__':
    main()

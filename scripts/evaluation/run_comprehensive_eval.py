"""
run_comprehensive_eval.py — Top-journal evaluation battery for redesigned EmRL.
=================================================================================
Produces:
  1. MULTI-TOPOLOGY + SCALABILITY table — BDR across RRN-A, RRN-B, and Walker
     constellations at 23 / 52 / 100 nodes (standard + hub-jamming anomaly),
     for EmRL vs ClassicalCGR (RFC 6260), RUCoP-style (reliability-aware,
     reimpl.), Spray-and-Wait, and Oracle.
  2. STATISTICAL RIGOR on RRN-A — paired t-tests vs every baseline, Holm-
     Bonferroni correction, Cohen's d, 95% CIs (300 episodes, per-episode
     delivery outcomes).

Saves: results/comprehensive_eval.json, results/comprehensive_stats.json
"""
import io, sys, os, json, random
from pathlib import Path
from copy import deepcopy
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, '.')

import numpy as np

from src.env.contact_plan import generate_rrna, generate_rrnb, generate_walker
from src.agents.ppo_agent import PPOAgent
from src.agents.rl_adapter import PPORoutingAdapter
from src.agents.classical_cgr import ClassicalCGR, RUCoPRouter, SprayAndWait
from src.agents.oracle import OracleCGR
from src.evaluation.evaluator import Evaluator, _RoutingEnv
from src.evaluation.metrics import compute_metrics
from src.analysis.stats import (pairwise_comparison_table,
                                per_episode_outcomes_from_metrics)


def find_ckpt():
    ov = os.environ.get("EmRL_CKPT")
    if ov and Path(ov).exists(): return ov
    f = Path('checkpoints/final/emrl_main_k16.pt')
    return str(f) if f.exists() else None


def jam_top_hubs(plan, n_hubs, rel_lo, rel_hi, seed):
    """Profile CGR's most-used transit nodes, then degrade them."""
    rng = random.Random(seed); np_rng = np.random.default_rng(seed)
    ev = Evaluator(max_hops=32, bundle_ttl=7200.0, bundle_size_bits=10_000_000, verbose=False)
    bundles = ev._generate_bundles(plan, 150, rng, np_rng)
    cgr = ClassicalCGR(); freq = defaultdict(int)
    gs_start = plan.n_nodes - 4  # last ~4 nodes are ground stations
    for b in bundles:
        node, t = b.source, b.creation_time
        for _ in range(32):
            if node == b.destination: break
            c = cgr.route(b, plan, node, t)
            if c is None: break
            r = c.receiver
            if r < gs_start and r != b.source and r != b.destination:
                freq[r] += 1
            t = max(t, c.start_time) + b.size_bits / max(c.data_rate, 1)
            node = r
    hubs = [k for k, _ in sorted(freq.items(), key=lambda x: -x[1])][:n_hubs]
    p = deepcopy(plan); hs = set(hubs); jr = random.Random(seed + 1)
    for c in p.contacts:
        if c.sender in hs or c.receiver in hs:
            c.reliability = jr.uniform(rel_lo, rel_hi)
    return p, hubs


def evaluate(plan, agents, n_ep, seed, want_metrics=False):
    rng = random.Random(seed); np_rng = np.random.default_rng(seed)
    ev = Evaluator(max_hops=32, bundle_ttl=7200.0, bundle_size_bits=10_000_000, verbose=False)
    bundles = ev._generate_bundles(plan, n_ep, rng, np_rng)
    oracle_bdr = OracleCGR().compute_oracle_bdr(bundles, plan)
    env = _RoutingEnv(plan, max_hops=32, rng=random.Random(seed))
    out = {}; metrics = {}
    for name, ag in agents.items():
        eps = [ev.run_episode(ag, env, b) for b in bundles]
        m = compute_metrics(eps, oracle_bdr)
        out[name] = round(m.bdr, 4)
        metrics[name] = m
    out['Oracle'] = round(oracle_bdr, 4)
    return (out, metrics) if want_metrics else out


def main():
    ckpt = find_ckpt()
    K = 16; OBS = 6 + K*8
    print("=" * 92, flush=True)
    print("COMPREHENSIVE EVALUATION — redesigned EmRL (destination-aware, K=16)", flush=True)
    print(f"Checkpoint: {ckpt}", flush=True)
    print("=" * 92, flush=True)

    ppo = PPOAgent(obs_dim=OBS, action_dim=K+2); ppo.load(ckpt)

    def make_agents(n_nodes):
        return {
            "EmRL":       PPORoutingAdapter(ppo, n_nodes=n_nodes, sim_duration=86400.0, use_aacr=True, k_contacts=K),
            "ClassicalCGR": ClassicalCGR(),
            "RUCoP":        RUCoPRouter(),
            "SprayAndWait": SprayAndWait(L=2),
        }

    topos = {
        "RRN-A (18)":     (generate_rrna(sim_duration=86400, seed=42), 18, 300),
        "RRN-B (18)":     (generate_rrnb(sim_duration=86400, seed=42), 18, 300),
        "Walker-23":      (generate_walker(4, 5, 3, seed=42), 23, 200),
        "Walker-52":      (generate_walker(6, 8, 4, seed=42), 52, 150),
        "Walker-100":     (generate_walker(8, 12, 4, seed=42), 100, 100),
    }

    results = {}
    print(f"\n{'Topology':<14}{'Cond':<10}{'EmRL':>8}{'CGR':>8}{'RUCoP':>8}{'S&W':>8}{'Oracle':>8}", flush=True)
    print("-" * 72, flush=True)
    for tname, (plan, nn, neps) in topos.items():
        agents = make_agents(nn)
        # standard
        r_std = evaluate(plan, agents, neps, seed=0)
        # hub-jamming anomaly (moderate, detours exist)
        pj, hubs = jam_top_hubs(plan, max(2, nn // 12), 0.2, 0.4, seed=7)
        r_jam = evaluate(pj, agents, neps, seed=1)
        results[tname] = {"n_nodes": nn, "episodes": neps,
                          "standard": r_std, "jammed": r_jam, "jam_hubs": hubs}
        for cond, r in [("standard", r_std), ("jammed", r_jam)]:
            print(f"{tname:<14}{cond:<10}{r['EmRL']:>8.3f}{r['ClassicalCGR']:>8.3f}"
                  f"{r['RUCoP']:>8.3f}{r['SprayAndWait']:>8.3f}{r['Oracle']:>8.3f}", flush=True)

    Path('results/comprehensive_eval.json').write_text(json.dumps(results, indent=2, default=str))

    # ── Statistical rigor on RRN-A (training topology), 300 eps ──────────────
    print("\n" + "=" * 92, flush=True)
    print("STATISTICAL RIGOR (RRN-A, 300 episodes) — paired t-tests + Holm + Cohen's d", flush=True)
    print("=" * 92, flush=True)
    plan = topos["RRN-A (18)"][0]
    agents = make_agents(18)
    _, mets = evaluate(plan, agents, 300, seed=0, want_metrics=True)
    outcomes = {name: per_episode_outcomes_from_metrics(m) for name, m in mets.items()}
    stats_results = pairwise_comparison_table(outcomes, primary_agent="EmRL")
    stats_dump = {}
    print(f"\n{'Comparison':<28}{'Δ mean':>9}{'Cohen d':>9}{'p-value':>11}{'Holm sig':>10}", flush=True)
    print("-" * 70, flush=True)
    for r in stats_results:
        delta = float(np.mean(outcomes['EmRL']) - np.mean(outcomes[r.agent_b]))
        print(f"EmRL vs {r.agent_b:<16}{delta:>+9.3f}{r.cohens_d:>9.2f}"
              f"{r.p_value:>11.2e}{str(r.significant_holm):>10}", flush=True)
        stats_dump[r.agent_b] = {
            "delta_mean": round(delta, 4), "cohens_d": round(r.cohens_d, 3),
            "p_value": r.p_value, "ci95": [round(r.ci_95_lo, 4), round(r.ci_95_hi, 4)],
            "significant_holm": bool(r.significant_holm),
        }
    out_stats = {"standard": stats_dump}

    # ── Statistical rigor UNDER ANOMALY (jammed RRN-A) ───────────────────────
    print("\n" + "-" * 92, flush=True)
    print("STATISTICAL RIGOR UNDER ANOMALY (RRN-A hub-jammed, 300 ep)", flush=True)
    print("-" * 92, flush=True)
    pj, _ = jam_top_hubs(plan, 2, 0.2, 0.4, seed=7)
    _, mets_j = evaluate(pj, agents, 300, seed=1, want_metrics=True)
    outcomes_j = {name: per_episode_outcomes_from_metrics(m) for name, m in mets_j.items()}
    stats_j = pairwise_comparison_table(outcomes_j, primary_agent="EmRL")
    jdump = {}
    print(f"\n{'Comparison':<28}{'Δ mean':>9}{'Cohen d':>9}{'p-value':>11}{'Holm sig':>10}", flush=True)
    print("-" * 70, flush=True)
    for r in stats_j:
        delta = float(np.mean(outcomes_j['EmRL']) - np.mean(outcomes_j[r.agent_b]))
        print(f"EmRL vs {r.agent_b:<16}{delta:>+9.3f}{r.cohens_d:>9.2f}"
              f"{r.p_value:>11.2e}{str(r.significant_holm):>10}", flush=True)
        jdump[r.agent_b] = {
            "delta_mean": round(delta, 4), "cohens_d": round(r.cohens_d, 3),
            "p_value": r.p_value, "ci95": [round(r.ci_95_lo, 4), round(r.ci_95_hi, 4)],
            "significant_holm": bool(r.significant_holm),
        }
    out_stats["anomaly"] = jdump

    Path('results/comprehensive_stats.json').write_text(json.dumps(out_stats, indent=2, default=str))
    print("\nSaved: results/comprehensive_eval.json, results/comprehensive_stats.json", flush=True)


if __name__ == '__main__':
    main()

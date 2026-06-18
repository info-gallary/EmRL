"""
run_generalization.py — Reviewer #8: Zero-shot generalization experiments.
===========================================================================
The K=16-trained EmRL checkpoint (no retraining, no fine-tuning) is evaluated
on topologies and reliability distributions never seen during training:

  G1. Topology scale: Walker-23 / Walker-52 / Walker-100  (3, 6, 8 planes)
  G2. Reliability distributions: uniform-high / uniform-low / bimodal / Weibull
  G3. Orbital parameters: altitude 400 km vs 800 km vs 1200 km (period changes)
  G4. Attack patterns: single-hub / two-hub / random 30% link failure

Compares EmRL vs ClassicalCGR vs RUCoP on all conditions.
Saves: results/generalization.json
"""
import io, sys, os, json, random, math
from pathlib import Path
from copy import deepcopy
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, '.')

import numpy as np
from src.env.contact_plan import generate_rrna, generate_walker, ContactPlan
from src.agents.ppo_agent import PPOAgent
from src.agents.rl_adapter import PPORoutingAdapter
from src.agents.classical_cgr import ClassicalCGR, RUCoPRouter
from src.agents.oracle import OracleCGR
from src.evaluation.evaluator import Evaluator, _RoutingEnv
from src.evaluation.metrics import compute_metrics

K = 16; OBS = 6 + K * 9; N_EP = 150; CKPT = 'checkpoints/final/emrl_main_k16.pt'

def load_emrl(plan):
    ppo = PPOAgent(obs_dim=OBS, action_dim=K+2); ppo.load(CKPT)
    return PPORoutingAdapter(ppo, n_nodes=plan.n_nodes, sim_duration=86400.0, use_aacr=True, k_contacts=K)

def eval_plan(plan, seed=0, n_ep=N_EP):
    rng = random.Random(seed); np_rng = np.random.default_rng(seed)
    ev = Evaluator(max_hops=32, bundle_ttl=7200.0, bundle_size_bits=10_000_000, verbose=False)
    bundles = ev._generate_bundles(plan, n_ep, rng, np_rng)
    oracle = OracleCGR().compute_oracle_bdr(bundles, plan)
    env = _RoutingEnv(plan, max_hops=32, rng=random.Random(seed))
    out = {}
    agents = {"EmRL": load_emrl(plan), "CGR": ClassicalCGR(), "RUCoP": RUCoPRouter()}
    for name, ag in agents.items():
        eps = [ev.run_episode(ag, env, b) for b in bundles]
        m = compute_metrics(eps, oracle)
        out[name] = {"bdr": round(m.bdr, 4), "delay": round(m.mean_delay, 1), "oracle": round(oracle, 4)}
    return out

def perturb_reliability(plan, dist, seed):
    p = deepcopy(plan); rng = random.Random(seed)
    for c in p.contacts:
        if dist == "uniform_low":
            c.reliability = rng.uniform(0.3, 0.6)
        elif dist == "uniform_high":
            c.reliability = rng.uniform(0.85, 0.99)
        elif dist == "bimodal":
            c.reliability = rng.choice([rng.uniform(0.1, 0.3), rng.uniform(0.8, 0.99)])
        elif dist == "weibull":
            raw = np.random.default_rng(seed).weibull(2.0)
            c.reliability = float(np.clip(raw, 0.05, 0.99))
    if hasattr(p, '_potential_to'): p._potential_to = {}
    if hasattr(p, '_dist_to'): p._dist_to = {}
    return p

def attack_pattern(plan, kind, seed):
    p = deepcopy(plan); rng = random.Random(seed)
    sats = [n for n in range(plan.n_nodes - 4)]
    if kind == "single_hub":
        hub = rng.choice(sats)
        for c in p.contacts:
            if c.sender == hub or c.receiver == hub:
                c.reliability = rng.uniform(0.05, 0.2)
    elif kind == "two_hub":
        hubs = set(rng.sample(sats, min(2, len(sats))))
        for c in p.contacts:
            if c.sender in hubs or c.receiver in hubs:
                c.reliability = rng.uniform(0.05, 0.2)
    elif kind == "random_30":
        for c in p.contacts:
            if rng.random() < 0.30:
                c.reliability = rng.uniform(0.05, 0.25)
    if hasattr(p, '_potential_to'): p._potential_to = {}
    if hasattr(p, '_dist_to'): p._dist_to = {}
    return p

def main():
    print("=" * 80, flush=True)
    print("ZERO-SHOT GENERALIZATION — same K=16 checkpoint, unseen conditions", flush=True)
    print("=" * 80, flush=True)
    results = {}

    # G1: Topology scale
    print("\n[G1] Topology scale", flush=True)
    for tname, kwargs in [
        ("Walker-23", dict(n_planes=3, sats_per_plane=7, n_ground=2, seed=77)),
        ("Walker-52", dict(n_planes=4, sats_per_plane=12, n_ground=4, seed=77)),
        ("Walker-100",dict(n_planes=8, sats_per_plane=12, n_ground=4, seed=77)),
    ]:
        plan = generate_walker(**kwargs)
        r = eval_plan(plan, seed=5)
        results[f"G1_{tname}"] = r
        print(f"  {tname}: EmRL={r['EmRL']['bdr']:.3f}  CGR={r['CGR']['bdr']:.3f}  "
              f"RUCoP={r['RUCoP']['bdr']:.3f}  oracle={r['EmRL']['oracle']:.3f}", flush=True)

    # G2: Reliability distribution
    print("\n[G2] Reliability distributions (Walker-23)", flush=True)
    base23 = generate_walker(n_planes=3, sats_per_plane=7, n_ground=2, seed=99)
    for dist in ("uniform_high", "uniform_low", "bimodal", "weibull"):
        plan = perturb_reliability(base23, dist, seed=20)
        r = eval_plan(plan, seed=6)
        results[f"G2_{dist}"] = r
        print(f"  {dist:<15}: EmRL={r['EmRL']['bdr']:.3f}  CGR={r['CGR']['bdr']:.3f}", flush=True)

    # G3: Orbital altitude (different contact window timing via orbital_period)
    print("\n[G3] Orbital altitude (via period)", flush=True)
    for label, period in [("400km", 5520.0), ("800km", 6069.0), ("1200km", 6750.0)]:
        plan = generate_walker(n_planes=3, sats_per_plane=7, n_ground=2, orbital_period=period, seed=33)
        r = eval_plan(plan, seed=7)
        results[f"G3_{label}"] = r
        print(f"  alt {label}: EmRL={r['EmRL']['bdr']:.3f}  CGR={r['CGR']['bdr']:.3f}", flush=True)

    # G4: Attack patterns
    print("\n[G4] Attack patterns (RRN-A)", flush=True)
    base_rrna = generate_rrna(sim_duration=86400, seed=42)
    for kind in ("single_hub", "two_hub", "random_30"):
        plan = attack_pattern(base_rrna, kind, seed=30)
        r = eval_plan(plan, seed=8)
        results[f"G4_{kind}"] = r
        print(f"  {kind:<12}: EmRL={r['EmRL']['bdr']:.3f}  CGR={r['CGR']['bdr']:.3f}", flush=True)

    # Summary: how often does EmRL rank >= CGR?
    print("\n" + "=" * 80, flush=True)
    emrl_wins = sum(1 for r in results.values()
                    if r.get("EmRL", {}).get("bdr", 0) >= r.get("CGR", {}).get("bdr", 0))
    print(f"EmRL >= CGR in {emrl_wins}/{len(results)} generalization conditions.", flush=True)

    Path('results/generalization.json').write_text(json.dumps(results, indent=2, default=str))
    print("Saved: results/generalization.json", flush=True)

if __name__ == '__main__':
    main()

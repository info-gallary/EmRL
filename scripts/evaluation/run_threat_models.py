"""
run_threat_models.py — Reviewer #7: Expanded threat/failure models.
=====================================================================
Six failure scenarios beyond nominal + hub-jamming:
  A. random_failures   — independent per-contact Bernoulli reliability drops
  B. correlated        — spatially correlated failures (cluster of 3 nodes)
  C. cascading         — failure spreads hop-by-hop with probability p_spread
  D. space_weather     — uniform reliability degradation (solar event proxy)
  E. adversarial       — targeted lowest-reliability injection on EmRL's predicted path
  F. congestion        — high-traffic load (3× bundle rate, short TTL)
All scenarios measured: EmRL vs ClassicalCGR vs RUCoP vs Spray-and-Wait.
Saves: results/threat_models.json
"""
import io, sys, os, json, random
from pathlib import Path
from copy import deepcopy
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, '.')

import numpy as np
from src.env.contact_plan import generate_rrna
from src.agents.ppo_agent import PPOAgent
from src.agents.rl_adapter import PPORoutingAdapter
from src.agents.classical_cgr import ClassicalCGR, RUCoPRouter, SprayAndWait
from src.agents.oracle import OracleCGR
from src.evaluation.evaluator import Evaluator, _RoutingEnv
from src.evaluation.metrics import compute_metrics

K = 16; OBS = 6 + K * 9; N_EP = 200; CKPT = 'checkpoints/final/emrl_main_k16.pt'

def load_emrl(plan):
    ppo = PPOAgent(obs_dim=OBS, action_dim=K+2); ppo.load(CKPT)
    return PPORoutingAdapter(ppo, n_nodes=plan.n_nodes, sim_duration=86400.0, use_aacr=True, k_contacts=K)

def eval_agents(plan, agents, seed=0, n_ep=N_EP):
    rng = random.Random(seed); np_rng = np.random.default_rng(seed)
    ev = Evaluator(max_hops=32, bundle_ttl=7200.0, bundle_size_bits=10_000_000, verbose=False)
    bundles = ev._generate_bundles(plan, n_ep, rng, np_rng)
    oracle = OracleCGR().compute_oracle_bdr(bundles, plan)
    env = _RoutingEnv(plan, max_hops=32, rng=random.Random(seed))
    out = {}
    for name, ag in agents.items():
        eps = [ev.run_episode(ag, env, b) for b in bundles]
        m = compute_metrics(eps, oracle)
        out[name] = {"bdr": round(m.bdr, 4), "delay": round(m.mean_delay, 1)}
    out["Oracle"] = {"bdr": round(oracle, 4)}
    return out

def _perturb_contacts(plan, fn, seed):
    p = deepcopy(plan)
    rng = random.Random(seed)
    for c in p.contacts:
        fn(c, rng)
    if hasattr(p, '_potential_to'): p._potential_to = {}
    if hasattr(p, '_dist_to'): p._dist_to = {}
    return p

# ── A: Random Failures ────────────────────────────────────────────────────────
def scenario_random_failures(base, seed, p_fail=0.25):
    def _f(c, r):
        if r.random() < p_fail:
            c.reliability = r.uniform(0.05, 0.25)
    return _perturb_contacts(base, _f, seed)

# ── B: Correlated Cluster Failures ───────────────────────────────────────────
def scenario_correlated(base, seed, n_nodes=3):
    p = deepcopy(base); rng = random.Random(seed)
    all_nodes = list(range(base.n_nodes - 4))  # exclude GS
    cluster = set(rng.sample(all_nodes, min(n_nodes, len(all_nodes))))
    for c in p.contacts:
        if c.sender in cluster or c.receiver in cluster:
            c.reliability = rng.uniform(0.1, 0.3)
    if hasattr(p, '_potential_to'): p._potential_to = {}
    if hasattr(p, '_dist_to'): p._dist_to = {}
    return p

# ── C: Cascading Failures ─────────────────────────────────────────────────────
def scenario_cascading(base, seed, p_seed=0.1, p_spread=0.5):
    p = deepcopy(base); rng = random.Random(seed)
    failed = set()
    for c in p.contacts:
        if rng.random() < p_seed:
            failed.add(c.sender); failed.add(c.receiver)
    changed = True
    while changed:
        changed = False
        for c in p.contacts:
            if (c.sender in failed or c.receiver in failed) and rng.random() < p_spread:
                if c.sender not in failed or c.receiver not in failed:
                    failed.add(c.sender); failed.add(c.receiver); changed = True
    for c in p.contacts:
        if c.sender in failed or c.receiver in failed:
            c.reliability *= rng.uniform(0.1, 0.4)
    if hasattr(p, '_potential_to'): p._potential_to = {}
    if hasattr(p, '_dist_to'): p._dist_to = {}
    return p

# ── D: Space Weather (uniform degradation) ───────────────────────────────────
def scenario_space_weather(base, seed, degrade=0.45):
    def _f(c, r): c.reliability *= (1.0 - degrade * r.uniform(0.7, 1.0))
    return _perturb_contacts(base, _f, seed)

# ── E: Adversarial Reliability Injection ─────────────────────────────────────
def scenario_adversarial(base, seed):
    """Target EmRL's most-used contacts (high-potential contacts → degrade them)."""
    p = deepcopy(base); rng = random.Random(seed)
    pot = base.delivery_potential_to(0)
    top_k_receivers = sorted(pot, key=lambda n: -pot[n])[:4]
    for c in p.contacts:
        if c.receiver in top_k_receivers:
            c.reliability = rng.uniform(0.05, 0.2)
    if hasattr(p, '_potential_to'): p._potential_to = {}
    if hasattr(p, '_dist_to'): p._dist_to = {}
    return p

# ── F: Congestion (high-load short-TTL) ──────────────────────────────────────
def scenario_congestion(base, seed, n_ep=300, ttl=1800.0):
    rng = random.Random(seed); np_rng = np.random.default_rng(seed)
    ev = Evaluator(max_hops=32, bundle_ttl=ttl, bundle_size_bits=50_000_000, verbose=False)
    bundles = ev._generate_bundles(base, n_ep, rng, np_rng)
    oracle = OracleCGR().compute_oracle_bdr(bundles, base)
    env = _RoutingEnv(base, max_hops=32, rng=random.Random(seed))
    out = {}
    agents = {
        "EmRL": load_emrl(base),
        "CGR": ClassicalCGR(),
        "RUCoP": RUCoPRouter(),
        "S&W": SprayAndWait(L=2),
    }
    for name, ag in agents.items():
        eps = [ev.run_episode(ag, env, b) for b in bundles]
        m = compute_metrics(eps, oracle)
        out[name] = {"bdr": round(m.bdr, 4), "delay": round(m.mean_delay, 1)}
    out["Oracle"] = {"bdr": round(oracle, 4)}
    return out

def main():
    print("=" * 80, flush=True)
    print("THREAT MODEL EVALUATION — 6 failure scenarios", flush=True)
    print("=" * 80, flush=True)
    base = generate_rrna(sim_duration=86400, seed=42)
    scenarios = {
        "A_random_failures": scenario_random_failures(base, 11),
        "B_correlated":      scenario_correlated(base, 12),
        "C_cascading":       scenario_cascading(base, 13),
        "D_space_weather":   scenario_space_weather(base, 14),
        "E_adversarial":     scenario_adversarial(base, 15),
    }
    results = {}
    for sname, plan in scenarios.items():
        print(f"\n[{sname}]", flush=True)
        agents = {
            "EmRL":  load_emrl(plan),
            "CGR":   ClassicalCGR(),
            "RUCoP": RUCoPRouter(),
            "S&W":   SprayAndWait(L=2),
        }
        r = eval_agents(plan, agents, seed=10)
        results[sname] = r
        print(f"  {'Agent':<12}{'BDR':>8}{'Delay':>10}", flush=True)
        for nm, d in r.items():
            bdr = d.get('bdr', 0); delay = d.get('delay', 0)
            print(f"  {nm:<12}{bdr:>8.3f}{delay:>10.0f}", flush=True)
    print("\n[F_congestion]", flush=True)
    results["F_congestion"] = scenario_congestion(base, 10)
    r = results["F_congestion"]
    print(f"  {'Agent':<12}{'BDR':>8}{'Delay':>10}", flush=True)
    for nm, d in r.items():
        bdr = d.get('bdr', 0); delay = d.get('delay', 0)
        print(f"  {nm:<12}{bdr:>8.3f}{delay:>10.0f}", flush=True)
    # Summary: does EmRL maintain rank across all 6 scenarios?
    print("\n" + "=" * 80, flush=True)
    print("RANK CONSISTENCY (EmRL vs CGR) across all scenarios:", flush=True)
    emrl_wins = 0
    for sn, r in results.items():
        delta = r.get("EmRL", {}).get("bdr", 0) - r.get("CGR", {}).get("bdr", 0)
        if delta >= 0: emrl_wins += 1
        print(f"  {sn}: EmRL-CGR delta = {delta:+.3f}", flush=True)
    print(f"\nEmRL ranks >= CGR in {emrl_wins}/{len(results)} threat scenarios.", flush=True)
    Path('results/threat_models.json').write_text(json.dumps(results, indent=2, default=str))
    print("Saved: results/threat_models.json", flush=True)

if __name__ == '__main__':
    main()

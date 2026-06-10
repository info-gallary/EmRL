"""
run_inference_cost.py — Per-decision inference cost: EmRL vs CGR vs RUCoP.
===========================================================================
Honest efficiency claim: EmRL makes each routing decision with a single
fixed-size neural forward pass over its K candidate contacts, so per-decision
cost is ~constant in network size. ClassicalCGR and RUCoP run a Dijkstra /
delivery-probability optimization over the time-expanded contact graph at every
hop, so their per-decision cost grows with the number of contacts/nodes.

Measures mean wall-clock per route() call across topology scales.
Saves: results/inference_cost.json
"""
import io, sys, os, json, time, random
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, '.')

import numpy as np
from src.env.contact_plan import generate_rrna, generate_walker
from src.agents.ppo_agent import PPOAgent
from src.agents.rl_adapter import PPORoutingAdapter
from src.agents.classical_cgr import ClassicalCGR, RUCoPRouter
from src.evaluation.evaluator import Evaluator


def find_ckpt():
    f = Path('checkpoints/final/emrl_main_k16.pt')
    return str(f) if f.exists() else None


def bench(agent, plan, bundles, n_calls=400):
    """Mean seconds per route() decision over sampled (bundle, node, time) states."""
    rng = random.Random(0)
    samples = []
    for b in bundles[:n_calls]:
        # sample a plausible current node/time along an early route
        node = b.source
        t = b.creation_time + rng.uniform(0, 600)
        samples.append((b, node, t))
    t0 = time.perf_counter()
    for b, node, t in samples:
        agent.route(b, plan, node, t)
    return (time.perf_counter() - t0) / len(samples)


def main():
    ckpt = find_ckpt(); K = 16; OBS = 6 + K*8
    ppo = PPOAgent(obs_dim=OBS, action_dim=K+2); ppo.load(ckpt)

    topos = {
        "RRN-A (18)":  (generate_rrna(sim_duration=86400, seed=42), 18),
        "Walker-23":   (generate_walker(4, 5, 3, seed=42), 23),
        "Walker-52":   (generate_walker(6, 8, 4, seed=42), 52),
        "Walker-100":  (generate_walker(8, 12, 4, seed=42), 100),
    }
    ev = Evaluator(max_hops=32, bundle_ttl=7200.0, bundle_size_bits=10_000_000, verbose=False)

    print("=" * 76, flush=True)
    print("PER-DECISION INFERENCE COST (mean ms per route() call)", flush=True)
    print("=" * 76, flush=True)
    print(f"{'Topology':<14}{'#contacts':>10}{'EmRL':>11}{'ClassicalCGR':>14}{'RUCoP':>11}", flush=True)
    print("-" * 76, flush=True)

    out = {}
    for tname, (plan, nn) in topos.items():
        bundles = ev._generate_bundles(plan, 400, random.Random(1), np.random.default_rng(1))
        sl = PPORoutingAdapter(ppo, n_nodes=nn, sim_duration=86400.0, use_aacr=True, k_contacts=K)
        t_sl  = bench(sl, plan, bundles)
        t_cgr = bench(ClassicalCGR(), plan, bundles)
        t_ruc = bench(RUCoPRouter(), plan, bundles)
        out[tname] = {"n_contacts": len(plan.contacts),
                      "emrl_ms": round(t_sl*1e3, 3),
                      "classical_ms": round(t_cgr*1e3, 3),
                      "rucop_ms": round(t_ruc*1e3, 3),
                      "speedup_vs_classical": round(t_cgr/max(t_sl,1e-9), 1),
                      "speedup_vs_rucop": round(t_ruc/max(t_sl,1e-9), 1)}
        print(f"{tname:<14}{len(plan.contacts):>10}{t_sl*1e3:>10.3f}m{t_cgr*1e3:>13.3f}m{t_ruc*1e3:>10.3f}m", flush=True)

    print("\nSpeedup (EmRL vs baselines, larger = EmRL cheaper):", flush=True)
    for tname, d in out.items():
        print(f"  {tname:<14}: {d['speedup_vs_classical']}x vs ClassicalCGR, "
              f"{d['speedup_vs_rucop']}x vs RUCoP", flush=True)

    Path('results/inference_cost.json').write_text(json.dumps(out, indent=2))
    print("\nSaved: results/inference_cost.json", flush=True)


if __name__ == '__main__':
    main()

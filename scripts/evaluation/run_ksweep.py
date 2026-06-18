"""
run_ksweep.py — Scalability vs candidate-window size K (reviewer #4).
=====================================================================
The EmRL network is K-agnostic at inference (attention/pointer over any number
of contacts), so we evaluate the SAME trained model with candidate windows
K = 8, 16, 32, 64 on each topology and measure:
  • delivery ratio (BDR)
  • per-decision runtime (ms)
  • peak memory (MB)
to test whether the large-topology (Walker-100) gap is caused by the window
limitation — i.e. does widening K recover delivery at scale?

Saves: results/ksweep.json
"""
import io, sys, os, json, random, time, tracemalloc
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, '.')

import numpy as np
from src.env.contact_plan import generate_rrna, generate_walker
from src.agents.ppo_agent import PPOAgent
from src.agents.rl_adapter import PPORoutingAdapter
from src.agents.oracle import OracleCGR
from src.evaluation.evaluator import Evaluator, _RoutingEnv
from src.evaluation.metrics import compute_metrics


def find_ckpt():
    f = Path('checkpoints/final/emrl_main_k16.pt'); return str(f) if f.exists() else None


def evaluate(plan, agent, n_ep, seed):
    rng = random.Random(seed); np_rng = np.random.default_rng(seed)
    ev = Evaluator(max_hops=32, bundle_ttl=7200.0, bundle_size_bits=10_000_000, verbose=False)
    bundles = ev._generate_bundles(plan, n_ep, rng, np_rng)
    oracle = OracleCGR().compute_oracle_bdr(bundles, plan)
    env = _RoutingEnv(plan, max_hops=32, rng=random.Random(seed))
    t0 = time.perf_counter()
    eps = [ev.run_episode(agent, env, b) for b in bundles]
    dt = time.perf_counter() - t0
    m = compute_metrics(eps, oracle)
    n_decisions = max(sum(e.hops for e in eps), 1)
    return round(m.bdr, 4), round(oracle, 4), (dt / n_decisions) * 1e3  # ms/decision


def main():
    ckpt = find_ckpt()
    print("=" * 84, flush=True)
    print("SCALABILITY vs CANDIDATE-WINDOW K  (same K=16-trained model, zero-shot K)", flush=True)
    print(f"Checkpoint: {ckpt}", flush=True)
    print("=" * 84, flush=True)

    topos = {
        "RRN-A (18)":  (generate_rrna(sim_duration=86400, seed=42), 18, 200),
        "Walker-52":   (generate_walker(6, 8, 4, seed=42), 52, 150),
        "Walker-100":  (generate_walker(8, 12, 4, seed=42), 100, 100),
    }
    Ks = [8, 16, 32, 64]
    out = {}

    for tname, (plan, nn, neps) in topos.items():
        out[tname] = {}
        print(f"\n[{tname}]  {'K':>4}{'BDR':>9}{'ms/dec':>9}{'peakMB':>9}", flush=True)
        print("-" * 32, flush=True)
        for K in Ks:
            OBS = 6 + K * 9
            agent = PPOAgent(obs_dim=OBS, action_dim=K + 2); agent.load(ckpt)
            ad = PPORoutingAdapter(agent, n_nodes=nn, sim_duration=86400.0, use_aacr=True, k_contacts=K)
            tracemalloc.start()
            bdr, oracle, ms = evaluate(plan, ad, neps, seed=0)
            cur, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
            out[tname][f"K={K}"] = {"bdr": bdr, "oracle": oracle,
                                    "ms_per_decision": round(ms, 3), "peak_mb": round(peak/1e6, 1)}
            print(f"      {K:>4}{bdr:>9.3f}{ms:>9.2f}{peak/1e6:>9.1f}", flush=True)

    # Verdict: does widening K recover Walker-100 delivery?
    w = out.get("Walker-100", {})
    if w:
        b8, b64 = w.get("K=8", {}).get("bdr", 0), w.get("K=64", {}).get("bdr", 0)
        print("\n" + "=" * 84, flush=True)
        print(f"Walker-100: K=8 BDR={b8:.3f}  →  K=64 BDR={b64:.3f}  (Δ={b64-b8:+.3f})", flush=True)
        if b64 - b8 > 0.05:
            print("=> Widening K materially helps: the gap is partly a WINDOW limitation.", flush=True)
        else:
            print("=> Widening K does NOT recover delivery: the gap is NOT mainly the window", flush=True)
            print("   (points to policy capacity / training-scale, not candidate visibility).", flush=True)
    Path('results/ksweep.json').write_text(json.dumps(out, indent=2))
    print("\nSaved: results/ksweep.json", flush=True)


if __name__ == '__main__':
    main()

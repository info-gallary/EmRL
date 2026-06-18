"""
run_rl_necessity.py — Prove the RL component is necessary (component ablation).
==============================================================================
Separates the contribution of each part of EmRL by comparing a ladder of agents
that share the SAME contact-graph reachability encoder but differ in the decision
rule:

  1. Reachability-only GREEDY    — pick the contact with highest delivery-potential
  2. Reachability + HEURISTIC    — score = potential · reliability · rate / (wait+1)
  3. Reachability-gated PPO       (EmRL, AACR off)   [learned]
  4. Full EmRL                    (reachability-gated PPO + AACR)   [learned]

(The reachability-gated DQN rung requires retraining a DQN on the 9-feature
gated observation; see note at end — it is compute-bound and tracked separately.)

Metrics: delivery ratio, delay, energy — nominal AND under hub jamming — with
paired t-tests (Holm) + Cohen's d showing the learned policies beat the
non-learned reachability heuristics by a significant margin (=> RL is necessary).

Saves: results/rl_necessity.json
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
from src.env.contact_plan import generate_rrna, generate_walker
from src.agents.ppo_agent import PPOAgent
from src.agents.rl_adapter import PPORoutingAdapter
from src.agents.classical_cgr import ClassicalCGR
from src.agents.oracle import OracleCGR
from src.evaluation.evaluator import Evaluator, _RoutingEnv
from src.evaluation.metrics import compute_metrics
from src.analysis.stats import paired_t_test, holm_correction

K = 16


def _feasible(bundle, c, t):
    send = max(t, c.start_time)
    fin = send + bundle.size_bits / max(c.data_rate, 1)
    return fin <= c.end_time and fin < bundle.deadline


class GreedyReachability:
    """Non-learned: hop to the contact whose receiver has the highest
    delivery-potential toward the destination (pure reachability greedy)."""
    def route(self, bundle, plan, node, t):
        pot = plan.delivery_potential_to(bundle.destination)
        cands = [c for c in plan.get_contacts_from(node, t, window=7200.0) if c.sender == node]
        if not cands:
            return None
        return max(cands, key=lambda c: pot.get(c.receiver, 0.0))


class HeuristicReachability:
    """Non-learned: score = potential · reliability · data_rate / (wait+1).
    Reachability + hand-tuned reliability/timing heuristic (no learning)."""
    def route(self, bundle, plan, node, t):
        pot = plan.delivery_potential_to(bundle.destination)
        cands = [c for c in plan.get_contacts_from(node, t, window=7200.0) if c.sender == node]
        if not cands:
            return None
        def score(c):
            wait = max(c.start_time - t, 0.0)
            return pot.get(c.receiver, 0.0) * c.reliability * (c.data_rate / 2e6) / (wait / 600.0 + 1.0)
        return max(cands, key=score)


def jam_top_hubs(plan, n_hubs, seed):
    rng = random.Random(seed); np_rng = np.random.default_rng(seed)
    ev = Evaluator(max_hops=32, bundle_ttl=7200.0, bundle_size_bits=10_000_000, verbose=False)
    bundles = ev._generate_bundles(plan, 150, rng, np_rng)
    cgr = ClassicalCGR(); freq = defaultdict(int); gs = plan.n_nodes - 4
    for b in bundles:
        node, t = b.source, b.creation_time
        for _ in range(32):
            if node == b.destination: break
            c = cgr.route(b, plan, node, t)
            if c is None: break
            if c.receiver < gs and c.receiver not in (b.source, b.destination): freq[c.receiver] += 1
            t = max(t, c.start_time) + b.size_bits / max(c.data_rate, 1); node = c.receiver
    hubs = set([k for k, _ in sorted(freq.items(), key=lambda x: -x[1])][:n_hubs])
    p = deepcopy(plan); jr = random.Random(seed + 1)
    for c in p.contacts:
        if c.sender in hubs or c.receiver in hubs: c.reliability = jr.uniform(0.2, 0.4)
    p._potential_to = {}; p._dist_to = {}
    return p


def evaluate(plan, agents, n_ep, seed):
    rng = random.Random(seed); np_rng = np.random.default_rng(seed)
    ev = Evaluator(max_hops=32, bundle_ttl=7200.0, bundle_size_bits=10_000_000, verbose=False)
    bundles = ev._generate_bundles(plan, n_ep, rng, np_rng)
    oracle = OracleCGR().compute_oracle_bdr(bundles, plan)
    env = _RoutingEnv(plan, max_hops=32, rng=random.Random(seed))
    out = {}; outc = {}
    for name, ag in agents.items():
        eps = [ev.run_episode(ag, env, b) for b in bundles]
        m = compute_metrics(eps, oracle)
        out[name] = {"bdr": round(m.bdr, 4), "delay": round(m.mean_delay, 1),
                     "energy": round(m.mean_energy, 3)}
        outc[name] = np.array([1.0 if e.delivered else 0.0 for e in eps])
    out["Oracle"] = {"bdr": round(oracle, 4)}
    return out, outc


def find_ckpt():
    f = Path('checkpoints/final/emrl_main_k16.pt'); return str(f) if f.exists() else None


def main():
    ckpt = find_ckpt(); OBS = 6 + K*9
    print("=" * 90, flush=True)
    print("RL-NECESSITY ABLATION — shared reachability encoder, different decision rules", flush=True)
    print(f"Checkpoint: {ckpt}", flush=True)
    print("=" * 90, flush=True)
    ppo = PPOAgent(obs_dim=OBS, action_dim=K+2); ppo.load(ckpt)
    rrna = generate_rrna(sim_duration=86400, seed=42)
    NN = rrna.n_nodes

    agents = {
        "1. Reach-Greedy":   GreedyReachability(),
        "2. Reach-Heuristic":HeuristicReachability(),
        "3. EmRL-PPO (noAACR)": PPORoutingAdapter(ppo, n_nodes=NN, sim_duration=86400.0, use_aacr=False, k_contacts=K),
        "4. EmRL-Full":      PPORoutingAdapter(ppo, n_nodes=NN, sim_duration=86400.0, use_aacr=True, k_contacts=K),
    }

    results = {}
    for cond, plan in [("nominal", rrna), ("jammed", jam_top_hubs(rrna, 2, 7))]:
        bdr, outc = evaluate(plan, agents, 300, seed=0 if cond == "nominal" else 1)
        results[cond] = bdr
        print(f"\n[{cond.upper()}]  {'Agent':<24}{'BDR':>8}{'Delay(s)':>10}{'Energy':>9}", flush=True)
        print("-" * 54, flush=True)
        for n in agents:
            d = bdr[n]; print(f"  {'':2}{n:<22}{d['bdr']:>8.3f}{d['delay']:>10.0f}{d['energy']:>9.3f}", flush=True)
        print(f"  {'':2}{'Oracle':<22}{bdr['Oracle']['bdr']:>8.3f}", flush=True)
        # stats: learned (EmRL-Full) vs each non-learned
        prim = "4. EmRL-Full"
        res = [paired_t_test(outc[prim], outc[b], prim, b) for b in
               ("1. Reach-Greedy", "2. Reach-Heuristic", "3. EmRL-PPO (noAACR)")]
        holm_correction(res)
        results[cond + "_stats"] = {}
        print(f"  --- significance (EmRL-Full vs) ---", flush=True)
        for r in res:
            print(f"    vs {r.agent_b:<22} Δ={r.mean_diff:+.3f}  d={r.cohens_d:+.2f}  p={r.p_value:.1e}  "
                  f"{'sig' if r.significant_holm else 'ns'}", flush=True)
            results[cond + "_stats"][r.agent_b] = {
                "delta": round(r.mean_diff, 4), "cohens_d": round(r.cohens_d, 3),
                "p_value": r.p_value, "significant_holm": bool(r.significant_holm)}

    print("\n" + "=" * 90, flush=True)
    print("VERDICT: if the learned policies significantly beat Reach-Greedy/Heuristic,", flush=True)
    print("the RL component is necessary (the reachability encoder alone is not sufficient).", flush=True)
    print("NOTE: rung 3 (reachability-gated DQN) needs retraining on the 9-feature gated obs", flush=True)
    print("(compute-bound; tracked separately).", flush=True)
    Path('results/rl_necessity.json').write_text(json.dumps(results, indent=2, default=str))
    print("Saved: results/rl_necessity.json", flush=True)


if __name__ == '__main__':
    main()

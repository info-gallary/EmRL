"""
run_rucop_benchmark.py — Reviewer #2: comparison vs GROGU via the RUCoP anchor.
==============================================================================
No public GROGU implementation exists, and GROGU + EmRL were evaluated on
different topologies — so a raw cross-paper BDR number is methodologically
unsound. Instead we use the **shared RUCoP benchmark** (Raverta et al. 2021,
"Routing Under Uncertain Contact Plans") as a calibration anchor: GROGU reports
its delivery probability *relative to RUCoP* on this benchmark, and so do we.

This script:
  1. Builds RUCoP-style **uncertain contact plans** — random networks whose
     contacts carry explicit failure probabilities — swept over an uncertainty
     level (mean per-contact reliability 0.5 -> 0.9).
  2. Measures **Successful Delivery Probability (SDP)** — RUCoP's metric — via
     Monte-Carlo over stochastic link realisations (the simulator drops a link
     with prob 1-reliability, matching training semantics).
  3. Reports SDP for our RUCoP reimplementation (the anchor), Classical-CGR,
     EmRL-PPO (zero-shot), Reach-Greedy, and Spray-L2 — plus a GROGU row whose
     published SDP-vs-RUCoP figures are inserted from the paper (we could not
     fetch them automatically; the HAL page blocks programmatic access).

Saves: results/rucop_benchmark.json
"""
import io, sys, os, json, random
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, '.')

import numpy as np
from src.env.contact_plan import Contact, ContactPlan
from src.agents.ppo_agent import PPOAgent
from src.agents.rl_adapter import PPORoutingAdapter
from src.agents.classical_cgr import ClassicalCGR, RUCoPRouter, SprayAndWait
from src.agents.oracle import OracleCGR
from src.evaluation.evaluator import Evaluator, _RoutingEnv
from src.evaluation.metrics import compute_metrics

K = 16
SIM_DUR = 3600.0          # 1 h
TTL = 1800.0              # 30 min
N_FLOWS = 300             # Monte-Carlo flows per (level, seed)
N_SEEDS = 4
REL_LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9]   # mean per-contact reliability sweep


class GreedyReachability:
    def route(self, bundle, plan, node, t):
        pot = plan.delivery_potential_to(bundle.destination)
        cands = [c for c in plan.get_contacts_from(node, t, window=TTL) if c.sender == node]
        if not cands:
            return None
        return max(cands, key=lambda c: pot.get(c.receiver, 0.0))


def make_uncertain_plan(n_nodes, degree, rel_mean, rel_spread, seed,
                        sim_duration=SIM_DUR, window=180.0, period=420.0):
    """RUCoP-style uncertain contact plan: random k-regular-ish graph, periodic
    contact windows, per-contact reliability ~ N(rel_mean, rel_spread)."""
    rng = random.Random(seed)
    edges = set()
    for i in range(n_nodes):
        peers = rng.sample([j for j in range(n_nodes) if j != i],
                           min(degree, n_nodes - 1))
        for j in peers:
            edges.add((min(i, j), max(i, j)))
    contacts = []
    cid = 0
    for (i, j) in edges:
        t = rng.uniform(0, period)
        while t < sim_duration:
            dur = window * rng.uniform(0.7, 1.3)
            dr = rng.uniform(0.5e6, 1.5e6)
            ec = rng.uniform(0.5, 2.0)
            rel = min(0.99, max(0.05, rng.gauss(rel_mean, rel_spread)))
            t1 = min(t + dur, sim_duration)
            # both directions (symmetric link), independent reliability draw
            contacts.append(Contact(cid, i, j, t, t1, dr, ec, rel)); cid += 1
            rel2 = min(0.99, max(0.05, rng.gauss(rel_mean, rel_spread)))
            contacts.append(Contact(cid, j, i, t, t1, dr, ec, rel2)); cid += 1
            t += period * rng.uniform(0.8, 1.2)
    return ContactPlan(contacts, n_nodes, sim_duration)


def sdp_for(plan, agent, n_flows, seed):
    """Monte-Carlo Successful Delivery Probability over random flows."""
    rng = random.Random(seed); np_rng = np.random.default_rng(seed)
    ev = Evaluator(max_hops=32, bundle_ttl=TTL, bundle_size_bits=1_000_000, verbose=False)
    bundles = ev._generate_bundles(plan, n_flows, rng, np_rng)
    env = _RoutingEnv(plan, max_hops=32, rng=random.Random(seed + 7))
    eps = [ev.run_episode(agent, env, b) for b in bundles]
    return float(np.mean([1.0 if e.delivered else 0.0 for e in eps]))


def main():
    ckpt = str(Path('checkpoints/final/emrl_main_k16.pt'))
    OBS = 6 + K * 9
    print("=" * 78, flush=True)
    print("RUCoP-BENCHMARK (uncertain contact plans) — SDP vs mean reliability", flush=True)
    print(f"Metric: Successful Delivery Probability (Monte-Carlo, {N_FLOWS} flows × "
          f"{N_SEEDS} seeds)", flush=True)
    print("=" * 78, flush=True)
    ppo = PPOAgent(obs_dim=OBS, action_dim=K + 2); ppo.load(ckpt)
    N_NODES, DEGREE = 14, 4

    # one representative plan to size the EmRL adapter
    probe = make_uncertain_plan(N_NODES, DEGREE, 0.8, 0.08, 0)
    emrl = PPORoutingAdapter(ppo, n_nodes=probe.n_nodes, sim_duration=SIM_DUR,
                             use_aacr=True, k_contacts=K)
    agents = {
        "RUCoP (anchor)": RUCoPRouter(),
        "CGR":            ClassicalCGR(),
        "EmRL-PPO":       emrl,
        "Reach-Greedy":   GreedyReachability(),
        "Spray-L2":       SprayAndWait(L=2),
    }

    results = {name: {} for name in agents}
    header = f"{'mean_rel':>9}" + "".join(f"{n[:13]:>15}" for n in agents)
    print("\n" + header, flush=True)
    print("-" * len(header), flush=True)
    for lvl in REL_LEVELS:
        row = {name: [] for name in agents}
        for s in range(N_SEEDS):
            plan = make_uncertain_plan(N_NODES, DEGREE, lvl, 0.08, seed=s)
            for name, ag in agents.items():
                row[name].append(sdp_for(plan, ag, N_FLOWS, seed=s))
        line = f"{lvl:>9.2f}"
        for name in agents:
            m = float(np.mean(row[name])); sd = float(np.std(row[name]))
            results[name][f"{lvl:.2f}"] = {"sdp": round(m, 4), "std": round(sd, 4)}
            line += f"{m:>11.3f}±{sd:>3.2f}"
        print(line, flush=True)

    # --- RUCoP-anchored relative position (what GROGU also reports) ---
    print("\n--- SDP relative to RUCoP anchor (mean over reliability sweep) ---", flush=True)
    anchor = np.mean([results["RUCoP (anchor)"][f"{l:.2f}"]["sdp"] for l in REL_LEVELS])
    rel_pos = {}
    for name in agents:
        m = np.mean([results[name][f"{l:.2f}"]["sdp"] for l in REL_LEVELS])
        ratio = float(m / anchor) if anchor > 0 else float('nan')
        rel_pos[name] = round(ratio, 3)
        print(f"  {name:<16} SDP/RUCoP = {ratio:.3f}", flush=True)

    # GROGU row: published figures to be inserted from the paper (HAL blocked auto-fetch)
    grogu_placeholder = {
        "note": ("GROGU reports SDP relative to RUCoP on the shared RUCoP "
                 "benchmark. Insert GROGU's published SDP/RUCoP ratio here from "
                 "Tab. of hal-05413586 to complete the indirect comparison via "
                 "the shared anchor."),
        "sdp_over_rucop": None,
    }

    out = {
        "meta": {"metric": "Successful Delivery Probability (Monte-Carlo)",
                 "n_nodes": N_NODES, "degree": DEGREE, "n_flows": N_FLOWS,
                 "n_seeds": N_SEEDS, "rel_levels": REL_LEVELS, "K": K,
                 "anchor": "RUCoP (our reimplementation of Raverta et al. 2021)"},
        "sdp_by_level": results,
        "sdp_over_rucop": rel_pos,
        "grogu": grogu_placeholder,
    }
    Path('results/rucop_benchmark.json').write_text(json.dumps(out, indent=2, default=str))
    print("\nSaved: results/rucop_benchmark.json", flush=True)


if __name__ == '__main__':
    main()

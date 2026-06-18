"""
run_tle_anomaly.py — Uncertain REALISTIC benchmark: real TLE geometry + anomalies.
==================================================================================
The synthetic threat-model study (§6.9) injects anomalies into a synthetic RRN-A
plan; the real-TLE study (§6.13) uses real geometry but only nominal reliability.
This script combines BOTH: anomalies are injected on contact plans built from real
NORAD TLEs (SGP4) — the deployment-realistic case where uncertainty/anomalies hit
real orbital geometry. Oracle-normalised BDR, directly comparable to both tables.

Scenarios (per constellation): nominal, random, correlated, cascading,
space-weather, adversarial. Ladder: Reach-Greedy / Reach-Heuristic (analytic
encoder, no learning), EmRL-PPO (synthetic-trained), EmRL-TLE-FT (fine-tuned on
real geometry, if checkpoint present), CGR / RUCoP / Spray-L2.

Saves: results/tle_anomaly.json
"""
import io, sys, os, json, random
from pathlib import Path
from copy import deepcopy
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, '.')

import numpy as np
from src.env.tle_contact_plan import generate_from_tle
from src.agents.ppo_agent import PPOAgent
from src.agents.rl_adapter import PPORoutingAdapter
from src.agents.classical_cgr import ClassicalCGR, RUCoPRouter, SprayAndWait
from src.agents.oracle import OracleCGR
from src.evaluation.evaluator import Evaluator, _RoutingEnv
from src.evaluation.metrics import compute_metrics

K = 16
OBS = 6 + K * 9
N_SEEDS = 3
N_EP = 100
SIM_H = 24.0
SIM_DUR = SIM_H * 3600.0
CONSTELLATIONS = [("iridium-NEXT", 24), ("orbcomm", 20), ("globalstar", 20)]
CKPT_SYN = 'checkpoints/final/emrl_main_k16.pt'
CKPT_FT = 'checkpoints/final/emrl_tle_ft_k16.pt'   # used iff present


# --- analytic encoder-only routers (share the reachability field) ---
class GreedyReachability:
    def route(self, bundle, plan, node, t):
        pot = plan.delivery_potential_to(bundle.destination)
        cands = [c for c in plan.get_contacts_from(node, t, window=7200.0) if c.sender == node]
        if not cands:
            return None
        return max(cands, key=lambda c: pot.get(c.receiver, 0.0))


class HeuristicReachability:
    def route(self, bundle, plan, node, t):
        pot = plan.delivery_potential_to(bundle.destination)
        cands = [c for c in plan.get_contacts_from(node, t, window=7200.0) if c.sender == node]
        if not cands:
            return None
        def score(c):
            wait = max(c.start_time - t, 0.0)
            return pot.get(c.receiver, 0.0) * c.reliability * (c.data_rate / 2e6) / (wait / 600.0 + 1.0)
        return max(cands, key=score)


# --- anomaly injectors (real geometry preserved; only reliabilities degraded) ---
def _clear_cache(p):
    if hasattr(p, '_potential_to'): p._potential_to = {}
    if hasattr(p, '_dist_to'): p._dist_to = {}
    return p


def _perturb(plan, fn, seed):
    p = deepcopy(plan); rng = random.Random(seed)
    for c in p.contacts:
        fn(c, rng)
    return _clear_cache(p)


def sc_nominal(base, seed):
    return base


def sc_random(base, seed, p_fail=0.25):
    def _f(c, r):
        if r.random() < p_fail:
            c.reliability = r.uniform(0.05, 0.25)
    return _perturb(base, _f, seed)


def sc_correlated(base, seed, n_sat_cluster=3):
    p = deepcopy(base); rng = random.Random(seed)
    sats = [n for n in range(base.n_nodes) if n < base.n_nodes - 4]  # exclude GS
    cluster = set(rng.sample(sats, min(n_sat_cluster, len(sats))))
    for c in p.contacts:
        if c.sender in cluster or c.receiver in cluster:
            c.reliability = rng.uniform(0.1, 0.3)
    return _clear_cache(p)


def sc_cascading(base, seed, p_seed=0.1, p_spread=0.5):
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
    return _clear_cache(p)


def sc_space_weather(base, seed, degrade=0.45):
    def _f(c, r):
        c.reliability *= (1.0 - degrade * r.uniform(0.7, 1.0))
    return _perturb(base, _f, seed)


def sc_adversarial(base, seed):
    """Degrade the highest delivery-potential receivers (targets the encoder's signal)."""
    p = deepcopy(base); rng = random.Random(seed)
    pot = base.delivery_potential_to(0)
    top = sorted(pot, key=lambda n: -pot[n])[:4]
    for c in p.contacts:
        if c.receiver in top:
            c.reliability = rng.uniform(0.05, 0.2)
    return _clear_cache(p)


SCENARIOS = {
    "nominal":       sc_nominal,
    "random":        sc_random,
    "correlated":    sc_correlated,
    "cascading":     sc_cascading,
    "space_weather": sc_space_weather,
    "adversarial":   sc_adversarial,
}
# fixed per-scenario perturbation seeds (reproducible; PYTHONHASHSEED-independent)
SCENARIO_SEED = {n: 20 + i for i, n in enumerate(SCENARIOS)}


def bootstrap_ci(arr, n_boot=2000, alpha=0.05, rng=None):
    rng = rng or np.random.default_rng(0)
    a = np.asarray(arr, dtype=float)
    if len(a) < 2:
        return [float(a.mean()), float(a.mean())]
    boots = [np.mean(rng.choice(a, len(a), replace=True)) for _ in range(n_boot)]
    return list(np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def build_agents(ppo_syn, ppo_ft, NN):
    ag = {
        "Reach-Greedy":    GreedyReachability(),
        "Reach-Heuristic": HeuristicReachability(),
        "EmRL-PPO":        PPORoutingAdapter(ppo_syn, n_nodes=NN, sim_duration=SIM_DUR,
                                             use_aacr=True, k_contacts=K),
    }
    if ppo_ft is not None:
        ag["EmRL-TLE-FT"] = PPORoutingAdapter(ppo_ft, n_nodes=NN, sim_duration=SIM_DUR,
                                              use_aacr=True, k_contacts=K)
    ag.update({"CGR": ClassicalCGR(), "RUCoP": RUCoPRouter(), "Spray-L2": SprayAndWait(L=2)})
    return ag


def eval_plan(plan, ppo_syn, ppo_ft):
    """Oracle-normalised BDR for every agent, averaged over N_SEEDS×N_EP."""
    NN = plan.n_nodes
    agents = build_agents(ppo_syn, ppo_ft, NN)
    seed_bdrs = {n: [] for n in agents}
    for s in range(N_SEEDS):
        r = random.Random(s); np_r = np.random.default_rng(s)
        ev = Evaluator(max_hops=32, bundle_ttl=7200.0, bundle_size_bits=10_000_000, verbose=False)
        bundles = ev._generate_bundles(plan, N_EP, r, np_r)
        oracle = OracleCGR().compute_oracle_bdr(bundles, plan)
        env = _RoutingEnv(plan, max_hops=32, rng=random.Random(s))
        for name, ag in agents.items():
            eps = [ev.run_episode(ag, env, b) for b in bundles]
            seed_bdrs[name].append(compute_metrics(eps, oracle).bdr)
    rng_b = np.random.default_rng(123)
    summary = {}
    for name, vals in seed_bdrs.items():
        a = np.array(vals); ci = bootstrap_ci(a, rng=rng_b)
        summary[name] = {"mean": round(float(a.mean()), 4),
                         "std": round(float(a.std(ddof=1)), 4),
                         "ci95": [round(float(ci[0]), 4), round(float(ci[1]), 4)]}
    return summary


def main():
    print("=" * 80, flush=True)
    print(f"UNCERTAIN REALISTIC BENCHMARK — real TLE geometry + anomalies "
          f"({SIM_H:.0f}h, {N_SEEDS}×{N_EP}ep, K={K})", flush=True)
    print("=" * 80, flush=True)

    ppo_syn = PPOAgent(obs_dim=OBS, action_dim=K + 2); ppo_syn.load(CKPT_SYN)
    ppo_ft = None
    if Path(CKPT_FT).exists():
        ppo_ft = PPOAgent(obs_dim=OBS, action_dim=K + 2); ppo_ft.load(CKPT_FT)
        print(f"Fine-tuned checkpoint found: {CKPT_FT} (EmRL-TLE-FT included)", flush=True)
    else:
        print(f"No fine-tuned checkpoint yet ({CKPT_FT}); EmRL-TLE-FT omitted.", flush=True)

    out = {"meta": {"sim_hours": SIM_H, "n_seeds": N_SEEDS, "n_ep": N_EP, "K": K,
                    "scenarios": list(SCENARIOS), "ft_included": ppo_ft is not None,
                    "data_source": "CelesTrak TLE + SGP4; anomalies injected on real geometry"},
           "constellations": {}}

    for group, n_sats in CONSTELLATIONS:
        try:
            base = generate_from_tle(group=group, n_sats=n_sats, sim_duration=SIM_DUR,
                                     step_s=60.0, seed=0, verbose=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {group}: {exc}", flush=True)
            continue
        print(f"\n### {group} ({base.n_nodes} nodes, {len(base.contacts)} contacts) ###", flush=True)
        cres = {}
        for sname, fn in SCENARIOS.items():
            plan = fn(base, seed=SCENARIO_SEED[sname])
            summ = eval_plan(plan, ppo_syn, ppo_ft)
            cres[sname] = summ
            row = "  ".join(f"{n}={d['mean']:.3f}" for n, d in summ.items())
            print(f"  [{sname:<13}] {row}", flush=True)
        out["constellations"][group] = cres

    # --- robustness verdict: encoder vs learned policy under anomalies (Iridium) ---
    prim = out["constellations"].get("iridium-NEXT")
    if prim:
        print("\n" + "=" * 80, flush=True)
        print("ROBUSTNESS UNDER ANOMALIES (Iridium-NEXT):", flush=True)
        verdict = {}
        for sname, summ in prim.items():
            rg = summ.get("Reach-Greedy", {}).get("mean", 0.0)
            cgr = summ.get("CGR", {}).get("mean", 0.0)
            em = summ.get("EmRL-PPO", {}).get("mean", 0.0)
            ft = summ.get("EmRL-TLE-FT", {}).get("mean")
            line = f"  {sname:<14} Reach-Greedy={rg:.3f}  CGR={cgr:.3f}  EmRL-PPO={em:.3f}"
            if ft is not None:
                line += f"  EmRL-TLE-FT={ft:.3f}"
            print(line, flush=True)
            verdict[sname] = {"reach_greedy": rg, "cgr": cgr, "emrl_ppo": em, "emrl_ft": ft}
        out["meta"]["verdict_iridium"] = verdict

    Path('results/tle_anomaly.json').write_text(json.dumps(out, indent=2, default=str))
    print("\nSaved: results/tle_anomaly.json", flush=True)


if __name__ == '__main__':
    main()

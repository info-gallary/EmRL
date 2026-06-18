"""
run_tle_eval.py — Reviewer #3: real contact traces (TLE-derived geometry).
==========================================================================
Evaluates the full ablation ladder on contact plans derived from **real public
TLEs** (CelesTrak) via SGP4 propagation — not synthetic Walker plans. Same
oracle-normalised BDR pipeline as the synthetic tables, so numbers are directly
comparable.

Ladder (shared reachability encoder, different decision rules):
  • Reach-Greedy      — analytic: hop to highest delivery-potential receiver (no learning)
  • Reach-Heuristic   — analytic: potential·reliability·rate/(wait+1) (hand-tuned, no learning)
  • EmRL-PPO          — learned policy (zero-shot; trained on synthetic Walker/RRN-A)
  • CGR / RUCoP       — analytic baselines that re-plan per topology
  • Spray-L2          — flooding baseline

Headline question: which components transfer ZERO-SHOT to real orbital geometry?

Saves: results/tle_eval.json
"""
import io, sys, os, json, random
from pathlib import Path
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
N_SEEDS = 5
N_EP = 150
SIM_H = 24.0                       # match the training-plan horizon
SIM_DUR = SIM_H * 3600.0
N_SATS = 24
# (group, n_sats) — primary + robustness constellations; skipped gracefully if fetch fails.
CONSTELLATIONS = [("iridium-NEXT", 24), ("orbcomm", 20), ("globalstar", 20)]


# --- non-learned encoder-only routers (analytic; share the reachability field) ---
class GreedyReachability:
    """Hop to the contact whose receiver has the highest delivery-potential."""
    def route(self, bundle, plan, node, t):
        pot = plan.delivery_potential_to(bundle.destination)
        cands = [c for c in plan.get_contacts_from(node, t, window=7200.0) if c.sender == node]
        if not cands:
            return None
        return max(cands, key=lambda c: pot.get(c.receiver, 0.0))


class HeuristicReachability:
    """score = potential · reliability · rate / (wait+1) (hand-tuned, no learning)."""
    def route(self, bundle, plan, node, t):
        pot = plan.delivery_potential_to(bundle.destination)
        cands = [c for c in plan.get_contacts_from(node, t, window=7200.0) if c.sender == node]
        if not cands:
            return None
        def score(c):
            wait = max(c.start_time - t, 0.0)
            return pot.get(c.receiver, 0.0) * c.reliability * (c.data_rate / 2e6) / (wait / 600.0 + 1.0)
        return max(cands, key=score)


def bootstrap_ci(arr, n_boot=2000, alpha=0.05, rng=None):
    rng = rng or np.random.default_rng(0)
    a = np.asarray(arr, dtype=float)
    if len(a) < 2:
        return [float(a.mean()), float(a.mean())]
    boots = [np.mean(rng.choice(a, len(a), replace=True)) for _ in range(n_boot)]
    return list(np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def build_agents(ppo, ppo_ft, NN):
    ag = {
        "Reach-Greedy":    GreedyReachability(),
        "Reach-Heuristic": HeuristicReachability(),
        "EmRL-PPO":        PPORoutingAdapter(ppo, n_nodes=NN, sim_duration=SIM_DUR,
                                             use_aacr=True, k_contacts=K),
    }
    if ppo_ft is not None:
        ag["EmRL-TLE-FT"] = PPORoutingAdapter(ppo_ft, n_nodes=NN, sim_duration=SIM_DUR,
                                              use_aacr=True, k_contacts=K)
    ag.update({"CGR": ClassicalCGR(), "RUCoP": RUCoPRouter(), "Spray-L2": SprayAndWait(L=2)})
    return ag


def eval_constellation(group, n_sats, ppo, ppo_ft, rng_b):
    try:
        plan = generate_from_tle(group=group, n_sats=n_sats, sim_duration=SIM_DUR,
                                 step_s=60.0, seed=0, verbose=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  [skip] {group}: {exc}", flush=True)
        return None
    NN = plan.n_nodes
    agents = build_agents(ppo, ppo_ft, NN)
    seed_bdrs = {n: [] for n in agents}
    for s in range(N_SEEDS):
        r = random.Random(s); np_r = np.random.default_rng(s)
        ev = Evaluator(max_hops=32, bundle_ttl=7200.0,
                       bundle_size_bits=10_000_000, verbose=False)
        bundles = ev._generate_bundles(plan, N_EP, r, np_r)
        oracle = OracleCGR().compute_oracle_bdr(bundles, plan)
        env = _RoutingEnv(plan, max_hops=32, rng=random.Random(s))
        for name, ag in agents.items():
            eps = [ev.run_episode(ag, env, b) for b in bundles]
            seed_bdrs[name].append(compute_metrics(eps, oracle).bdr)
    summary = {}
    print(f"\n  {'Agent':<18}{'Mean':>8}{'Std':>8}{'CI95_lo':>10}{'CI95_hi':>10}", flush=True)
    print("  " + "-" * 52, flush=True)
    for name, vals in seed_bdrs.items():
        a = np.array(vals); ci = bootstrap_ci(a, rng=rng_b)
        summary[name] = {"mean": round(float(a.mean()), 4),
                         "std": round(float(a.std(ddof=1)), 4),
                         "ci95": [round(float(ci[0]), 4), round(float(ci[1]), 4)],
                         "seeds": [round(float(v), 4) for v in vals]}
        print(f"  {name:<18}{a.mean():>8.3f}{a.std(ddof=1):>8.3f}{ci[0]:>10.3f}{ci[1]:>10.3f}", flush=True)
    return {"n_nodes": NN, "n_sats": n_sats, "n_contacts": len(plan.contacts),
            "summary": summary}


def main():
    ckpt = str(Path('checkpoints/final/emrl_main_k16.pt'))
    ckpt_ft = Path('checkpoints/final/emrl_tle_ft_k16.pt')
    OBS = 6 + K * 9
    print("=" * 72, flush=True)
    print(f"REAL-TLE EVALUATION  ({SIM_H:.0f}h, {N_SEEDS} seeds × {N_EP} ep, K={K})", flush=True)
    print("Source: CelesTrak public TLE + SGP4 propagation (TEME->ECEF)", flush=True)
    print("=" * 72, flush=True)
    ppo = PPOAgent(obs_dim=OBS, action_dim=K + 2); ppo.load(ckpt)
    ppo_ft = None
    if ckpt_ft.exists():
        ppo_ft = PPOAgent(obs_dim=OBS, action_dim=K + 2); ppo_ft.load(str(ckpt_ft))
        print(f"Fine-tuned checkpoint found: {ckpt_ft} (EmRL-TLE-FT added to ladder)", flush=True)
    else:
        print(f"No fine-tuned checkpoint ({ckpt_ft}); EmRL-TLE-FT omitted.", flush=True)
    rng_b = np.random.default_rng(123)

    out = {"meta": {"sim_hours": SIM_H, "n_seeds": N_SEEDS, "n_ep_per_seed": N_EP,
                    "K": K, "data_source": "CelesTrak TLE + SGP4 (TEME->ECEF)"},
           "constellations": {}}
    for group, n_sats in CONSTELLATIONS:
        print(f"\n### {group} ({n_sats} sats) ###", flush=True)
        res = eval_constellation(group, n_sats, ppo, ppo_ft, rng_b)
        if res is not None:
            out["constellations"][group] = res

    # --- transfer verdict (primary constellation) ---
    prim = out["constellations"].get("iridium-NEXT")
    if prim:
        s = prim["summary"]
        rg = s["Reach-Greedy"]["mean"]; cgr = s["CGR"]["mean"]
        em = s["EmRL-PPO"]["mean"]; ru = s["RUCoP"]["mean"]
        ft = s.get("EmRL-TLE-FT", {}).get("mean")
        print("\n" + "=" * 72, flush=True)
        print("TRANSFER VERDICT (Iridium-NEXT, zero-shot to real geometry):", flush=True)
        print(f"  Analytic encoder (Reach-Greedy) = {rg:.3f}  vs  CGR = {cgr:.3f}"
              f"  -> encoder {'TRANSFERS' if rg >= cgr - 0.05 else 'underperforms'}", flush=True)
        print(f"  Learned policy   (EmRL-PPO)     = {em:.3f}  ->  "
              f"{'distribution-shift sensitive' if em < rg - 0.05 else 'transfers'}", flush=True)
        if ft is not None:
            print(f"  Fine-tuned       (EmRL-TLE-FT)  = {ft:.3f}  ->  "
                  f"{'recovers (>=CGR)' if ft >= cgr - 0.05 else 'partial recovery'}"
                  f"  (zero-shot {em:.3f} -> FT {ft:.3f}, Δ={ft-em:+.3f})", flush=True)
        print(f"  Ceiling          (RUCoP)        = {ru:.3f}", flush=True)
        out["meta"]["verdict"] = {
            "encoder_transfers": bool(rg >= cgr - 0.05),
            "learned_policy_degrades": bool(em < rg - 0.05),
            "reach_greedy": rg, "cgr": cgr, "emrl_ppo": em, "rucop": ru,
            "emrl_tle_ft": ft,
            "ft_recovers": (bool(ft >= cgr - 0.05) if ft is not None else None),
            "ft_gain_over_zeroshot": (round(ft - em, 4) if ft is not None else None)}

    Path('results/tle_eval.json').write_text(json.dumps(out, indent=2, default=str))
    print("\nSaved: results/tle_eval.json", flush=True)


if __name__ == '__main__':
    main()

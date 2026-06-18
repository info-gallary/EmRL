"""
run_tle_holdout.py — Leave-one-constellation-out GENERALIZATION test.
=====================================================================
Answers the "you tested on your training set" critique. EmRL-TLE-FT (the full
fine-tune) trained on Iridium/Orbcomm/Globalstar/OneWeb, so evaluating it on
Iridium is in-distribution. Here we evaluate a SEPARATE model that was fine-tuned
with Iridium HELD OUT (trained only on Orbcomm/Globalstar/OneWeb + synthetic) on
Iridium — true zero-shot transfer to an unseen real constellation.

Three learned models on the held-out constellation (default Iridium-NEXT):
  • EmRL-PPO       — synthetic-only (never saw real geometry)
  • EmRL-TLE-FT    — full fine-tune (SAW Iridium)            [in-distribution]
  • EmRL-Holdout   — fine-tune with Iridium EXCLUDED (unseen) [generalization]
plus analytic refs: Reach-Greedy, CGR, RUCoP.

Generalization gap = EmRL-TLE-FT(Iridium) − EmRL-Holdout(Iridium). Small gap →
the fine-tuning gain generalizes across constellations rather than memorizing.

Saves: results/tle_holdout.json
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
from src.agents.classical_cgr import ClassicalCGR, RUCoPRouter
from src.agents.oracle import OracleCGR
from src.evaluation.evaluator import Evaluator, _RoutingEnv
from src.evaluation.metrics import compute_metrics

K = 16
OBS = 6 + K * 9
N_SEEDS = 5
N_EP = 150
SIM_DUR = 24.0 * 3600.0
HOLDOUT = os.environ.get('TLE_HOLDOUT', 'iridium-NEXT')
HOLDOUT_NSAT = int(os.environ.get('TLE_HOLDOUT_NSAT', 24))

CKPT_SYN = 'checkpoints/final/emrl_main_k16.pt'
CKPT_FULL = 'checkpoints/final/emrl_tle_ft_k16.pt'
CKPT_HOLD = f'checkpoints/final/emrl_tle_ft_holdout_{HOLDOUT}_k16.pt'


class GreedyReachability:
    def route(self, bundle, plan, node, t):
        pot = plan.delivery_potential_to(bundle.destination)
        cands = [c for c in plan.get_contacts_from(node, t, window=7200.0) if c.sender == node]
        return max(cands, key=lambda c: pot.get(c.receiver, 0.0)) if cands else None


def bootstrap_ci(arr, n_boot=2000, alpha=0.05, rng=None):
    rng = rng or np.random.default_rng(0)
    a = np.asarray(arr, dtype=float)
    if len(a) < 2:
        return [float(a.mean()), float(a.mean())]
    boots = [np.mean(rng.choice(a, len(a), replace=True)) for _ in range(n_boot)]
    return list(np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def load(ckpt):
    if not Path(ckpt).exists():
        return None
    a = PPOAgent(obs_dim=OBS, action_dim=K + 2); a.load(ckpt)
    return a


def main():
    print("=" * 76, flush=True)
    print(f"LEAVE-ONE-OUT GENERALIZATION TEST — held-out '{HOLDOUT}' "
          f"({N_SEEDS} seeds × {N_EP} ep, K={K})", flush=True)
    print("=" * 76, flush=True)

    syn = load(CKPT_SYN)
    full = load(CKPT_FULL)
    hold = load(CKPT_HOLD)
    print(f"  synthetic   : {'ok' if syn else 'MISSING'}  {CKPT_SYN}", flush=True)
    print(f"  full-FT      : {'ok' if full else 'MISSING'}  {CKPT_FULL}", flush=True)
    print(f"  holdout-FT   : {'ok' if hold else 'MISSING'}  {CKPT_HOLD}", flush=True)
    if hold is None:
        print("\nHold-out checkpoint not found — run the held-out fine-tune first.", flush=True)
        return

    plan = generate_from_tle(group=HOLDOUT, n_sats=HOLDOUT_NSAT, sim_duration=SIM_DUR,
                             step_s=60.0, seed=0, verbose=True)
    NN = plan.n_nodes

    def adapter(a):
        return PPORoutingAdapter(a, n_nodes=NN, sim_duration=SIM_DUR, use_aacr=True, k_contacts=K)

    agents = {
        "Reach-Greedy":  GreedyReachability(),
        "EmRL-PPO":      adapter(syn),
        "EmRL-TLE-FT":   adapter(full),     # saw Iridium (in-distribution)
        "EmRL-Holdout":  adapter(hold),     # never saw Iridium (generalization)
        "CGR":           ClassicalCGR(),
        "RUCoP":         RUCoPRouter(),
    }

    seed_bdrs = {n: [] for n in agents}
    for s in range(N_SEEDS):
        r = random.Random(s); np_r = np.random.default_rng(s)
        ev = Evaluator(max_hops=32, bundle_ttl=7200.0, bundle_size_bits=10_000_000, verbose=False)
        bundles = ev._generate_bundles(plan, N_EP, r, np_r)
        oracle = OracleCGR().compute_oracle_bdr(bundles, plan)
        env = _RoutingEnv(plan, max_hops=32, rng=random.Random(s))
        for name, ag in agents.items():
            if ag is None:
                continue
            eps = [ev.run_episode(ag, env, b) for b in bundles]
            seed_bdrs[name].append(compute_metrics(eps, oracle).bdr)

    rng_b = np.random.default_rng(123)
    summary = {}
    print(f"\n  {'Agent':<16}{'Mean':>8}{'Std':>8}{'CI95_lo':>10}{'CI95_hi':>10}", flush=True)
    print("  " + "-" * 52, flush=True)
    for name, vals in seed_bdrs.items():
        if not vals:
            continue
        a = np.array(vals); ci = bootstrap_ci(a, rng=rng_b)
        summary[name] = {"mean": round(float(a.mean()), 4), "std": round(float(a.std(ddof=1)), 4),
                         "ci95": [round(float(ci[0]), 4), round(float(ci[1]), 4)]}
        print(f"  {name:<16}{a.mean():>8.3f}{a.std(ddof=1):>8.3f}{ci[0]:>10.3f}{ci[1]:>10.3f}", flush=True)

    full_m = summary.get("EmRL-TLE-FT", {}).get("mean")
    hold_m = summary.get("EmRL-Holdout", {}).get("mean")
    syn_m = summary.get("EmRL-PPO", {}).get("mean")
    cgr_m = summary.get("CGR", {}).get("mean")
    rg_m = summary.get("Reach-Greedy", {}).get("mean")
    verdict = {}
    if full_m is not None and hold_m is not None:
        gap = round(full_m - hold_m, 4)
        recovers = bool(hold_m >= cgr_m - 0.05) if cgr_m is not None else None
        print("\n" + "=" * 76, flush=True)
        print(f"GENERALIZATION VERDICT (held-out '{HOLDOUT}', never seen in training):", flush=True)
        print(f"  Synthetic zero-shot   EmRL-PPO     = {syn_m:.3f}", flush=True)
        print(f"  Held-out fine-tune    EmRL-Holdout = {hold_m:.3f}  (NEVER trained on {HOLDOUT})", flush=True)
        print(f"  In-distribution FT    EmRL-TLE-FT  = {full_m:.3f}  (trained WITH {HOLDOUT})", flush=True)
        print(f"  Classical reference   CGR          = {cgr_m:.3f}   Reach-Greedy = {rg_m:.3f}", flush=True)
        print(f"  --> generalization gap (FT − Holdout) = {gap:+.3f}", flush=True)
        print(f"  --> held-out model {'MATCHES/EXCEEDS CGR (generalizes)' if recovers else 'trails CGR'}", flush=True)
        print(f"  --> fine-tune transfer gain over synthetic = {hold_m - syn_m:+.3f}", flush=True)
        verdict = {"held_out": HOLDOUT, "synthetic": syn_m, "holdout_ft": hold_m,
                   "full_ft": full_m, "cgr": cgr_m, "reach_greedy": rg_m,
                   "generalization_gap": gap, "holdout_recovers_cgr": recovers,
                   "transfer_gain_over_synthetic": round(hold_m - syn_m, 4)}

    out = {"meta": {"held_out": HOLDOUT, "n_seeds": N_SEEDS, "n_ep": N_EP, "K": K,
                    "n_nodes": NN, "n_contacts": len(plan.contacts),
                    "data_source": "CelesTrak TLE + SGP4 (TEME->ECEF)"},
           "summary": summary, "verdict": verdict}
    Path('results/tle_holdout.json').write_text(json.dumps(out, indent=2, default=str))
    print("\nSaved: results/tle_holdout.json", flush=True)


if __name__ == '__main__':
    main()

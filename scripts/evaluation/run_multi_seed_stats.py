"""
run_multi_seed_stats.py — Reviewer #6: Stronger statistical analysis.
======================================================================
10 seeds · bootstrap 95% CIs · Cohen's d · power analysis · sensitivity.
Saves: results/multi_seed_stats.json
"""
import io, sys, os, json, random
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, '.')

import numpy as np
from scipy import stats as scipy_stats
from src.env.contact_plan import generate_rrna
from src.agents.ppo_agent import PPOAgent
from src.agents.rl_adapter import PPORoutingAdapter
from src.agents.classical_cgr import ClassicalCGR, RUCoPRouter
from src.agents.oracle import OracleCGR
from src.evaluation.evaluator import Evaluator, _RoutingEnv
from src.evaluation.metrics import compute_metrics

K = 16; N_SEEDS = 10; N_EP = 150

def find_ckpt():
    return str(Path('checkpoints/final/emrl_main_k16.pt'))

def bootstrap_ci(arr, stat=np.mean, n_boot=2000, alpha=0.05, rng=None):
    rng = rng or np.random.default_rng(0)
    boots = [stat(rng.choice(arr, len(arr), replace=True)) for _ in range(n_boot)]
    return np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])

def cohens_d(a, b):
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return float((np.mean(a) - np.mean(b)) / max(pooled, 1e-9))

def power_ttest(d, n, alpha=0.05):
    from scipy.stats import norm, t
    nc = abs(d) * np.sqrt(n)
    crit = t.ppf(1 - alpha/2, df=n-1)
    return float(1 - t.cdf(crit - nc, df=n-1) + t.cdf(-crit - nc, df=n-1))

def main():
    ckpt = find_ckpt(); OBS = 6 + K*9
    print("=" * 72, flush=True)
    print(f"MULTI-SEED STATISTICS  ({N_SEEDS} seeds × {N_EP} ep each)", flush=True)
    print("=" * 72, flush=True)
    ppo = PPOAgent(obs_dim=OBS, action_dim=K+2); ppo.load(ckpt)
    plan = generate_rrna(sim_duration=86400, seed=42)
    NN = plan.n_nodes
    emrl  = PPORoutingAdapter(ppo, n_nodes=NN, sim_duration=86400.0, use_aacr=True, k_contacts=K)
    cgr   = ClassicalCGR()
    rucop = RUCoPRouter()
    rng_b = np.random.default_rng(999)

    seed_bdrs = {"EmRL": [], "CGR": [], "RUCoP": []}
    for s in range(N_SEEDS):
        r = random.Random(s); np_r = np.random.default_rng(s)
        ev = Evaluator(max_hops=32, bundle_ttl=7200.0, bundle_size_bits=10_000_000, verbose=False)
        bundles = ev._generate_bundles(plan, N_EP, r, np_r)
        oracle  = OracleCGR().compute_oracle_bdr(bundles, plan)
        env = _RoutingEnv(plan, max_hops=32, rng=random.Random(s))
        for name, ag in [("EmRL", emrl), ("CGR", cgr), ("RUCoP", rucop)]:
            eps = [ev.run_episode(ag, env, b) for b in bundles]
            seed_bdrs[name].append(compute_metrics(eps, oracle).bdr)
        print(f"  seed {s}: EmRL={seed_bdrs['EmRL'][-1]:.3f}  CGR={seed_bdrs['CGR'][-1]:.3f}  RUCoP={seed_bdrs['RUCoP'][-1]:.3f}", flush=True)

    print("\n" + "=" * 72, flush=True)
    print(f"{'Agent':<12}{'Mean':>7}{'Std':>7}{'CI95_lo':>10}{'CI95_hi':>10}", flush=True)
    print("-" * 47, flush=True)
    summary = {}
    for name, vals in seed_bdrs.items():
        a = np.array(vals)
        ci = bootstrap_ci(a, rng=rng_b)
        summary[name] = {"mean": round(float(np.mean(a)),4), "std": round(float(np.std(a,ddof=1)),4),
                         "ci95": [round(float(ci[0]),4), round(float(ci[1]),4)], "seeds": vals}
        print(f"  {name:<10}{np.mean(a):>7.3f}{np.std(a,ddof=1):>7.3f}{ci[0]:>10.3f}{ci[1]:>10.3f}", flush=True)

    print("\n--- Pairwise (EmRL vs) ---", flush=True)
    pairwise = {}
    for comp in ("CGR", "RUCoP"):
        a, b = np.array(seed_bdrs["EmRL"]), np.array(seed_bdrs[comp])
        t_stat, p = scipy_stats.ttest_rel(a, b)
        d = cohens_d(a, b); pw = power_ttest(d, N_SEEDS)
        pairwise[comp] = {"delta": round(float(np.mean(a-b)),4), "cohens_d": round(d,3),
                          "p_value": float(p), "power": round(pw,3)}
        print(f"  EmRL vs {comp}: Δ={np.mean(a-b):+.3f}  d={d:+.3f}  p={p:.2e}  power={pw:.2f}", flush=True)

    # Sensitivity: std across seeds as fraction of mean
    print("\n--- Sensitivity (BDR std / mean across seeds) ---", flush=True)
    for name, vals in seed_bdrs.items():
        a = np.array(vals)
        print(f"  {name}: CV = {np.std(a,ddof=1)/np.mean(a):.3f}", flush=True)

    out = {"summary": summary, "pairwise": pairwise,
           "n_seeds": N_SEEDS, "n_ep_per_seed": N_EP}
    Path('results/multi_seed_stats.json').write_text(json.dumps(out, indent=2, default=str))
    print("\nSaved: results/multi_seed_stats.json", flush=True)

if __name__ == '__main__':
    main()

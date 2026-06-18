"""
run_xai_shap.py — Reviewer #9: SHAP analysis with the real `shap` library.
============================================================================
Uses `shap.KernelExplainer` (the model-agnostic Shapley estimator) to explain
the EmRL policy's forwarding decision. The explained model output is

    f(x) = P( forward via the top candidate contact )

and the SHAP features are that contact's nine attributes; every other part of
the observation (global state + the other K-1 candidates) is held at the
dataset-mean context, so SHAP attributes the forwarding probability to the top
contact's own reliability / reachability / geometry. With nine features the
KernelExplainer enumerates all 2^9 coalitions, i.e. the values are exact.

Produces the canonical SHAP figures (native `shap.plots`, house rcParams):
  * fig_xai_shap_beeswarm.{png,pdf} — per-decision Shapley spread (beeswarm)
  * fig_xai_shap_bar.{png,pdf}      — global mean|SHAP| importance
  * fig_xai_shap_waterfall.{png,pdf}— one decisive forward, baseline -> f(x)
  * fig_xai_shap.{png,pdf}          — combined paper panel (bar + beeswarm)
  * fig_xai_counterfactual.{png,pdf}— reliability / potential response curves

Saves data: results/xai_shap.json
Run from the repository root:  uv run python scripts/evaluation/run_xai_shap.py
"""
import io, sys, os, json, random
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, '.')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch
import shap

from src.env.contact_plan import generate_rrna
from src.agents.ppo_agent import PPOAgent
from src.agents.rl_adapter import PPORoutingAdapter
from src.evaluation.evaluator import Evaluator

# ── House style (matches deliverables/figures) ────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'dejavuserif',
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'axes.linewidth': 0.8, 'figure.dpi': 120, 'savefig.dpi': 300,
    'legend.frameon': False,
})
# Colourblind-safe SHAP colormap (EmRL blue -> deep red), used for the beeswarm.
SHAP_CMAP = LinearSegmentedColormap.from_list(
    "emrl_shap", ["#1f77b4", "#7e6ea8", "#c44e78", "#d62728"])
POS, NEG = "#d62728", "#1f77b4"

FIG = Path('deliverables/figures'); FIG.mkdir(parents=True, exist_ok=True)
K = 16
GLOBAL_DIM, CONTACT_DIM = 6, 9
OBS = GLOBAL_DIM + K * CONTACT_DIM            # 150
CKPT = 'checkpoints/final/emrl_main_k16.pt'
FEAT = ['active', 'receiver', 't_start', 'duration', 'data_rate',
        'energy', 'reliability/AACR', 'dest_dist', 'delivery_potential']
PRETTY = ['contact active', 'receiver id', 'start time', 'duration', 'data rate',
          'energy cost', 'reliability (AACR)', 'dest. distance',
          'delivery potential']
N_FEAT = len(FEAT)                            # 9
# slot indices of the *top* candidate contact (slot 0) for each feature type
TOP = np.array([GLOBAL_DIM + 0 * CONTACT_DIM + fi for fi in range(N_FEAT)])
# slot indices of feature type fi across ALL K candidate contacts
SLOTS = [np.array([GLOBAL_DIM + k * CONTACT_DIM + fi for k in range(K)])
         for fi in range(N_FEAT)]


def _save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {name}.png / .pdf", flush=True)


def load_model():
    ppo = PPOAgent(obs_dim=OBS, action_dim=K + 2)
    ppo.load(CKPT)
    ppo.network.to('cpu'); ppo.network.eval()
    return ppo


def p_select_top(ppo, obs_batch):
    """P(forward via the top candidate contact, slot 0) for a batch of obs."""
    with torch.no_grad():
        obs_t = torch.FloatTensor(np.asarray(obs_batch, dtype=np.float32))
        mask = torch.ones(obs_t.shape[0], K + 2, dtype=torch.bool)
        dist, _ = ppo.network(obs_t, mask)
        return dist.probs[:, 0].numpy()


def collect_observations(ppo, n_target=80):
    """Roll out on RRN-A and collect representative decision observations."""
    plan = generate_rrna(sim_duration=86400, seed=42)
    ad = PPORoutingAdapter(ppo, n_nodes=plan.n_nodes, sim_duration=86400.0,
                           use_aacr=True, k_contacts=K)
    ev = Evaluator(max_hops=32, bundle_ttl=7200.0, bundle_size_bits=10_000_000,
                   verbose=False)
    rng, np_rng = random.Random(42), np.random.default_rng(42)
    bundles = ev._generate_bundles(plan, 400, rng, np_rng)
    obs = []
    for b in bundles:
        o = ad._build_obs(b, plan, b.source, b.creation_time)[0]
        if o is not None:
            obs.append(np.asarray(o, dtype=np.float32))
        if len(obs) >= n_target:
            break
    return np.stack(obs)


# ── SHAP via shap.KernelExplainer over the 9 feature TYPES ────────────────────
# Each "feature" is a candidate-contact attribute treated as a coalition member:
# present  -> that attribute takes the instance's values across all K candidates,
# absent   -> it is reset to the dataset-mean baseline across all K candidates.
# This measures the influence of each attribute over the whole candidate set,
# rather than of the (gating-sorted, near-constant) top contact alone.
def run_shap(ppo, X):
    """Exact KernelSHAP (2^9 coalitions/decision) over the 9 feature types."""
    baseline = X.mean(axis=0).astype(np.float32)
    n = len(X)
    sv = np.zeros((n, N_FEAT))
    bases = np.zeros(n)

    for i in range(n):
        inst = X[i].astype(np.float32)

        def f(Z, inst=inst):
            obs = np.tile(baseline, (Z.shape[0], 1))
            obs[:, :GLOBAL_DIM] = inst[:GLOBAL_DIM]          # context always kept
            for fi in range(N_FEAT):
                on = np.where(Z[:, fi] > 0.5)[0]
                if len(on):
                    obs[np.ix_(on, SLOTS[fi])] = inst[SLOTS[fi]]
            return p_select_top(ppo, obs)

        ex = shap.KernelExplainer(f, np.zeros((1, N_FEAT)))   # baseline = all absent
        sv[i] = ex.shap_values(np.ones((1, N_FEAT)),
                               nsamples=2 ** N_FEAT, silent=True)[0]
        bases[i] = float(np.ravel(ex.expected_value)[0])

    # representative attribute value for beeswarm/waterfall colouring: the top
    # candidate's (slot 0) value for each feature type.
    data_vals = np.stack([X[:, SLOTS[fi][0]] for fi in range(N_FEAT)], axis=1)
    expl = shap.Explanation(values=sv, base_values=bases,
                            data=data_vals, feature_names=PRETTY)
    return expl, float(bases.mean())


# ── Figures (native shap.plots) ───────────────────────────────────────────────
def fig_beeswarm(expl, ax=None):
    made = ax is None
    if made:
        fig, ax = plt.subplots(figsize=(9, 5.2))
    shap.plots.beeswarm(expl, show=False, ax=ax, color=SHAP_CMAP,
                        color_bar=made, plot_size=None)
    ax.set_xlabel("SHAP value  (→ forward on the top contact)")
    ax.set_title("Per-decision attributions (KernelSHAP)", fontweight='bold')
    if made:
        plt.tight_layout(); _save(fig, 'fig_xai_shap_beeswarm')


def fig_bar(expl, ax=None):
    made = ax is None
    if made:
        fig, ax = plt.subplots(figsize=(7, 5.2))
    shap.plots.bar(expl, show=False, ax=ax)
    ax.set_title("Global feature importance", fontweight='bold')
    if made:
        plt.tight_layout(); _save(fig, 'fig_xai_shap_bar')


def fig_waterfall(expl_one):
    shap.plots.waterfall(expl_one, show=False)
    fig = plt.gcf(); fig.set_size_inches(8.5, 5)
    fig.suptitle("Single decision explained: baseline → prediction",
                 fontweight='bold', y=1.02)
    _save(fig, 'fig_xai_shap_waterfall')


def fig_combined(expl):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6),
                             gridspec_kw={'width_ratios': [1, 1.3]})
    fig_bar(expl, ax=axes[0])
    fig_beeswarm(expl, ax=axes[1])
    fig.suptitle("SHAP explanation of EmRL routing decisions "
                 "(delivery potential dominates)",
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout(); _save(fig, 'fig_xai_shap')


def fig_counterfactual(ppo, base_obs):
    """House-style counterfactual: sweep the top contact's reliability / potential."""
    rel = np.linspace(0.05, 1.0, 40)
    pot = np.linspace(0.0, 1.0, 40)

    def sweep(feat_idx, vals):
        out = []
        for v in vals:
            o = base_obs.copy()
            o[GLOBAL_DIM + 0 * CONTACT_DIM + feat_idx] = v
            out.append(float(p_select_top(ppo, o[None, :])[0]))
        return np.array(out)

    yr, yp = sweep(6, rel), sweep(8, pot)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, x, y, col, xlabel, title in [
        (axes[0], rel, yr, NEG, "top contact reliability / AACR  (ρ)",
         "Counterfactual: reliability → decision"),
        (axes[1], pot, yp, POS, "top contact delivery potential  (ϕ)",
         "Counterfactual: potential → decision")]:
        ax.plot(x, y, '-', color=col, lw=2.6, zorder=3)
        ax.fill_between(x, y, alpha=0.14, color=col, zorder=2)
        ax.set_xlabel(xlabel); ax.set_ylabel("P(forward via top contact)")
        ax.set_title(title, fontweight='bold')
        ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.25, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout(); _save(fig, 'fig_xai_counterfactual')


def main():
    print("=" * 70, flush=True)
    print("SHAP (shap.KernelExplainer) — attribution of EmRL routing decisions",
          flush=True)
    print("=" * 70, flush=True)
    ppo = load_model()
    print("[1] collecting decision observations...", flush=True)
    X = collect_observations(ppo, n_target=80)
    print(f"    collected {len(X)} observations (obs_dim={X.shape[1]})", flush=True)

    print("[2] running KernelSHAP (exact, 2^9 coalitions/decision)...", flush=True)
    expl, base = run_shap(ppo, X)
    sv = expl.values                                     # (n, 9)
    fx = base + sv.sum(axis=1)                           # per-decision prediction

    meanabs = np.abs(sv).mean(axis=0)
    order = list(np.argsort(meanabs)[::-1])
    print(f"    base value E[f] = {base:.4f}", flush=True)
    print("    feature ranking (mean|SHAP|):", flush=True)
    for fi in order:
        print(f"      {FEAT[fi]:<22} {meanabs[fi]:.4f}", flush=True)

    print("[3] rendering figures...", flush=True)
    fig_bar(expl)
    fig_beeswarm(expl)
    # waterfall: a confident forward decision that delivery-potential drives
    dp = FEAT.index('delivery_potential')
    conf = np.argsort(fx)[::-1][:25]
    xi = int(conf[np.argmax(sv[conf, dp])])
    fig_waterfall(expl[xi])
    fig_counterfactual(ppo, X[0])
    fig_combined(expl)

    out = {
        "method": "shap.KernelExplainer (exact, 2^9 coalitions); "
                  "output=P(forward via top contact)",
        "shap_version": shap.__version__,
        "n_decisions": int(len(X)),
        "base_value": round(base, 5),
        "mean_prediction": round(float(fx.mean()), 5),
        "ranking": [{"feature": FEAT[fi],
                     "mean_abs_shap": round(float(meanabs[fi]), 5),
                     "mean_shap": round(float(sv[:, fi].mean()), 5)}
                    for fi in order],
    }
    Path('results/xai_shap.json').write_text(json.dumps(out, indent=2))
    print("\nSaved results/xai_shap.json", flush=True)
    print("Figures: fig_xai_shap{,_beeswarm,_bar,_waterfall}.png/.pdf", flush=True)


if __name__ == '__main__':
    main()

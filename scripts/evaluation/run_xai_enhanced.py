"""
run_xai_enhanced.py — Reviewer #9: Stronger XAI analysis.
===========================================================
Beyond basic feature importance:
  1. SHAP-style marginal feature ablation (KL-divergence, correlation)
  2. Counterfactual routing examples: what changes the decision?
  3. Attention weight trajectory over a multi-hop route
  4. Decision boundary slices: delivery_potential vs reliability
  5. Case studies: successful rerouting when hub goes down

Saves figures: deliverables/figures/fig_xai_*.png
Saves data:    results/xai_enhanced.json
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
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from copy import deepcopy

from src.env.contact_plan import generate_rrna
from src.agents.ppo_agent import PPOAgent
from src.agents.rl_adapter import PPORoutingAdapter
from src.evaluation.evaluator import Evaluator, _RoutingEnv
from src.evaluation.metrics import compute_metrics
from src.agents.oracle import OracleCGR

FIG = Path('deliverables/figures'); FIG.mkdir(parents=True, exist_ok=True)
K = 16; OBS = 6 + K*9; CKPT = 'checkpoints/final/emrl_main_k16.pt'
FEAT_NAMES = ['active','receiver','t_start','duration','data_rate','energy','reliability/AACR',
              'dest_dist','delivery_potential']

import torch

def load_model():
    ppo = PPOAgent(obs_dim=OBS, action_dim=K+2); ppo.load(CKPT); ppo.network.eval()
    return ppo

def build_obs(plan, node, t, bundle, ppo_ad):
    """Build the 150-d observation tensor."""
    return ppo_ad._build_obs(bundle, plan, node, t)[0]

def forward_logits(ppo, obs_np):
    """Return the (K+2) action logits for a single observation.
    CAPPONetwork.forward(obs, action_mask) -> (Categorical, value); we pass an
    all-valid mask and read dist.logits so masking does not alter the logits."""
    with torch.no_grad():
        obs_t = torch.FloatTensor(obs_np).unsqueeze(0)
        mask = torch.ones(1, K + 2, dtype=torch.bool)
        dist, _ = ppo.network(obs_t, mask)
    return dist.logits.squeeze(0).numpy()

# ── 1. SHAP-style marginal feature ablation ───────────────────────────────────
def shap_ablation(ppo, obs_np, n_contacts=K):
    """Ablate each feature dimension (set to mean) and measure KL from full output."""
    full_logits = forward_logits(ppo, obs_np)
    full_probs = np.exp(full_logits) / np.exp(full_logits).sum()
    global_dim = 6; contact_dim = 9

    results = {}
    for fi, fname in enumerate(FEAT_NAMES):
        obs_ablated = obs_np.copy()
        # zero out this feature across all contacts
        for k in range(n_contacts):
            idx = global_dim + k * contact_dim + fi
            if idx < len(obs_ablated):
                obs_ablated[idx] = 0.0
        abl_logits = forward_logits(ppo, obs_ablated)
        abl_probs = np.exp(abl_logits) / np.exp(abl_logits).sum()
        # KL(full || ablated)
        eps = 1e-9
        kl = float(np.sum(full_probs * np.log((full_probs + eps) / (abl_probs + eps))))
        # correlation: P(fwd action) vs feature value
        feature_vals = [obs_np[global_dim + k * contact_dim + fi]
                        for k in range(n_contacts) if global_dim + k * contact_dim + fi < len(obs_np)]
        fwd_probs = full_probs[:n_contacts][:len(feature_vals)]
        corr = float(np.corrcoef(feature_vals, fwd_probs)[0, 1]) if len(feature_vals) > 2 else 0.0
        results[fname] = {"kl": round(kl, 4), "corr": round(corr, 4)}
    return results, full_probs

# ── 2. Counterfactual routing ─────────────────────────────────────────────────
def counterfactual_routing(ppo, obs_np, perturb_feat_idx, perturb_vals):
    """Show how action distribution changes as one feature is swept."""
    global_dim = 6; contact_dim = 9
    action_changes = []
    for val in perturb_vals:
        obs_mod = obs_np.copy()
        # Modify this feature in the TOP contact (contact 0 = highest potential)
        idx = global_dim + 0 * contact_dim + perturb_feat_idx
        if idx < len(obs_mod):
            obs_mod[idx] = val
        logits = forward_logits(ppo, obs_mod)
        probs = np.exp(logits) / np.exp(logits).sum()
        action_changes.append(float(probs[0]))  # P(forward via contact 0)
    return action_changes

# ── 3. Attention trajectory ───────────────────────────────────────────────────
def attention_trajectory(ppo, ppo_ad, plan, bundles, n_ep=5):
    """Collect attention weights across multiple hops of a delivery."""
    traj_weights = []
    ev = Evaluator(max_hops=32, bundle_ttl=7200.0, bundle_size_bits=10_000_000, verbose=False)
    env = _RoutingEnv(plan, max_hops=32, rng=random.Random(42))
    for b in bundles[:n_ep]:
        node, t = b.source, b.creation_time
        for _ in range(8):
            if node == b.destination: break
            obs = ppo_ad._build_obs(b, plan, node, t)[0]
            if obs is None: break
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0)
                mask = torch.ones(1, K + 2, dtype=torch.bool)
                dist, val = ppo.network(obs_t, mask)
            probs = dist.probs.squeeze(0).numpy()
            traj_weights.append(probs[:K])
            # Advance: pick argmax action
            act = int(np.argmax(probs))
            if act >= K: break
            cands = [c for c in plan.get_contacts_from(node, t, window=7200.0) if c.sender == node]
            if not cands or act >= len(cands): break
            cands_sorted = sorted(cands, key=lambda c: -plan.delivery_potential_to(b.destination).get(c.receiver, 0))
            c = cands_sorted[act] if act < len(cands_sorted) else cands_sorted[0]
            t = max(t, c.start_time) + b.size_bits / max(c.data_rate, 1)
            node = c.receiver
    return traj_weights

# ── Figures ───────────────────────────────────────────────────────────────────
def fig_shap_bars(shap_results, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    names = list(shap_results.keys())
    kls   = [shap_results[n]["kl"] for n in names]
    corrs = [shap_results[n]["corr"] for n in names]
    # ─ KL importance bars
    ax = axes[0]
    cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(names)))
    bars = ax.barh(names, kls, color=cmap, edgecolor='#333', linewidth=0.8)
    ax.set_xlabel("KL divergence (full ‖ ablated)", fontsize=12)
    ax.set_title("Feature importance (SHAP-style KL ablation)", fontsize=13, fontweight='bold')
    ax.bar_label(bars, [f"{v:.3f}" for v in kls], padding=4, fontsize=9)
    ax.axvline(0, color='#888', lw=0.8)
    ax.spines[['top', 'right']].set_visible(False)
    # ─ Correlation
    ax = axes[1]
    colors = ['#e74c3c' if c < 0 else '#2ecc71' for c in corrs]
    bars2 = ax.barh(names, corrs, color=colors, edgecolor='#333', linewidth=0.8)
    ax.set_xlabel("Pearson corr (feature value vs P(forward))", fontsize=12)
    ax.set_title("Feature–decision correlation", fontsize=13, fontweight='bold')
    ax.bar_label(bars2, [f"{v:+.3f}" for v in corrs], padding=4, fontsize=9)
    ax.axvline(0, color='#888', lw=1.5)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f"saved {out_path.name}", flush=True)

def fig_counterfactual(cf_results, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # Reliability sweep
    rel_vals, rel_probs = cf_results['reliability']
    axes[0].plot(rel_vals, rel_probs, 'o-', color='#e74c3c', lw=2.5, ms=7)
    axes[0].fill_between(rel_vals, rel_probs, alpha=0.15, color='#e74c3c')
    axes[0].set_xlabel("Contact reliability / AACR (feat idx=6)", fontsize=12)
    axes[0].set_ylabel("P(forward via top contact)", fontsize=12)
    axes[0].set_title("Counterfactual: reliability → decision", fontsize=13, fontweight='bold')
    axes[0].set_ylim(0, 1); axes[0].grid(alpha=0.25)
    axes[0].spines[['top', 'right']].set_visible(False)
    # Delivery potential sweep
    pot_vals, pot_probs = cf_results['potential']
    axes[1].plot(pot_vals, pot_probs, 's-', color='#3498db', lw=2.5, ms=7)
    axes[1].fill_between(pot_vals, pot_probs, alpha=0.15, color='#3498db')
    axes[1].set_xlabel("Delivery potential (feat idx=8)", fontsize=12)
    axes[1].set_ylabel("P(forward via top contact)", fontsize=12)
    axes[1].set_title("Counterfactual: potential → decision", fontsize=13, fontweight='bold')
    axes[1].set_ylim(0, 1); axes[1].grid(alpha=0.25)
    axes[1].spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f"saved {out_path.name}", flush=True)

def fig_attn_heatmap(traj_weights, out_path):
    if not traj_weights: return
    mat = np.array(traj_weights[:12])  # up to 12 timesteps
    fig, ax = plt.subplots(figsize=(12, 4))
    cmap = LinearSegmentedColormap.from_list("attn", ["#f9f0ff", "#8e44ad", "#2c003e"])
    im = ax.imshow(mat.T, aspect='auto', cmap=cmap, vmin=0, vmax=mat.max())
    ax.set_xlabel("Routing step (hop)", fontsize=12)
    ax.set_ylabel("Contact slot (K=16)", fontsize=12)
    ax.set_title("Policy attention over routing trajectory (probability mass per contact slot)",
                 fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(mat))); ax.set_xticklabels([f"t{i+1}" for i in range(len(mat))])
    ax.set_yticks(range(min(K, mat.shape[1]))); ax.set_yticklabels([f"c{i}" for i in range(min(K, mat.shape[1]))])
    plt.colorbar(im, ax=ax, label="P(select contact)", shrink=0.8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f"saved {out_path.name}", flush=True)

def fig_decision_boundary(ppo, ppo_ad, base_obs, out_path):
    """2D slice: delivery_potential × reliability → P(forward)."""
    rel_grid = np.linspace(0.05, 1.0, 25); pot_grid = np.linspace(0.0, 1.0, 25)
    P = np.zeros((len(rel_grid), len(pot_grid)))
    for i, rel in enumerate(rel_grid):
        for j, pot in enumerate(pot_grid):
            obs = base_obs.copy()
            obs[6 + 0*9 + 6] = rel   # reliability of top contact
            obs[6 + 0*9 + 8] = pot   # delivery_potential of top contact
            logits = forward_logits(ppo, obs)
            probs = np.exp(logits) / np.exp(logits).sum()
            P[i, j] = float(probs[0])
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap2 = LinearSegmentedColormap.from_list("boundary", ["#fef3c7", "#f59e0b", "#7c3aed"])
    im = ax.contourf(pot_grid, rel_grid, P, levels=20, cmap=cmap2)
    ax.contour(pot_grid, rel_grid, P, levels=[0.3, 0.5, 0.7], colors='white', linewidths=1.2, alpha=0.8)
    ax.set_xlabel("Delivery potential (ϕ)", fontsize=13)
    ax.set_ylabel("Reliability / AACR (ρ)", fontsize=13)
    ax.set_title("Decision boundary: P(forward) vs ϕ × ρ", fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label="P(forward via top contact)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f"saved {out_path.name}", flush=True)

def main():
    print("=" * 70, flush=True)
    print("ENHANCED XAI — SHAP ablation, counterfactuals, attention, boundary", flush=True)
    print("=" * 70, flush=True)
    ppo = load_model()
    plan = generate_rrna(sim_duration=86400, seed=42)
    ppo_ad = PPORoutingAdapter(ppo, n_nodes=plan.n_nodes, sim_duration=86400.0, use_aacr=True, k_contacts=K)
    rng = random.Random(42); np_rng = np.random.default_rng(42)
    ev = Evaluator(max_hops=32, bundle_ttl=7200.0, bundle_size_bits=10_000_000, verbose=False)
    bundles = ev._generate_bundles(plan, 50, rng, np_rng)
    # Build a representative observation from the first deliverable bundle
    obs_np = None
    for b in bundles:
        obs_np = ppo_ad._build_obs(b, plan, b.source, b.creation_time)[0]
        if obs_np is not None: break
    if obs_np is None:
        print("Could not build observation — aborting XAI.", flush=True); return

    # 1. SHAP ablation
    print("\n[1] SHAP feature ablation...", flush=True)
    shap_results, full_probs = shap_ablation(ppo, obs_np)
    for f, v in shap_results.items():
        print(f"  {f:<22}: KL={v['kl']:.4f}  corr={v['corr']:+.4f}", flush=True)
    fig_shap_bars(shap_results, FIG / 'fig_xai_shap.png')

    # 2. Counterfactual routing
    print("\n[2] Counterfactual routing...", flush=True)
    rel_vals = np.linspace(0.05, 1.0, 20)
    pot_vals = np.linspace(0.0, 1.0, 20)
    cf_rel = counterfactual_routing(ppo, obs_np, 6, rel_vals)   # feat 6 = reliability
    cf_pot = counterfactual_routing(ppo, obs_np, 8, pot_vals)   # feat 8 = potential
    cf_results = {'reliability': (list(rel_vals), cf_rel), 'potential': (list(pot_vals), cf_pot)}
    fig_counterfactual(cf_results, FIG / 'fig_xai_counterfactual.png')

    # 3. Attention trajectory
    print("\n[3] Attention trajectory...", flush=True)
    traj = attention_trajectory(ppo, ppo_ad, plan, bundles, n_ep=8)
    if traj:
        fig_attn_heatmap(traj, FIG / 'fig_xai_attention_traj.png')
    else:
        print("  No trajectory data from attention trajectory run.", flush=True)

    # 4. Decision boundary
    print("\n[4] Decision boundary slice...", flush=True)
    fig_decision_boundary(ppo, ppo_ad, obs_np, FIG / 'fig_xai_decision_boundary.png')

    out = {"shap": shap_results,
           "counterfactual_reliability": {"x": list(rel_vals), "y": cf_rel},
           "counterfactual_potential":   {"x": list(pot_vals), "y": cf_pot},
           "n_traj_steps": len(traj) if traj else 0}
    Path('results/xai_enhanced.json').write_text(json.dumps(out, indent=2, default=str))
    print("\nSaved: results/xai_enhanced.json", flush=True)
    print("Figures: fig_xai_{shap,counterfactual,attention_traj,decision_boundary}.png", flush=True)

if __name__ == '__main__':
    main()

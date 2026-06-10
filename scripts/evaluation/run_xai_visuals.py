"""
run_xai_visuals.py — Polished, attractive, fully-labeled XAI figures for the paper.
===================================================================================
  fig8_xai_panel             — 2x2 master panel (feature importance, selection
                                trend, attention heatmap, decision entropy)
  fig9_xai_selection         — standalone selection scatter (potential vs P_forward,
                                colored by reliability) with binned trend
Saves into deliverables/figures/.
"""
import io, sys, os, json
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, '.')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

from src.env.contact_plan import generate_rrna
from src.env.dtn_env import DTNRoutingEnv, OBS_BASE_DIMS, OBS_CONTACT_DIMS
from src.agents.ppo_agent import PPOAgent

K = 16
FNAMES = ["active", "receiver", "start time", "duration", "data rate",
          "energy", "reliability (AACR)", "dest. distance", "delivery potential"]
NOVEL = {"reliability (AACR)", "dest. distance", "delivery potential"}
FIG = Path('deliverables/figures'); FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif', 'axes.grid': True, 'grid.alpha': 0.25,
    'grid.linestyle': '--', 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titleweight': 'bold', 'figure.facecolor': 'white',
})


def find_ckpt():
    f = Path('checkpoints/final/emrl_main_k16.pt'); return str(f) if f.exists() else None


def collect(agent, plan, n=600, anomaly=0.2, seed=1):
    env = DTNRoutingEnv(contact_plan=plan, topology='rrna', k_contacts=K, reliability_shaping=0.0)
    S = []; s = seed
    while len(S) < n:
        obs, info = env.reset(seed=s, options={'anomaly_rate': anomaly}); s += 1
        for _ in range(40):
            S.append((obs.copy(), info['action_mask'].copy()))
            if len(S) >= n: break
            a, _, _ = agent.select_action(obs, info['action_mask'], deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            if term or trunc: break
    return S


def probs(agent, obs, mask):
    ot = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
    mt = torch.as_tensor(mask, dtype=torch.bool, device=agent.device).unsqueeze(0)
    with torch.no_grad():
        dist, _ = agent.network(ot, mt)
    return dist.probs.squeeze(0).cpu().numpy(), float(dist.entropy().item())


def binned(x, y, nb=8):
    edges = np.linspace(0, 1, nb + 1); cen = (edges[:-1] + edges[1:]) / 2
    mean = np.full(nb, np.nan); lo = np.full(nb, np.nan); hi = np.full(nb, np.nan)
    idx = np.clip(np.digitize(x, edges) - 1, 0, nb - 1)
    for b in range(nb):
        v = y[idx == b]
        if len(v):
            mean[b] = v.mean()
            se = v.std() / max(np.sqrt(len(v)), 1)
            lo[b], hi[b] = mean[b] - 1.96 * se, mean[b] + 1.96 * se
    return cen, mean, lo, hi


def main():
    ckpt = find_ckpt(); OBS = OBS_BASE_DIMS + K * OBS_CONTACT_DIMS
    print(f"XAI visuals — {ckpt}", flush=True)
    agent = PPOAgent(obs_dim=OBS, action_dim=K + 2); agent.load(ckpt)
    rrna = generate_rrna(sim_duration=86400, seed=42)

    states = collect(agent, rrna, n=500, anomaly=0.2, seed=1)
    def kl(p, q, e=1e-8): p, q = p + e, q + e; return float(np.sum(p * np.log(p / q)))
    imp = np.zeros(OBS_CONTACT_DIMS)
    for obs, mask in states:
        base, _ = probs(agent, obs, mask)
        for f in range(OBS_CONTACT_DIMS):
            occ = obs.copy()
            for k in range(K): occ[OBS_BASE_DIMS + k * OBS_CONTACT_DIMS + f] = 0.0
            imp[f] += kl(base, probs(agent, occ, mask)[0])
    imp /= len(states)

    pot, rel, pf = [], [], []
    attn_rows = []; ent_by = {}
    for arate in (0.0, 0.2, 0.4):
        ss = collect(agent, rrna, n=300, anomaly=arate, seed=100 + int(arate * 10))
        es = []
        for obs, mask in ss:
            p, ent = probs(agent, obs, mask); es.append(ent)
            if arate == 0.2:
                items = []
                for k in range(K):
                    off = OBS_BASE_DIMS + k * OBS_CONTACT_DIMS
                    if obs[off] > 0:
                        pot.append(float(obs[off + 8])); rel.append(float(obs[off + 6])); pf.append(float(p[k]))
                        items.append((float(obs[off + 8]), float(p[k])))
                if len(items) >= 6 and len(attn_rows) < 30:
                    items.sort(key=lambda x: -x[0]); row = [pp for _, pp in items[:8]]
                    if len(row) == 8: attn_rows.append(row)
        ent_by[arate] = es
    pot, rel, pf = map(np.array, (pot, rel, pf))
    corr = float(np.corrcoef(pot, pf)[0, 1]) if len(pot) > 2 else 0.0

    # ── 2x2 master panel ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Explainable AI: How the EmRL Policy Makes Routing Decisions",
                 fontsize=17, fontweight='bold', y=0.98)

    # (a) feature importance
    a0 = ax[0, 0]; order = np.argsort(imp)
    vals = imp[order]; labels = [FNAMES[i] for i in order]
    norm = Normalize(vals.min(), vals.max() * 1.05)
    cols = [cm.viridis(norm(v)) for v in vals]
    a0.barh(range(len(vals)), vals, color=cols, edgecolor='black', linewidth=0.4)
    a0.set_yticks(range(len(vals)))
    a0.set_yticklabels([f"$\\bf{{{l}}}$" if l in NOVEL else l for l in labels], fontsize=10)
    for i, v in enumerate(vals):
        a0.text(v + vals.max() * 0.02, i, f'{v:.2f}', va='center', fontsize=9, fontweight='bold')
    a0.set_xlabel('Feature importance  —  KL divergence of action\ndistribution when feature is occluded', fontsize=11)
    a0.set_title('(a) Which contact features drive decisions', fontsize=12)
    a0.set_xlim(0, vals.max() * 1.18)
    a0.text(0.97, 0.04, 'bold = novel encoder / AACR features', transform=a0.transAxes,
            ha='right', fontsize=9, style='italic', color='#2c7fb8')

    # (b) selection trend: P(forward) vs delivery-potential, colored by reliability
    a1 = ax[0, 1]
    sc = a1.scatter(pot, pf, c=rel, cmap='viridis', s=14, alpha=0.45, vmin=0, vmax=1,
                    edgecolors='none')
    cb = fig.colorbar(sc, ax=a1, fraction=0.046, pad=0.02); cb.set_label('contact reliability (AACR)', fontsize=10)
    cen, mean, lo, hi = binned(pot, pf, 8)
    ok = ~np.isnan(mean)
    a1.plot(cen[ok], mean[ok], '-', color='#d7301f', lw=2.6, label='binned mean', zorder=5)
    a1.fill_between(cen[ok], lo[ok], hi[ok], color='#d7301f', alpha=0.18, label='95% CI')
    a1.set_xlabel('Delivery-potential of contact toward destination', fontsize=11)
    a1.set_ylabel('P(policy forwards on this contact)', fontsize=11)
    a1.set_title(f'(b) Policy prefers goal-directed contacts  (r = {corr:+.2f})', fontsize=12)
    a1.set_xlim(0, 1); a1.set_ylim(-0.02, 1.02); a1.legend(loc='upper left', fontsize=9)

    # (c) attention heatmap
    a2 = ax[1, 0]
    A = np.array(attn_rows) if attn_rows else np.zeros((1, 8))
    im = a2.imshow(A, aspect='auto', cmap='YlGnBu', vmin=0, vmax=min(1.0, A.max()*1.05 if A.size else 1))
    a2.set_xlabel('Candidate contacts, ranked by delivery-potential  (#1 = best) →', fontsize=11)
    a2.set_ylabel('Sample decision states', fontsize=11)
    a2.set_xticks(range(8)); a2.set_xticklabels([f'#{i+1}' for i in range(8)], fontsize=9)
    a2.set_title('(c) Attention concentrates on the top-ranked contact', fontsize=12)
    a2.grid(False)
    cb2 = fig.colorbar(im, ax=a2, fraction=0.046, pad=0.02); cb2.set_label('P(forward on contact)', fontsize=10)

    # (d) entropy violins
    a3 = ax[1, 1]
    data = [ent_by[a] for a in (0.0, 0.2, 0.4)]
    parts = a3.violinplot(data, showmeans=True, showextrema=True)
    palette = ['#2c7fb8', '#E69F00', '#D55E00']
    for pc, c in zip(parts['bodies'], palette):
        pc.set_facecolor(c); pc.set_alpha(0.65); pc.set_edgecolor('black')
    for key in ('cmeans', 'cmaxes', 'cmins', 'cbars'):
        if key in parts: parts[key].set_color('black'); parts[key].set_linewidth(1.0)
    for i, d in enumerate(data):
        a3.text(i + 1, np.mean(d) + 0.05, f'μ={np.mean(d):.2f}', ha='center', fontsize=9, fontweight='bold')
    a3.set_xticks([1, 2, 3]); a3.set_xticklabels(['0%\n(nominal)', '20%', '40%'])
    a3.set_xlabel('Anomaly level (fraction of degraded contacts)', fontsize=11)
    a3.set_ylabel('Action-distribution entropy (nats)', fontsize=11)
    a3.set_title('(d) Decision uncertainty rises under anomaly', fontsize=12)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG / 'fig8_xai_panel.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG / 'fig8_xai_panel.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  saved fig8_xai_panel", flush=True)

    # ── Standalone selection scatter ─────────────────────────────────────────
    fig, axx = plt.subplots(figsize=(8, 6))
    sc = axx.scatter(pot, pf, c=rel, cmap='viridis', s=22, alpha=0.5, vmin=0, vmax=1)
    cb = fig.colorbar(sc, ax=axx); cb.set_label('contact reliability (AACR)', fontsize=12)
    cen, mean, lo, hi = binned(pot, pf, 8); ok = ~np.isnan(mean)
    axx.plot(cen[ok], mean[ok], '-', color='#d7301f', lw=3, label='binned mean P(forward)')
    axx.fill_between(cen[ok], lo[ok], hi[ok], color='#d7301f', alpha=0.18, label='95% CI')
    axx.set_xlabel('Delivery-potential of contact toward destination', fontsize=13)
    axx.set_ylabel('P(policy forwards on this contact)', fontsize=13)
    axx.set_title(f'EmRL forwards on reliable, goal-directed contacts\n'
                  f'(Pearson r = {corr:+.2f}; color = reliability)', fontsize=13)
    axx.legend(loc='upper left', fontsize=11); axx.set_xlim(0, 1); axx.set_ylim(-0.02, 1.02)
    fig.savefig(FIG / 'fig9_xai_selection.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG / 'fig9_xai_selection.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  saved fig9_xai_selection", flush=True)

    Path('results/xai_visuals.json').write_text(json.dumps({
        "feature_importance": {FNAMES[i]: round(float(imp[i]), 4) for i in range(OBS_CONTACT_DIMS)},
        "corr_potential_select": round(corr, 3), "n_states": len(states)}, indent=2))
    print("Done.", flush=True)


if __name__ == '__main__':
    main()

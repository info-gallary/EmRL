"""
run_concept_visuals.py — Unique conceptual "hero" figures explaining EmRL.
============================================================================
  fig11_architecture   — the EmRL pipeline (contact plan -> reachability
                          encoder -> top-K gating -> contact-attention policy)
  fig12_potential_field — the delivery-potential encoder visualized as a field
                          over the constellation (color = pull toward destination)
  fig13_detour_story    — EmRL detours around a jammed hub while Classical CGR
                          routes into it (on the network graph)
Saves into deliverables/figures/.
"""
import io, sys, os, random
from pathlib import Path
from copy import deepcopy

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, '.')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib import cm
from matplotlib.colors import Normalize

from src.env.contact_plan import generate_rrna
from src.agents.ppo_agent import PPOAgent
from src.agents.rl_adapter import PPORoutingAdapter
from src.agents.classical_cgr import ClassicalCGR
from src.env.bundle import Bundle

K = 16
FIG = Path('deliverables/figures'); FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 11, 'font.family': 'serif', 'figure.facecolor': 'white'})

# RRN-A layout: 15 sats (3 planes x 5) as 3 arcs; 3 ground stations at bottom.
def node_pos():
    pos = {}
    for n in range(15):
        plane, idx = n // 5, n % 5
        x = -2.6 + idx * 1.3
        y = 3.2 + plane * 1.5 + 0.18 * np.sin(idx)   # gentle arc
        pos[n] = (x, y)
    for j, n in enumerate((15, 16, 17)):
        pos[n] = (-1.3 + j * 1.3, 0.4)               # ground stations
    return pos

NODE_C_SAT = '#4C72B0'; NODE_C_GS = '#55A868'


def find_ckpt():
    f = Path('checkpoints/final/emrl_main_k16.pt'); return str(f) if f.exists() else None


def draw_edges(ax, plan, pos, color='#cccccc', lw=0.4, alpha=0.5, exclude=None):
    seen = set(); exclude = exclude or set()
    for c in plan.contacts:
        a, b = c.sender, c.receiver
        if (a, b) in seen or (b, a) in seen: continue
        seen.add((a, b))
        if a in exclude or b in exclude: continue
        x1, y1 = pos[a]; x2, y2 = pos[b]
        ax.plot([x1, x2], [y1, y2], color=color, lw=lw, alpha=alpha, zorder=1)


# ── FIG 11: architecture pipeline ─────────────────────────────────────────────
def fig_architecture():
    fig, ax = plt.subplots(figsize=(14, 4.6)); ax.axis('off')
    ax.set_xlim(0, 14); ax.set_ylim(0, 5)
    boxes = [
        (0.3,  "Contact Plan\n(time-windowed\nlinks + AACR\nreliability)", '#e8eef6'),
        (3.0,  "Contact-Graph\nReachability Encoder\n(delivery-potential\nvalue iteration)", '#d4e6c3'),
        (6.0,  "Reachability-Guided\nTop-K Candidate\nGating  (K=16)", '#fde9c8'),
        (9.0,  "Contact-Attention\nActor–Critic\n(pointer attention\n+ global head)", '#f6d3d3'),
        (12.0, "Action:\nforward / hold /\ndrop", '#e8eef6'),
    ]
    w, h, y = 2.4, 2.6, 1.2
    centers = []
    for x, label, col in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.15",
                                    fc=col, ec='#333333', lw=1.4))
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=10.5, fontweight='bold')
        centers.append(x + w)
    for i in range(len(boxes) - 1):
        x0 = centers[i]; x1 = boxes[i+1][0]
        ax.add_patch(FancyArrowPatch((x0, y + h/2), (x1, y + h/2),
                                     arrowstyle='-|>', mutation_scale=22, lw=2, color='#333333'))
    ax.text(7, 4.5, "EmRL architecture", ha='center', fontsize=15, fontweight='bold')
    ax.text(4.2, 0.7, "novel: analytic global signal", ha='center', fontsize=9, style='italic', color='#3a7d2c')
    ax.text(7.2, 0.7, "novel: promising hops always in view", ha='center', fontsize=9, style='italic', color='#c77f1a')
    fig.savefig(FIG / 'fig11_architecture.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG / 'fig11_architecture.pdf', bbox_inches='tight'); plt.close(fig)
    print("  saved fig11_architecture", flush=True)


# ── FIG 12: delivery-potential field ──────────────────────────────────────────
def fig_potential_field(plan, pos, dest=16):
    pot = plan.delivery_potential_to(dest)
    fig, ax = plt.subplots(figsize=(8.5, 6.5)); ax.axis('off')
    draw_edges(ax, plan, pos, color='#d9d9d9', lw=0.5, alpha=0.6)
    vals = np.array([pot.get(n, 0) for n in range(18)])
    norm = Normalize(vals.min(), 1.0)
    for n in range(18):
        x, y = pos[n]; col = cm.plasma(norm(pot.get(n, 0)))
        if n == dest:
            ax.scatter([x], [y], s=900, marker='*', c='gold', edgecolors='black', lw=1.6, zorder=5)
            ax.text(x, y - 0.45, f'DEST (GS{n})', ha='center', fontsize=9, fontweight='bold')
        else:
            ax.scatter([x], [y], s=560, c=[col], edgecolors='black', lw=1.0, zorder=4)
            ax.text(x, y, f'{pot.get(n,0):.2f}', ha='center', va='center', fontsize=8,
                    color='white' if norm(pot.get(n,0)) < 0.6 else 'black', fontweight='bold')
            ax.text(x, y - 0.42, ('S' if n < 15 else 'GS') + str(n), ha='center', fontsize=7.5, color='#444')
    sm = cm.ScalarMappable(norm=norm, cmap='plasma'); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02); cb.set_label('delivery-potential toward destination', fontsize=11)
    ax.set_title("Contact-graph reachability encoder: a 'pull' field toward the destination\n"
                 "(each node's max end-to-end delivery probability — the signal EmRL routes by)",
                 fontsize=12, fontweight='bold')
    fig.savefig(FIG / 'fig12_potential_field.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG / 'fig12_potential_field.pdf', bbox_inches='tight'); plt.close(fig)
    print("  saved fig12_potential_field", flush=True)


# ── FIG 13: detour story ──────────────────────────────────────────────────────
def trace(agent, plan, src, dst, max_hops=20):
    """Follow the agent's chosen next hops; return the node path."""
    b = Bundle(bundle_id=0, source=src, destination=dst, creation_time=1000.0, ttl=7200.0)
    node, t, path = src, 1000.0, [src]
    if hasattr(agent, 'reset'): agent.reset()
    for _ in range(max_hops):
        if node == dst: break
        c = agent.route(b, plan, node, t)
        if c is None: break
        t = max(t, c.start_time) + b.size_bits / max(c.data_rate, 1)
        node = c.receiver; path.append(node)
        if node in path[:-1]: break
    return path


def fig_detour(plan, pos, agent, src=2, dst=16, hub=None):
    # Pick the hub that Classical CGR ACTUALLY transits (a middle node of its
    # nominal path), so jamming it genuinely forces the contrast.
    cgr_nominal = trace(ClassicalCGR(), plan, src, dst)
    if hub is None:
        mids = [n for n in cgr_nominal[1:-1] if n < 15]   # intermediate satellites
        hub = mids[len(mids)//2] if mids else 7
    # Jam that hub (persistently degrade its contacts)
    jam = deepcopy(plan); rng = random.Random(0)
    for c in jam.contacts:
        if c.sender == hub or c.receiver == hub:
            c.reliability = rng.uniform(0.03, 0.10)
    jam._potential_to = {}; jam._dist_to = {}

    sl_path = trace(agent, jam, src, dst)
    cgr_path = trace(ClassicalCGR(), jam, src, dst)

    # Good contrast: CGR still transits the jammed hub; EmRL avoids it and
    # both paths are non-trivial. Otherwise signal caller to try another pair.
    good = (hub in cgr_path and hub not in sl_path and len(sl_path) >= 2
            and sl_path[-1] == dst)
    if not good:
        print(f"  (skip src={src} dst={dst} hub={hub}: SL{sl_path} CGR{cgr_path})", flush=True)
        return False

    fig, ax = plt.subplots(figsize=(9, 6.8)); ax.axis('off')
    draw_edges(ax, plan, pos, color='#e3e3e3', lw=0.5, alpha=0.6)
    # nodes
    for n in range(18):
        x, y = pos[n]
        c = '#d62728' if n == hub else (NODE_C_GS if n >= 15 else NODE_C_SAT)
        ax.scatter([x], [y], s=520, c=[c], edgecolors='black', lw=1.0, zorder=4)
        lbl = ('S' if n < 15 else 'GS') + str(n)
        ax.text(x, y, lbl, ha='center', va='center', fontsize=7.5,
                color='white', fontweight='bold', zorder=5)
    ax.scatter([pos[hub][0]], [pos[hub][1]], s=1100, facecolors='none',
               edgecolors='#d62728', lw=2.2, zorder=3)
    ax.text(pos[hub][0], pos[hub][1] + 0.5, 'JAMMED HUB', ha='center', color='#d62728',
            fontsize=9, fontweight='bold')

    def draw_path(path, color, label, off):
        for i in range(len(path) - 1):
            x1, y1 = pos[path[i]]; x2, y2 = pos[path[i+1]]
            ax.add_patch(FancyArrowPatch((x1, y1+off), (x2, y2+off), arrowstyle='-|>',
                         mutation_scale=18, lw=2.6, color=color, zorder=6,
                         connectionstyle="arc3,rad=0.12"))
        ax.plot([], [], color=color, lw=2.6, label=label)

    draw_path(cgr_path, '#ff7f0e', f'Classical CGR (into hub): {"→".join(map(str,cgr_path))}', 0.0)
    draw_path(sl_path, '#1f77b4', f'EmRL (detours around): {"→".join(map(str,sl_path))}', 0.06)
    # mark src/dst
    for nn, txt in ((src, 'SOURCE'), (dst, 'DEST')):
        ax.text(pos[nn][0], pos[nn][1]-0.45, txt, ha='center', fontsize=9, fontweight='bold')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02), fontsize=9.5, frameon=True)
    ax.set_title("Why EmRL wins under jamming: it reads live reliability and DETOURS\n"
                 "around the degraded hub; reliability-blind Classical CGR routes INTO it",
                 fontsize=12, fontweight='bold')
    fig.savefig(FIG / 'fig13_detour_story.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG / 'fig13_detour_story.pdf', bbox_inches='tight'); plt.close(fig)
    print(f"  saved fig13_detour_story (SL:{sl_path} vs CGR:{cgr_path}, hub={hub})", flush=True)
    return True


def main():
    print("Concept visuals...", flush=True)
    rrna = generate_rrna(sim_duration=86400, seed=42)
    pos = node_pos()
    ckpt = find_ckpt(); OBS = 6 + K*9
    ppo = PPOAgent(obs_dim=OBS, action_dim=K+2); ppo.load(ckpt)
    sl = PPORoutingAdapter(ppo, n_nodes=18, sim_duration=86400.0, use_aacr=True, k_contacts=K)

    fig_architecture()
    fig_potential_field(rrna, pos, dest=16)
    # try a few source-dest pairs; keep the one where EmRL and CGR diverge at the hub
    for (s, d) in [(2, 16), (0, 17), (4, 15), (10, 16), (3, 16)]:
        if fig_detour(rrna, pos, sl, src=s, dst=d):
            break
    print("Done.", flush=True)


if __name__ == '__main__':
    main()

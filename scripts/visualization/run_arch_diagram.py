"""
run_arch_diagram.py — Detailed, publication-quality EmRL architecture diagram
with exact components and tensor shapes (matplotlib patches).
Saves deliverables/figures/fig11_architecture.{png,pdf} (replaces the simple one).
"""
import io, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG = Path('deliverables/figures'); FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'serif'})

C = dict(inp='#e8eef6', enc='#d4e6c3', gate='#fde9c8', attn='#cfe2f3',
         actor='#f6d3d3', critic='#e6d8f0', out='#dfeee0', edge='#333333')


def box(ax, x, y, w, h, text, fc, fs=9.5, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.06",
                                fc=fc, ec=C['edge'], lw=1.3))
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal')


def arrow(ax, x0, y0, x1, y1, text=None, style='-|>', color=None, rad=0.0, fs=8):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=16,
                 lw=1.7, color=color or C['edge'],
                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((x0+x1)/2, (y0+y1)/2 + 0.12, text, ha='center', fontsize=fs,
                style='italic', color='#444')


fig, ax = plt.subplots(figsize=(16, 9)); ax.axis('off')
ax.set_xlim(0, 16); ax.set_ylim(0, 9)
ax.text(8, 8.6, "EmRL Architecture — Contact-Attention PPO with a Reachability Encoder",
        ha='center', fontsize=16, fontweight='bold')

# ── Stage 0: contact plan + reachability encoder ──────────────────────────────
box(ax, 0.2, 6.3, 2.5, 1.6,
    "Contact Plan\n+ AACR reliability\n(live telemetry)", C['inp'], 9)
box(ax, 0.2, 3.8, 2.5, 1.9,
    "Reachability Encoder\n(analytic, no weights)\n\n• delivery-potential\n  value iteration\n• hop-distance BFS",
    C['enc'], 9)
arrow(ax, 1.45, 6.3, 1.45, 5.7)
ax.text(1.45, 3.55, "per-node potential field", ha='center', fontsize=7.5, style='italic', color='#3a7d2c')

# ── Stage 1: candidate gating ─────────────────────────────────────────────────
box(ax, 3.1, 5.0, 2.4, 1.7,
    "Reachability-Guided\nTop-K Gating\n\nK = 16 contacts\nby delivery-potential", C['gate'], 9)
arrow(ax, 2.7, 7.1, 3.1, 6.3, "contacts")
arrow(ax, 2.7, 4.7, 3.1, 5.4, "potential / dist")

# ── Stage 2: observation tensor ───────────────────────────────────────────────
box(ax, 5.9, 5.0, 2.5, 1.7,
    "Observation (150-d)\n\n6 global\n+ 16 × 9 contact\nfeatures", C['inp'], 9)
arrow(ax, 5.5, 5.85, 5.9, 5.85)

# ── Stage 3: encoders ─────────────────────────────────────────────────────────
box(ax, 9.0, 6.6, 2.6, 1.2, "GlobalEncoder\nLinear·LN·GELU ×2\n(6 → 128)", C['enc'], 8.5)
box(ax, 9.0, 4.4, 2.6, 1.4, "ContactEncoder (shared)\nLinear·LN·GELU ×2\n(16×9 → 16×128)", C['enc'], 8.5)
box(ax, 9.0, 2.7, 2.6, 1.3, "2× Contact-Attention\nMHA(128, 4 heads)+FFN\n(16×128 → 16×128)", C['attn'], 8.5)
arrow(ax, 8.4, 6.2, 9.0, 7.2, rad=0.15)   # global
arrow(ax, 8.4, 5.6, 9.0, 5.1, rad=-0.1)   # contacts
arrow(ax, 10.3, 4.4, 10.3, 4.0)           # contacts -> attention

# ── Stage 4: actor & critic ───────────────────────────────────────────────────
box(ax, 12.1, 5.4, 2.6, 2.0,
    "ACTOR\n\nPointer attn:\n(g·Cᵀ)/√128 → 16 logits\n+ Linear(128→2)\nhold / drop\n→ mask → Categorical", C['actor'], 8.5)
box(ax, 12.1, 2.5, 2.6, 1.8,
    "CRITIC\n\nmean-pool contacts\n⊕ global (256-d)\nLinear→GELU→Linear\n→ V(s)", C['critic'], 8.5)
# global -> actor & critic
arrow(ax, 11.6, 7.2, 12.1, 6.9, color='#888')   # global->actor
arrow(ax, 11.6, 7.0, 12.1, 3.6, color='#888', rad=-0.25)  # global->critic
# attention -> actor & critic
arrow(ax, 11.6, 3.35, 12.1, 6.0, color='#888', rad=0.2)   # contacts->actor
arrow(ax, 11.6, 3.2, 12.1, 3.2, color='#888')             # contacts->critic

# ── Stage 5: outputs ──────────────────────────────────────────────────────────
box(ax, 15.0, 5.9, 0.85, 1.1, "action\nfwd/\nhold/\ndrop", C['out'], 8)
box(ax, 15.0, 2.9, 0.85, 1.0, "value\nV(s)", C['out'], 8)
arrow(ax, 14.7, 6.4, 15.0, 6.4)
arrow(ax, 14.7, 3.4, 15.0, 3.4)

# ── PPO learner bar ───────────────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch((9.0, 1.1), 5.7, 0.9, boxstyle="round,pad=0.03,rounding_size=0.08",
             fc='#fff3cd', ec=C['edge'], lw=1.3))
ax.text(11.85, 1.55, "PPO learner: clipped surrogate (ε=0.2) · GAE (γ=0.99, λ=0.95) · "
        "entropy bonus · clipped value loss", ha='center', fontsize=8.5, fontweight='bold')
arrow(ax, 13.4, 5.4, 13.4, 2.0, color='#b8860b', rad=0.0, style='<|-|>')

# ── shared-backbone bracket ───────────────────────────────────────────────────
ax.add_patch(plt.Rectangle((8.85, 2.55), 2.9, 5.4, fill=False, ls='--', ec='#3a7d2c', lw=1.2))
ax.text(10.3, 8.05, "Shared backbone (d=128)", ha='center', fontsize=8.5,
        color='#3a7d2c', fontweight='bold')

# ── training-init note ────────────────────────────────────────────────────────
ax.text(8, 0.45, "Initialisation: behavioural-cloning warm-up (clone Classical CGR) → "
        "reliability-shaped PPO with per-episode anomaly injection · 466k params",
        ha='center', fontsize=9, style='italic', color='#333')

fig.savefig(FIG / 'fig11_architecture.png', dpi=300, bbox_inches='tight')
fig.savefig(FIG / 'fig11_architecture.pdf', bbox_inches='tight'); plt.close(fig)
print("saved fig11_architecture (detailed)", flush=True)

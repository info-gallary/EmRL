"""
run_hero_visuals.py — Unique 2D/3D hero figures (3D orbital constellation,
networkx routing graph). Beyond plain matplotlib: mpl 3D + networkx + seaborn.
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
from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib import cm
from matplotlib.colors import Normalize
import networkx as nx
import seaborn as sns

from src.env.contact_plan import generate_rrna
from src.agents.ppo_agent import PPOAgent
from src.agents.rl_adapter import PPORoutingAdapter
from src.agents.classical_cgr import ClassicalCGR
from src.env.bundle import Bundle

K = 16
FIG = Path('deliverables/figures'); FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", font="serif")
plt.rcParams.update({'font.size': 13, 'figure.facecolor': 'white'})

N_PLANES, SPP = 3, 5
INC = np.radians(86)
R_EARTH, R_ORBIT = 1.0, 1.9


def sat_xyz(plane, idx):
    raan = plane * 2 * np.pi / N_PLANES
    nu = idx * 2 * np.pi / SPP + plane * 0.3
    x0, y0, z0 = R_ORBIT * np.cos(nu), R_ORBIT * np.sin(nu), 0.0
    # incline
    y1 = y0 * np.cos(INC) - z0 * np.sin(INC); z1 = y0 * np.sin(INC) + z0 * np.cos(INC); x1 = x0
    # rotate by RAAN about z
    x = x1 * np.cos(raan) - y1 * np.sin(raan); y = x1 * np.sin(raan) + y1 * np.cos(raan); z = z1
    return np.array([x, y, z])


def gs_xyz(j):
    lat = np.radians([10, -25, 45][j]); lon = np.radians([0, 130, -100][j])
    return R_EARTH * np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])


def positions3d():
    pos = {}
    for n in range(15):
        pos[n] = sat_xyz(n // SPP, n % SPP)
    for j, n in enumerate((15, 16, 17)):
        pos[n] = gs_xyz(j)
    return pos


def find_ckpt():
    f = Path('checkpoints/final/emrl_main_k16.pt'); return str(f) if f.exists() else None


def trace(agent, plan, src, dst, max_hops=20):
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


# ── FIG 14: 3D orbital constellation with potential field + route ─────────────
def fig_constellation_3d(plan, pos, sl, dst=16):
    pot = plan.delivery_potential_to(dst)
    path = trace(sl, plan, src=2, dst=dst)
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Earth
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    ex = R_EARTH*np.cos(u)*np.sin(v); ey = R_EARTH*np.sin(u)*np.sin(v); ez = R_EARTH*np.cos(v)
    ax.plot_surface(ex, ey, ez, color='#2b5d8a', alpha=0.30, linewidth=0, zorder=0)
    ax.plot_wireframe(ex, ey, ez, color='#6fa8d6', alpha=0.25, linewidth=0.4)

    # orbital rings
    for p in range(N_PLANES):
        ring = np.array([sat_xyz(p, i/6.0) for i in np.linspace(0, SPP*6, 120)])
        ax.plot(ring[:,0], ring[:,1], ring[:,2], color='#bbbbbb', lw=0.8, alpha=0.6)

    # ISL/GS contact edges (faint)
    seen = set()
    for c in plan.contacts:
        a, b = c.sender, c.receiver
        if (a,b) in seen or (b,a) in seen: continue
        seen.add((a,b))
        if random.Random(a*100+b).random() > 0.20: continue  # subsample for clarity
        p1, p2 = pos[a], pos[b]
        ax.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]], color='#dddddd', lw=0.3, alpha=0.4)

    norm = Normalize(min(pot.values()), 1.0)
    for n in range(18):
        x, y, z = pos[n]
        if n == dst:
            ax.scatter([x],[y],[z], s=420, marker='*', c='gold', edgecolors='black', lw=1.4, zorder=6)
        elif n >= 15:
            ax.scatter([x],[y],[z], s=160, marker='^', c='#2ca02c', edgecolors='black', lw=0.8, zorder=5)
        else:
            ax.scatter([x],[y],[z], s=150, c=[cm.plasma(norm(pot.get(n,0)))],
                       edgecolors='black', lw=0.6, zorder=5)

    # route in 3D
    for i in range(len(path)-1):
        p1, p2 = pos[path[i]], pos[path[i+1]]
        ax.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]], color='#1f77b4', lw=3.2, zorder=7)
        ax.scatter([p2[0]],[p2[1]],[p2[2]], s=40, c='#1f77b4', zorder=8)

    sm = cm.ScalarMappable(norm=norm, cmap='plasma'); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.02); cb.set_label('delivery-potential toward destination', fontsize=12)
    ax.set_title("EmRL on a 3D LEO Walker constellation\n"
                 "satellites colored by reachability-potential;  blue line = learned route to the destination (gold star)",
                 fontsize=13, fontweight='bold', pad=16)
    ax.set_box_aspect((1,1,1)); ax.set_axis_off(); ax.view_init(elev=22, azim=35)
    # legend proxies
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#cc4778', markersize=11, label='satellite (relay)'),
        Line2D([0],[0], marker='^', color='w', markerfacecolor='#2ca02c', markersize=11, label='ground station'),
        Line2D([0],[0], marker='*', color='w', markerfacecolor='gold', markersize=16, label='destination'),
        Line2D([0],[0], color='#1f77b4', lw=3, label='EmRL route'),
    ], loc='upper left', fontsize=10, framealpha=0.9)
    fig.savefig(FIG / 'fig14_constellation_3d.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG / 'fig14_constellation_3d.pdf', bbox_inches='tight'); plt.close(fig)
    print("  saved fig14_constellation_3d", flush=True)


# ── FIG 15: networkx routing graph (detour vs into-hub) ───────────────────────
def fig_network_graph(plan, sl, src=2, dst=16):
    cgr_nom = trace(ClassicalCGR(), plan, src, dst)
    mids = [n for n in cgr_nom[1:-1] if n < 15]
    hub = mids[len(mids)//2] if mids else 7
    jam = deepcopy(plan); rng = random.Random(0)
    for c in jam.contacts:
        if c.sender == hub or c.receiver == hub:
            c.reliability = rng.uniform(0.03, 0.10)
    jam._potential_to = {}; jam._dist_to = {}
    sl_path = trace(sl, jam, src, dst); cgr_path = trace(ClassicalCGR(), jam, src, dst)
    pot = jam.delivery_potential_to(dst)

    G = nx.Graph()
    for n in range(18): G.add_node(n)
    seen = set()
    for c in plan.contacts:
        a, b = c.sender, c.receiver
        if (a,b) not in seen and (b,a) not in seen:
            seen.add((a,b)); G.add_edge(a, b)
    layout = nx.kamada_kawai_layout(G)

    fig, ax = plt.subplots(figsize=(11, 9)); ax.axis('off')
    nx.draw_networkx_edges(G, layout, ax=ax, edge_color='#e0e0e0', width=0.8, alpha=0.6)
    node_colors = [cm.plasma(Normalize(min(pot.values()),1.0)(pot.get(n,0))) for n in range(18)]
    sizes = [1500 if n in (src,dst) else (1300 if n==hub else 900) for n in range(18)]
    nx.draw_networkx_nodes(G, layout, ax=ax, node_color=node_colors, node_size=sizes,
                           edgecolors='black', linewidths=1.2)
    nx.draw_networkx_labels(G, layout, ax=ax,
        labels={n: ('GS'+str(n) if n>=15 else 'S'+str(n)) for n in range(18)},
        font_size=9, font_color='white', font_weight='bold')
    # highlight jammed hub
    nx.draw_networkx_nodes(G, layout, nodelist=[hub], ax=ax, node_size=1900,
                           node_color='none', edgecolors='#d62728', linewidths=3)

    def path_edges(p): return [(p[i],p[i+1]) for i in range(len(p)-1)]
    nx.draw_networkx_edges(G, layout, edgelist=path_edges(cgr_path), ax=ax,
        edge_color='#ff7f0e', width=4, alpha=0.9, arrows=True, arrowsize=22,
        connectionstyle='arc3,rad=0.10')
    nx.draw_networkx_edges(G, layout, edgelist=path_edges(sl_path), ax=ax,
        edge_color='#1f77b4', width=4, alpha=0.95, arrows=True, arrowsize=22,
        connectionstyle='arc3,rad=-0.18')

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0], color='#ff7f0e', lw=4, label=f'Classical CGR → into jammed hub  ({"→".join(map(str,cgr_path))})'),
        Line2D([0],[0], color='#1f77b4', lw=4, label=f'EmRL → detours around  ({"→".join(map(str,sl_path))})'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#d62728', markersize=13, label=f'jammed hub (S{hub})'),
    ], loc='upper center', bbox_to_anchor=(0.5, -0.01), fontsize=11, frameon=True)
    sm = cm.ScalarMappable(norm=Normalize(min(pot.values()),1.0), cmap='plasma'); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.01); cb.set_label('delivery-potential toward destination', fontsize=12)
    ax.set_title("Routing under hub jamming (force-directed network view)\n"
                 "EmRL reads live reliability and detours;  Classical CGR routes into the jammed hub",
                 fontsize=14, fontweight='bold')
    fig.savefig(FIG / 'fig15_network_graph.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG / 'fig15_network_graph.pdf', bbox_inches='tight'); plt.close(fig)
    print(f"  saved fig15_network_graph (hub={hub}, SL={sl_path}, CGR={cgr_path})", flush=True)


def main():
    print("Hero visuals...", flush=True)
    rrna = generate_rrna(sim_duration=86400, seed=42)
    pos = positions3d()
    ppo = PPOAgent(obs_dim=6+K*9, action_dim=K+2); ppo.load(find_ckpt())
    sl = PPORoutingAdapter(ppo, n_nodes=18, sim_duration=86400.0, use_aacr=True, k_contacts=K)
    fig_constellation_3d(rrna, pos, sl, dst=16)
    fig_network_graph(rrna, sl, src=2, dst=16)
    print("Done.", flush=True)


if __name__ == '__main__':
    main()

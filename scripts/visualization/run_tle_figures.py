"""
run_tle_figures.py — Real-TLE figures for the EmRL paper (§6.13, §6.15).
========================================================================
Generates three publication figures from the real-geometry result JSONs:

  fig17_tle_finetune  — BDR on three real constellations (Iridium / Orbcomm /
                        Globalstar): encoder transfer, synthetic zero-shot
                        collapse, and in-distribution fine-tune recovery vs
                        CGR and the RUCoP ceiling.            [results/tle_eval.json]
  fig18_tle_holdout   — Leave-one-out generalization on held-out Iridium:
                        synthetic -> held-out-FT (never trained on Iridium)
                        -> in-distribution-FT, against CGR / encoder / RUCoP.
                                                              [results/tle_holdout.json]
  fig19_tle_anomaly   — Real Iridium geometry under six injected anomaly
                        scenarios; the fine-tuned policy's value shows up under
                        uncertainty, not nominal.            [results/tle_anomaly.json]

Style matches scripts/visualization/plot_results.py (serif, colorblind-safe,
PDF+PNG @ 300 dpi). Run from the repo root:

    uv run python scripts/visualization/run_tle_figures.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(".")
RESULTS = ROOT / "results"
FIG = ROOT / "deliverables" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    8,
    "legend.framealpha":  0.9,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})

# Consistent colour / hatch identity across all three figures.
#   encoder (no learning) = brown · synthetic policy = light blue
#   fine-tuned policy (ours) = blue · CGR = red · RUCoP ceiling = green
STYLE = {
    "Reach-Greedy": ("#8c564b", "",    "Reach-Greedy\n(encoder)"),
    "EmRL-PPO":     ("#aec7e8", "..",  "EmRL-PPO\n(synth, zero-shot)"),
    "EmRL-Holdout": ("#5b9bd5", "xx",  "EmRL-Holdout\n(Iridium unseen)"),
    "EmRL-TLE-FT":  ("#1f77b4", "",    "EmRL-TLE-FT\n(fine-tuned)"),
    "CGR":          ("#d62728", "//",  "CGR"),
    "RUCoP":        ("#2ca02c", "\\\\", "RUCoP\n(ceiling)"),
}


def _load(name: str) -> dict:
    p = RESULTS / name
    if not p.exists():
        raise SystemExit(f"[tle-fig] missing {p} — run the eval first.")
    return json.loads(p.read_text())


def _err(entry: dict) -> tuple[float, float]:
    """Return (lo, hi) error-bar lengths from a {mean, ci95} entry."""
    m = entry["mean"]
    lo, hi = entry.get("ci95", [m, m])
    return max(0.0, m - lo), max(0.0, hi - m)


def _save(fig, name: str):
    for ext in ("pdf", "png"):
        path = FIG / f"{name}.{ext}"
        fig.savefig(path, format=ext)
        print(f"  saved {path}")
    plt.close(fig)


# ── fig17: fine-tune recovery across three real constellations ────────────────

def fig17_finetune(data: dict):
    consts = [("iridium-NEXT", "Iridium-NEXT\n(24 sats, dense)"),
              ("orbcomm",      "Orbcomm\n(20 sats, sparse)"),
              ("globalstar",   "Globalstar\n(20 sats, sparse)")]
    agents = ["Reach-Greedy", "EmRL-PPO", "EmRL-TLE-FT", "CGR", "RUCoP"]

    x = np.arange(len(consts))
    bar_w = 0.16
    fig, ax = plt.subplots(figsize=(9, 4.6))

    for i, ag in enumerate(agents):
        means, los, his = [], [], []
        for ckey, _ in consts:
            e = data["constellations"][ckey]["summary"][ag]
            means.append(e["mean"])
            lo, hi = _err(e)
            los.append(lo); his.append(hi)
        color, hatch, label = STYLE[ag]
        off = (i - len(agents) / 2 + 0.5) * bar_w
        ax.bar(x + off, means, bar_w, color=color, hatch=hatch,
               edgecolor="black", linewidth=0.6, label=label, zorder=3)
        ax.errorbar(x + off, means, yerr=[los, his], fmt="none",
                    color="black", capsize=2.5, linewidth=0.7, zorder=4)

    # Highlight the Iridium recovery: synthetic -> fine-tuned.
    irid = data["constellations"]["iridium-NEXT"]["summary"]
    zs, ft = irid["EmRL-PPO"]["mean"], irid["EmRL-TLE-FT"]["mean"]
    ppo_off = (1 - len(agents) / 2 + 0.5) * bar_w
    ft_off  = (2 - len(agents) / 2 + 0.5) * bar_w
    ax.annotate("", xy=(0 + ft_off, ft + 0.01), xytext=(0 + ppo_off, zs + 0.01),
                arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.4,
                                connectionstyle="arc3,rad=-0.3"), zorder=6)
    ax.annotate(f"fine-tune\n+{(ft - zs):.2f} BDR", xy=(0, ft + 0.07),
                ha="center", fontsize=7.5, color="#1f77b4", fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels([lb for _, lb in consts])
    ax.set_ylabel("Bundle Delivery Ratio (BDR)")
    ax.set_ylim(0, 1.14)
    ax.set_title("Real-TLE BDR: encoder transfers, synthetic policy does not,\n"
                 "in-distribution fine-tuning recovers it on dense geometry")
    ax.legend(loc="upper center", ncol=5, fontsize=7, columnspacing=0.8,
              handletextpad=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "fig17_tle_finetune")


# ── fig18: leave-one-out generalization (held-out Iridium) ────────────────────

def fig18_holdout(data: dict):
    order = ["EmRL-PPO", "EmRL-Holdout", "EmRL-TLE-FT",
             "Reach-Greedy", "CGR", "RUCoP"]
    s = data["summary"]
    means = [s[a]["mean"] for a in order]
    errs = np.array([_err(s[a]) for a in order]).T
    colors = [STYLE[a][0] for a in order]
    hatches = [STYLE[a][1] for a in order]
    labels = [STYLE[a][2].replace("\n", " ") for a in order]

    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    bars = ax.bar(x, means, 0.62, color=colors, edgecolor="black",
                  linewidth=0.6, zorder=3)
    for b, h in zip(bars, hatches):
        b.set_hatch(h)
    ax.errorbar(x, means, yerr=errs, fmt="none", color="black",
                capsize=4, linewidth=0.8, zorder=4)

    for xi, m in zip(x, means):
        ax.annotate(f"{m:.3f}", xy=(xi, m), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8,
                    fontweight="bold")

    v = data["verdict"]
    # transfer gain (synthetic -> held-out) and generalization gap (held-out -> full)
    ax.annotate("", xy=(1, s["EmRL-Holdout"]["mean"] - 0.02),
                xytext=(0, s["EmRL-PPO"]["mean"] + 0.02),
                arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.4,
                                connectionstyle="arc3,rad=-0.25"), zorder=6)
    ax.annotate(f"+{v['transfer_gain_over_synthetic']:.3f}\n(unseen transfer)",
                xy=(0.5, 0.66), ha="center", fontsize=7.5,
                color="#1f77b4", fontweight="bold")
    ax.annotate(f"gap to in-dist FT\n= {v['generalization_gap']:+.3f}",
                xy=(2.0, s["EmRL-TLE-FT"]["mean"] + 0.06), ha="center",
                fontsize=7.5, color="#555")

    ax.axhline(s["CGR"]["mean"], color="#d62728", ls=":", lw=1.0, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("Bundle Delivery Ratio (BDR)")
    ax.set_ylim(0, 1.10)
    ax.set_title("Leave-one-out generalization — Iridium-NEXT held out of training\n"
                 "(5 seeds × 150 episodes; held-out model never saw Iridium)")
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "fig18_tle_holdout")


# ── fig19: real Iridium geometry under injected anomalies ─────────────────────

def fig19_anomaly(data: dict):
    scen = [("nominal", "Nominal"), ("random", "Random\nfailures"),
            ("correlated", "Correlated"), ("space_weather", "Space\nweather"),
            ("adversarial", "Adversarial"), ("cascading", "Cascading")]
    agents = ["Reach-Greedy", "EmRL-PPO", "EmRL-TLE-FT", "CGR", "RUCoP"]
    irid = data["constellations"]["iridium-NEXT"]

    x = np.arange(len(scen))
    bar_w = 0.16
    fig, ax = plt.subplots(figsize=(10.5, 4.8))

    for i, ag in enumerate(agents):
        means, los, his = [], [], []
        for skey, _ in scen:
            e = irid[skey][ag]
            means.append(e["mean"])
            lo, hi = _err(e)
            los.append(lo); his.append(hi)
        color, hatch, label = STYLE[ag]
        off = (i - len(agents) / 2 + 0.5) * bar_w
        ax.bar(x + off, means, bar_w, color=color, hatch=hatch,
               edgecolor="black", linewidth=0.6, label=label, zorder=3)
        ax.errorbar(x + off, means, yerr=[los, his], fmt="none",
                    color="black", capsize=2, linewidth=0.6, zorder=4)

    # Mark scenarios where the fine-tuned policy beats CGR.
    ft_off = (2 - len(agents) / 2 + 0.5) * bar_w
    for xi, (skey, _) in enumerate(scen):
        ft = irid[skey]["EmRL-TLE-FT"]["mean"]
        cgr = irid[skey]["CGR"]["mean"]
        if ft > cgr + 0.01:
            ax.annotate(f"+{ft - cgr:.2f}\nvs CGR", xy=(xi + ft_off, ft + 0.02),
                        ha="center", fontsize=6.8, color="#1f77b4",
                        fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels([lb for _, lb in scen])
    ax.set_ylabel("Bundle Delivery Ratio (BDR)")
    ax.set_ylim(0, 1.14)
    ax.set_title("Real Iridium-NEXT geometry under injected anomalies — "
                 "the learned policy's value shows up under uncertainty")
    ax.legend(loc="upper center", ncol=5, fontsize=7.5, columnspacing=0.9,
              handletextpad=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "fig19_tle_anomaly")


def main():
    print("=" * 64)
    print("EmRL — real-TLE figures (fig17 fine-tune · fig18 held-out · fig19 anomaly)")
    print("=" * 64)
    fig17_finetune(_load("tle_eval.json"))
    fig18_holdout(_load("tle_holdout.json"))
    fig19_anomaly(_load("tle_anomaly.json"))
    print(f"\nAll real-TLE figures written to {FIG}")


if __name__ == "__main__":
    main()

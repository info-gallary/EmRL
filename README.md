# EmRL: Self-Learning Contact Graph Routing for DTNs

Anomaly-aware reinforcement-learning router for Delay/Disruption-Tolerant
Networks (DTNs), built on PPO with a contact-attention backbone and
Anomaly-Aware Contact Reliability (AACR).

> **Headline result.** The destination-aware EmRL architecture **matches
> optimal Classical CGR on clean routing** (0.93 vs Dijkstra's 0.99; 0.88 vs
> 0.95 with stochastic failures) **and outperforms it under targeted hub
> jamming** — beating even idealized re-routing CGR by **+11–35%**, and
> realistic (pre-computed) CGR by far more. AACR adds +0.10–0.14 BDR.
> See `deliverables/RESULTS_FOR_MENTOR.md`.

---

## Repository layout

```
src/                         # Core package
├── env/                     # DTN environment + contact-plan generators
│   ├── dtn_env.py           #   Gymnasium env (K-configurable obs, anomaly injection,
│   │                        #   reliability-shaped reward)
│   ├── contact_plan.py      #   RRN-A / RRN-B / synthetic topology generators
│   ├── bundle.py            #   Bundle (packet) model
│   └── anomaly_loader.py    #   ESA spacecraft-anomaly telemetry loader (AACR)
├── agents/
│   ├── ppo_agent.py         #   PPO agent (clip objective, GAE)
│   ├── networks.py          #   Contact-attention actor-critic (K-agnostic)
│   ├── rl_adapter.py        #   Routing adapter wrapping a trained PPO policy
│   ├── classical_cgr.py     #   ClassicalCGR (RFC 6260), AdaptiveCGR, Spray-and-Wait
│   ├── dqn_agent.py         #   DQN baseline
│   └── oracle.py            #   Oracle (upper-bound BDR)
├── training/
│   ├── trainer.py           #   PPO training loop + curriculum
│   └── bc_warmup.py         #   Behavioral-cloning warm-up (clones ClassicalCGR)
├── evaluation/
│   ├── evaluator.py         #   Episode runner / routing env
│   └── metrics.py           #   BDR, delay, energy, 95% CIs
└── analysis/
    └── stats.py             #   Paired t-tests, Holm correction, Cohen's d

# Entry points — ALWAYS run from repo root, e.g. `uv run python scripts/...`
main.py                      # train | eval | full pipeline
scripts/
├── training/
│   ├── run_train.py             # baseline PPO curriculum (K=10)
│   ├── run_phase7_train.py      # FINAL model: wide-obs (K=16) BC + reliability-shaped PPO
│   ├── run_no_bc_train.py       # ablation: no BC warm-up
│   ├── run_no_attn_train.py     # ablation: no contact attention
│   └── run_dqn_train.py         # baseline: DQN
├── evaluation/
│   ├── run_eval_full.py             # 300-episode eval, all baselines + paired stats
│   ├── run_drastic_difference.py    # HEADLINE: bottleneck-hub jamming (CGR collapses)
│   ├── run_adversarial_targeting.py # hub-targeting severity sweep
│   └── run_behavioral_analysis.py   # action distributions + trajectories
└── visualization/
    ├── make_publication_outputs.py  # generates deliverables/ (figures + tables)
    └── plot_results.py              # legacy 16-figure generator (K=10 results)

checkpoints/final/           # canonical trained models (see below)
results/                     # evaluation JSON outputs
deliverables/                # publication figures + tables (for mentor/paper)
archive/                     # superseded experiment scripts (kept for provenance)
```

## Canonical models (`checkpoints/final/`)

| File | Description |
|------|-------------|
| `emrl_main_k16.pt`      | **Final EmRL** — wide observation (K=16), reliability-aware (BDR 0.68 @25% anomaly) |
| `emrl_bc_warmup_k16.pt` | BC warm-up weights for the K=16 model |
| `emrl_main_k10.pt`      | Earlier K=10 EmRL (BDR 0.78), used for K=10 ablations |
| `ablation_no_bc.pt`      | Ablation: trained without BC warm-up |
| `ablation_no_attn.pt`    | Ablation: trained without contact attention |
| `baseline_dqn.pt`        | DQN baseline |

## Setup

```bash
uv sync          # or: pip install -r requirements.txt
```

## Reproduce the headline result

```bash
# 1. (optional) train the final model — ~5 h on a single GPU
uv run python scripts/training/run_phase7_train.py

# 2. evaluate under adversarial bottleneck jamming
uv run python scripts/evaluation/run_drastic_difference.py     # CGR collapses, EmRL detours
uv run python scripts/evaluation/run_adversarial_targeting.py  # severity sweep

# 3. regenerate publication figures + tables
uv run python scripts/visualization/make_publication_outputs.py   # -> deliverables/
```

> Always run scripts **from the repository root** (they resolve `src/`,
> `checkpoints/`, and `results/` relative to the working directory).

## Key architecture notes

- **K-agnostic network.** The contact-attention backbone (`src/agents/networks.py`)
  infers the contact count from the observation width, so the same architecture
  trains at any K. The env (`src/env/dtn_env.py`) and routing adapter
  (`src/agents/rl_adapter.py`) take a `k_contacts` parameter (default 10).
- **Real anomaly injection.** `DTNRoutingEnv.reset(options={'anomaly_rate': r})`
  degrades a fraction of contacts per episode so the policy experiences
  reliability variance during training.
- **Reliability-shaped reward.** `reliability_shaping * (reliability - 0.5)` on
  contact commit teaches reliability-aware (AACR) routing.

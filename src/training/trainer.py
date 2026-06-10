"""
PPO Trainer for EmRL DTN routing research.

Training curriculum (total 1 M steps):
  Phase 1  [0 – 300 k]:      Synthetic topologies, no anomalies
  Phase 2  [300 k – 600 k]:  Mix synthetic + RRN-A/B, 10 % anomaly rate
  Phase 3  [600 k – 1 M]:    Full RRN-A/B, 30 % anomaly rate

Logging: results/training_curves.csv
Columns: update, timestep, mean_reward, mean_bdr, mean_delay, mean_energy,
          policy_loss, value_loss, entropy
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..agents.ppo_agent import PPOAgent, RolloutBuffer

# ── Curriculum config ─────────────────────────────────────────────────────────

@dataclass
class CurriculumPhase:
    name:         str
    start_step:   int
    end_step:     int
    anomaly_rate: float
    use_rrn:      bool


CURRICULUM = [
    CurriculumPhase("synthetic",    0,        300_000, 0.00, False),
    CurriculumPhase("mixed",        300_000,  600_000, 0.10, True),
    CurriculumPhase("full_rrn",     600_000, 1_000_000, 0.30, True),
]


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class EvalMetrics:
    mean_reward: float = 0.0
    mean_bdr:    float = 0.0      # Bundle Delivery Ratio
    mean_delay:  float = 0.0
    mean_energy: float = 0.0
    std_reward:  float = 0.0
    n_episodes:  int   = 0


@dataclass
class TrainingResult:
    total_timesteps:  int
    total_updates:    int
    best_bdr:         float
    best_checkpoint:  str
    eval_history:     list[EvalMetrics] = field(default_factory=list)
    training_log:     list[dict]        = field(default_factory=list)


# ── Trainer ───────────────────────────────────────────────────────────────────

class PPOTrainer:
    """
    Orchestrates PPO training on a DTN routing environment.

    The environment is expected to expose a Gymnasium-compatible interface:
      - obs, info     = env.reset()
      - obs, rew, terminated, truncated, info = env.step(action)
      - info must contain: 'action_mask' (np.ndarray bool, shape (action_dim,))
      - info may contain:  'bdr', 'delay', 'energy' for eval metrics

    Anomaly injection and topology switching are controlled by passing
    `anomaly_rate` and `use_rrn` keyword arguments to env.reset().
    If the environment does not accept these, subclass and override
    _reset_env() accordingly.
    """

    def __init__(
        self,
        agent:      PPOAgent,
        obs_dim:    int = 76,
        action_dim: int = 12,
    ) -> None:
        self.agent      = agent
        self.obs_dim    = obs_dim
        self.action_dim = action_dim

    # ------------------------------------------------------------------
    def train(
        self,
        env_factory:       Callable,
        total_timesteps:   int  = 1_000_000,
        rollout_steps:     int  = 2048,
        eval_interval:     int  = 10,
        checkpoint_dir:    str  = "checkpoints/",
        log_dir:           str  = "logs/",
        results_dir:       str  = "results/",
        n_eval_episodes:   int  = 100,
        early_stop_patience: int = 30,
        seed:              int  = 42,
    ) -> TrainingResult:
        """
        Full training loop.

        Args:
            env_factory:       Callable that returns a fresh env instance.
            total_timesteps:   Total environment steps.
            rollout_steps:     Steps collected per PPO update.
            eval_interval:     PPO updates between evaluation rounds.
            checkpoint_dir:    Directory for model checkpoints.
            log_dir:           Directory for tensorboard (if available).
            results_dir:       Directory for CSV training log.
            n_eval_episodes:   Episodes per evaluation round.
            early_stop_patience: Stop if BDR does not improve for this many evals.
            seed:              Random seed.

        Returns:
            TrainingResult with summary statistics.
        """
        np.random.seed(seed)

        ckpt_dir    = Path(checkpoint_dir)
        results_dir = Path(results_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        csv_path    = results_dir / "training_curves.csv"
        csv_columns = [
            "update", "timestep", "mean_reward", "mean_bdr", "mean_delay",
            "mean_energy", "policy_loss", "value_loss", "entropy",
        ]

        csv_file    = open(csv_path, "w", newline="")
        csv_writer  = csv.DictWriter(csv_file, fieldnames=csv_columns)
        csv_writer.writeheader()

        # Optional TensorBoard
        tb_writer = self._try_tensorboard(log_dir)

        env = env_factory()
        buffer = RolloutBuffer(
            capacity=rollout_steps,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
        )

        result = TrainingResult(
            total_timesteps=total_timesteps,
            total_updates=0,
            best_bdr=0.0,
            best_checkpoint="",
        )

        timestep       = 0
        update_count   = 0
        no_improve_cnt = 0
        episode_rewards: list[float] = []
        episode_reward  = 0.0

        phase    = self._get_phase(timestep)
        obs, info = self._reset_env(env, phase)
        action_mask = info.get("action_mask", np.ones(self.action_dim, dtype=bool))

        start_time = time.time()

        while timestep < total_timesteps:
            buffer.reset()

            # ── Collect rollout ────────────────────────────────────────
            for _ in range(rollout_steps):
                action, log_prob, value = self.agent.select_action(obs, action_mask)

                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                buffer.store(
                    obs         = obs,
                    action      = action,
                    log_prob    = log_prob,
                    reward      = reward,
                    value       = value,
                    done        = done,
                    action_mask = action_mask,
                )

                episode_reward += reward
                timestep       += 1

                if done:
                    episode_rewards.append(episode_reward)
                    episode_reward = 0.0
                    new_phase = self._get_phase(timestep)
                    if new_phase.name != phase.name:
                        phase = new_phase
                    obs, info = self._reset_env(env, phase)
                else:
                    obs = next_obs

                action_mask = info.get("action_mask", np.ones(self.action_dim, dtype=bool))

                if timestep >= total_timesteps:
                    break

            # ── Compute GAE ────────────────────────────────────────────
            _, _, last_value = self.agent.select_action(obs, action_mask)
            if buffer.dones[-1]:
                last_value = 0.0
            buffer.compute_gae(last_value, self.agent.gamma, self.agent.gae_lambda)

            # ── PPO update ─────────────────────────────────────────────
            train_metrics = self.agent.update(buffer)
            update_count += 1

            mean_reward = float(np.mean(episode_rewards[-20:])) if episode_rewards else 0.0
            elapsed     = time.time() - start_time

            print(
                f"[{timestep:>8d}/{total_timesteps}] update={update_count:4d} "
                f"rew={mean_reward:.3f} "
                f"pi_loss={train_metrics['policy_loss']:+.4f} "
                f"v_loss={train_metrics['value_loss']:.4f} "
                f"ent={train_metrics['entropy']:.4f} "
                f"phase={phase.name} "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )

            # ── Evaluation ─────────────────────────────────────────────
            eval_metrics = EvalMetrics()
            if update_count % eval_interval == 0:
                eval_env = env_factory()
                eval_metrics = self.evaluate(eval_env, n_eval_episodes, phase)
                eval_env.close()
                result.eval_history.append(eval_metrics)

                if tb_writer is not None:
                    tb_writer.add_scalar("eval/bdr",         eval_metrics.mean_bdr,    timestep)
                    tb_writer.add_scalar("eval/delay",        eval_metrics.mean_delay,  timestep)
                    tb_writer.add_scalar("eval/mean_reward",  eval_metrics.mean_reward, timestep)

                if eval_metrics.mean_bdr > result.best_bdr:
                    result.best_bdr = eval_metrics.mean_bdr
                    ckpt_path = ckpt_dir / f"best_model_bdr{result.best_bdr:.4f}.pt"
                    self.agent.save(ckpt_path)
                    result.best_checkpoint = str(ckpt_path)
                    no_improve_cnt = 0
                    print(f"  New best BDR={result.best_bdr:.4f} -> saved {ckpt_path}", flush=True)
                else:
                    no_improve_cnt += 1

                if no_improve_cnt >= early_stop_patience:
                    print(f"Early stopping: BDR has not improved for {no_improve_cnt} evaluations.", flush=True)
                    break

            # ── Periodic checkpoint ────────────────────────────────────
            if update_count % 50 == 0:
                self.agent.save(ckpt_dir / f"ckpt_step{timestep}.pt")

            # ── CSV logging ────────────────────────────────────────────
            row = {
                "update":       update_count,
                "timestep":     timestep,
                "mean_reward":  round(mean_reward, 5),
                "mean_bdr":     round(eval_metrics.mean_bdr,    5),
                "mean_delay":   round(eval_metrics.mean_delay,  5),
                "mean_energy":  round(eval_metrics.mean_energy, 5),
                "policy_loss":  round(train_metrics["policy_loss"], 6),
                "value_loss":   round(train_metrics["value_loss"],  6),
                "entropy":      round(train_metrics["entropy"],     6),
            }
            csv_writer.writerow(row)
            csv_file.flush()
            result.training_log.append(row)

            if tb_writer is not None:
                tb_writer.add_scalar("train/policy_loss", train_metrics["policy_loss"], timestep)
                tb_writer.add_scalar("train/value_loss",  train_metrics["value_loss"],  timestep)
                tb_writer.add_scalar("train/entropy",     train_metrics["entropy"],     timestep)
                tb_writer.add_scalar("train/mean_reward", mean_reward,                  timestep)

        # ── Finalise ───────────────────────────────────────────────────
        csv_file.close()
        env.close()
        if tb_writer is not None:
            tb_writer.close()

        result.total_updates = update_count
        self.agent.save(ckpt_dir / "final_model.pt")
        print(f"\nTraining complete. Best BDR={result.best_bdr:.4f}  "
              f"checkpoint={result.best_checkpoint}")
        return result

    # ------------------------------------------------------------------
    def evaluate(
        self,
        env,
        n_episodes: int = 100,
        phase: Optional[CurriculumPhase] = None,
    ) -> EvalMetrics:
        """
        Run `n_episodes` deterministic rollouts and aggregate metrics.

        Expects info dict to optionally contain 'bdr', 'delay', 'energy'.
        """
        rewards, bdrs, delays, energies = [], [], [], []

        if phase is None:
            phase = CURRICULUM[-1]   # default: hardest phase

        for _ in range(n_episodes):
            obs, info = self._reset_env(env, phase)
            action_mask = info.get("action_mask", np.ones(self.action_dim, dtype=bool))
            ep_reward   = 0.0
            ep_done     = False

            while not ep_done:
                action, _, _ = self.agent.select_action(obs, action_mask, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_done     = terminated or truncated
                ep_reward  += reward
                action_mask = info.get("action_mask", np.ones(self.action_dim, dtype=bool))

            rewards.append(ep_reward)
            bdrs.append(info.get("bdr",    0.0))
            delays.append(info.get("delay", 0.0))
            energies.append(info.get("energy", 0.0))

        return EvalMetrics(
            mean_reward = float(np.mean(rewards)),
            mean_bdr    = float(np.mean(bdrs)),
            mean_delay  = float(np.mean(delays)),
            mean_energy = float(np.mean(energies)),
            std_reward  = float(np.std(rewards)),
            n_episodes  = n_episodes,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _get_phase(timestep: int) -> CurriculumPhase:
        for phase in CURRICULUM:
            if phase.start_step <= timestep < phase.end_step:
                return phase
        return CURRICULUM[-1]

    @staticmethod
    def _reset_env(env, phase: CurriculumPhase):
        """
        Reset environment with curriculum kwargs.

        If the environment does not accept these kwargs, override this method.
        """
        try:
            obs, info = env.reset(
                options={
                    "anomaly_rate": phase.anomaly_rate,
                    "use_rrn":      phase.use_rrn,
                }
            )
        except TypeError:
            obs, info = env.reset()
        return obs, info

    @staticmethod
    def _try_tensorboard(log_dir: str):
        return None  # disabled — TF/oneDNN crashes on this machine

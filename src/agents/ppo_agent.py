"""
PPO Agent with Generalized Advantage Estimation for DTN routing.

Implements:
  - RolloutBuffer: trajectory storage + GAE computation + minibatch iteration
  - PPOAgent:      action selection, PPO update, checkpoint I/O
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam

from .networks import CAPPONetwork

# ── Defaults ──────────────────────────────────────────────────────────────────
OBS_DIM: int = 76
ACTION_DIM: int = 12   # K + 2


# ── Rollout buffer ────────────────────────────────────────────────────────────

@dataclass
class RolloutBuffer:
    """
    Fixed-size circular buffer for on-policy PPO rollouts.

    All tensors live on CPU during collection; moved to device during update.
    """

    capacity: int
    obs_dim: int = OBS_DIM
    action_dim: int = ACTION_DIM

    # Storage (populated by store())
    obs:          np.ndarray = field(init=False)
    actions:      np.ndarray = field(init=False)
    log_probs:    np.ndarray = field(init=False)
    rewards:      np.ndarray = field(init=False)
    values:       np.ndarray = field(init=False)
    dones:        np.ndarray = field(init=False)
    action_masks: np.ndarray = field(init=False)

    # Computed by compute_gae()
    advantages:   np.ndarray = field(init=False)
    returns:      np.ndarray = field(init=False)

    _ptr: int = field(default=0, init=False)
    _full: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        n = self.capacity
        self.obs          = np.zeros((n, self.obs_dim),     dtype=np.float32)
        self.actions      = np.zeros(n,                     dtype=np.int64)
        self.log_probs    = np.zeros(n,                     dtype=np.float32)
        self.rewards      = np.zeros(n,                     dtype=np.float32)
        self.values       = np.zeros(n,                     dtype=np.float32)
        self.dones        = np.zeros(n,                     dtype=np.float32)
        self.action_masks = np.ones((n, self.action_dim),   dtype=bool)
        self.advantages   = np.zeros(n,                     dtype=np.float32)
        self.returns      = np.zeros(n,                     dtype=np.float32)

    # ------------------------------------------------------------------
    def store(
        self,
        obs: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
        action_mask: np.ndarray,
    ) -> None:
        i = self._ptr
        self.obs[i]          = obs
        self.actions[i]      = action
        self.log_probs[i]    = log_prob
        self.rewards[i]      = reward
        self.values[i]       = value
        self.dones[i]        = float(done)
        self.action_masks[i] = action_mask
        self._ptr += 1
        if self._ptr >= self.capacity:
            self._full = True

    def is_full(self) -> bool:
        return self._full

    def reset(self) -> None:
        self._ptr  = 0
        self._full = False

    # ------------------------------------------------------------------
    def compute_gae(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """
        Compute GAE advantages and discounted returns in-place.

        Args:
            last_value: V(s_{T+1}) from critic (0 if terminal).
        """
        n = self.capacity
        gae = 0.0
        for t in reversed(range(n)):
            next_value = last_value if t == n - 1 else self.values[t + 1]
            next_done  = 0.0        if t == n - 1 else self.dones[t + 1]
            delta = self.rewards[t] + gamma * next_value * (1.0 - next_done) - self.values[t]
            gae   = delta + gamma * gae_lambda * (1.0 - self.dones[t]) * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values

        # Normalize advantages
        adv_mean = self.advantages.mean()
        adv_std  = self.advantages.std() + 1e-8
        self.advantages = (self.advantages - adv_mean) / adv_std

    # ------------------------------------------------------------------
    def get_batches(
        self, batch_size: int = 64, device: torch.device = torch.device("cpu")
    ) -> Iterator[dict[str, Tensor]]:
        """Yield randomized minibatches as dicts of Tensors on `device`."""
        n       = self.capacity
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            idx = indices[start : start + batch_size]
            yield {
                "obs":          torch.as_tensor(self.obs[idx],          device=device),
                "actions":      torch.as_tensor(self.actions[idx],      device=device),
                "old_log_probs":torch.as_tensor(self.log_probs[idx],    device=device),
                "old_values":   torch.as_tensor(self.values[idx],       device=device),
                "advantages":   torch.as_tensor(self.advantages[idx],   device=device),
                "returns":      torch.as_tensor(self.returns[idx],      device=device),
                "action_masks": torch.as_tensor(self.action_masks[idx], device=device),
            }


# ── PPO Agent ─────────────────────────────────────────────────────────────────

class PPOAgent:
    """
    Proximal Policy Optimization agent for DTN routing.

    Implements the clipped surrogate objective with GAE.
    """

    def __init__(
        self,
        obs_dim:       int   = OBS_DIM,
        action_dim:    int   = ACTION_DIM,
        d_model:       int   = 128,
        n_heads:       int   = 4,
        n_attn_layers: int   = 2,
        dropout:       float = 0.1,
        lr:            float = 3e-4,
        clip_epsilon:  float = 0.2,
        gamma:         float = 0.99,
        gae_lambda:    float = 0.95,
        entropy_coef:  float = 0.01,
        value_coef:    float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs:      int   = 10,
        batch_size:    int   = 64,
        value_clip:    bool  = True,
        device:        str   = "auto",
    ) -> None:

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.network = CAPPONetwork(d_model, n_heads, n_attn_layers, dropout).to(self.device)
        self.optimizer = Adam(self.network.parameters(), lr=lr, eps=1e-5)

        # Hyper-parameters
        self.clip_epsilon  = clip_epsilon
        self.gamma         = gamma
        self.gae_lambda    = gae_lambda
        self.entropy_coef  = entropy_coef
        self.value_coef    = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs      = n_epochs
        self.batch_size    = batch_size
        self.value_clip    = value_clip
        self.obs_dim       = obs_dim
        self.action_dim    = action_dim

    # ------------------------------------------------------------------
    @torch.no_grad()
    def select_action(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        """
        Args:
            obs:         (obs_dim,) float32
            action_mask: (action_dim,) bool — True = valid
            deterministic: if True, return argmax action

        Returns:
            action:   int
            log_prob: float
            value:    float
        """
        obs_t    = torch.as_tensor(obs,         dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t   = torch.as_tensor(action_mask, dtype=torch.bool,    device=self.device).unsqueeze(0)

        dist, value = self.network(obs_t, mask_t)

        action   = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)

        return int(action.item()), float(log_prob.item()), float(value.item())

    # ------------------------------------------------------------------
    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        """
        Run `n_epochs` of PPO on the stored rollout.

        Returns:
            Dictionary of mean training metrics across all mini-batch updates.
        """
        metrics: dict[str, list[float]] = {
            "policy_loss": [], "value_loss": [], "entropy": [],
            "approx_kl": [], "clip_fraction": [],
        }

        for _ in range(self.n_epochs):
            for batch in buffer.get_batches(self.batch_size, self.device):
                obs           = batch["obs"]
                actions       = batch["actions"]
                old_log_probs = batch["old_log_probs"]
                old_values    = batch["old_values"]
                advantages    = batch["advantages"]
                returns       = batch["returns"]
                action_masks  = batch["action_masks"]

                dist, values = self.network(obs, action_masks)
                new_log_probs = dist.log_prob(actions)
                entropy       = dist.entropy().mean()

                # Policy loss (clipped surrogate)
                ratio       = torch.exp(new_log_probs - old_log_probs)
                surr1       = ratio * advantages
                surr2       = torch.clamp(ratio, 1.0 - self.clip_epsilon,
                                                 1.0 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (optionally clipped)
                if self.value_clip:
                    values_clipped = old_values + torch.clamp(
                        values - old_values, -self.clip_epsilon, self.clip_epsilon
                    )
                    vf_loss1   = (values - returns).pow(2)
                    vf_loss2   = (values_clipped - returns).pow(2)
                    value_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()
                else:
                    value_loss = 0.5 * (values - returns).pow(2).mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Diagnostics
                with torch.no_grad():
                    approx_kl     = ((ratio - 1.0) - (new_log_probs - old_log_probs)).mean()
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()

                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy"].append(entropy.item())
                metrics["approx_kl"].append(approx_kl.item())
                metrics["clip_fraction"].append(clip_fraction.item())

        return {k: float(np.mean(v)) for k, v in metrics.items()}

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "network_state":    self.network.state_dict(),
                "optimizer_state":  self.optimizer.state_dict(),
                "hyperparams": {
                    "clip_epsilon":  self.clip_epsilon,
                    "gamma":         self.gamma,
                    "gae_lambda":    self.gae_lambda,
                    "entropy_coef":  self.entropy_coef,
                    "value_coef":    self.value_coef,
                    "max_grad_norm": self.max_grad_norm,
                    "n_epochs":      self.n_epochs,
                    "batch_size":    self.batch_size,
                },
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["network_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])

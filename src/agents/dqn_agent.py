"""
DQN Agent for DTN routing — vanilla RL baseline and RUCoP reproduction.

Implements:
  - QNetwork:      MLP Q-function with masked action selection
  - ReplayBuffer:  fixed-size circular replay buffer
  - DQNAgent:      epsilon-greedy action selection, target network, training step

Two configurations supported:
  - 'dqn_basic'   : 2-layer MLP Q-network (matches RUCoP-style simplicity)
  - 'dqn_attn'    : Uses the ContactAttentionBackbone (for architecture-fair RL baseline)

The default config 'dqn_basic' is intentionally minimal: a flat MLP over the
76-dim observation, no attention. This is the "vanilla RL" comparison that
answers the question 'is PPO+attention better than DQN+MLP?'.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from .networks import ContactBackbone, D_MODEL


OBS_DIM = 76
ACTION_DIM = 12


# ── Q-Networks ────────────────────────────────────────────────────────────────


class FlatQNetwork(nn.Module):
    """
    Simple MLP Q-network. No attention, no structure-aware encoding.
    This is the RUCoP-equivalent baseline (flat representation).
    """

    def __init__(self, obs_dim: int = OBS_DIM, action_dim: int = ACTION_DIM,
                 hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AttnQNetwork(nn.Module):
    """
    Q-network using the same contact-attention backbone as EmRL.
    For comparison: 'is PPO better than DQN given the same architecture?'
    """

    def __init__(self, action_dim: int = ACTION_DIM, d_model: int = D_MODEL,
                 n_heads: int = 4, n_layers: int = 2) -> None:
        super().__init__()
        self.backbone = ContactBackbone(d_model, n_heads, n_layers, dropout=0.1)
        self.contact_head = nn.Linear(d_model, 1)   # per-contact Q value
        self.extra_head = nn.Linear(d_model, 2)     # hold + drop Q values

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        global_emb, contact_embs = self.backbone(obs)
        # Per-contact Q: (B, K, 1) → (B, K)
        q_contacts = self.contact_head(contact_embs).squeeze(-1)
        # Hold/drop Q: (B, 2)
        q_extra = self.extra_head(global_emb)
        return torch.cat([q_contacts, q_extra], dim=-1)   # (B, K+2)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Replay Buffer ─────────────────────────────────────────────────────────────


@dataclass
class ReplayBuffer:
    capacity: int = 100_000
    obs_dim: int = OBS_DIM
    action_dim: int = ACTION_DIM

    _obs:          np.ndarray = field(init=False)
    _actions:      np.ndarray = field(init=False)
    _rewards:      np.ndarray = field(init=False)
    _next_obs:     np.ndarray = field(init=False)
    _dones:        np.ndarray = field(init=False)
    _masks:        np.ndarray = field(init=False)
    _next_masks:   np.ndarray = field(init=False)
    _ptr:          int = 0
    _size:         int = 0

    def __post_init__(self):
        n = self.capacity
        self._obs        = np.zeros((n, self.obs_dim), dtype=np.float32)
        self._actions    = np.zeros(n, dtype=np.int64)
        self._rewards    = np.zeros(n, dtype=np.float32)
        self._next_obs   = np.zeros((n, self.obs_dim), dtype=np.float32)
        self._dones      = np.zeros(n, dtype=np.float32)
        self._masks      = np.ones((n, self.action_dim), dtype=bool)
        self._next_masks = np.ones((n, self.action_dim), dtype=bool)

    def __len__(self) -> int:
        return self._size

    def add(self, obs, action, reward, next_obs, done, mask, next_mask):
        i = self._ptr
        self._obs[i]        = obs
        self._actions[i]    = action
        self._rewards[i]    = reward
        self._next_obs[i]   = next_obs
        self._dones[i]      = float(done)
        self._masks[i]      = mask
        self._next_masks[i] = next_mask
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(0, self._size, size=batch_size)
        return {
            "obs":        self._obs[idx],
            "actions":    self._actions[idx],
            "rewards":    self._rewards[idx],
            "next_obs":   self._next_obs[idx],
            "dones":      self._dones[idx],
            "masks":      self._masks[idx],
            "next_masks": self._next_masks[idx],
        }


# ── DQN Agent ─────────────────────────────────────────────────────────────────


class DQNAgent:
    """
    Vanilla DQN with target network and ε-greedy exploration.

    Supports two architectures:
      arch='flat' : FlatQNetwork (MLP)         — RUCoP-style baseline
      arch='attn' : AttnQNetwork (with backbone) — architecture-fair RL baseline
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        action_dim: int = ACTION_DIM,
        arch: str = "flat",
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 200_000,
        target_update_interval: int = 1000,
        replay_capacity: int = 100_000,
        batch_size: int = 128,
        min_buffer_size: int = 5000,
        device: str = "auto",
    ) -> None:
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if arch == "flat":
            self.network = FlatQNetwork(obs_dim, action_dim).to(self.device)
            self.target  = FlatQNetwork(obs_dim, action_dim).to(self.device)
        elif arch == "attn":
            self.network = AttnQNetwork(action_dim).to(self.device)
            self.target  = AttnQNetwork(action_dim).to(self.device)
        else:
            raise ValueError(f"Unknown arch: {arch!r}")

        self.target.load_state_dict(self.network.state_dict())
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad = False

        self.optimizer = Adam(self.network.parameters(), lr=lr)

        self.arch = arch
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.target_update_interval = target_update_interval
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.buffer = ReplayBuffer(replay_capacity, obs_dim, action_dim)
        self._train_step = 0
        self._env_step = 0
        self._rng = np.random.default_rng(42)

    # ------------------------------------------------------------------
    def epsilon(self) -> float:
        frac = min(1.0, self._env_step / self.epsilon_decay_steps)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, action_mask: np.ndarray,
                      deterministic: bool = False) -> int:
        """ε-greedy action selection with action masking."""
        self._env_step += 1
        valid_actions = np.where(action_mask)[0]
        if len(valid_actions) == 0:
            return 0

        if not deterministic and self._rng.random() < self.epsilon():
            return int(self._rng.choice(valid_actions))

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.network(obs_t).cpu().numpy()[0]   # (action_dim,)
        # Mask invalid actions
        q_masked = np.where(action_mask, q, -1e9)
        return int(np.argmax(q_masked))

    def store(self, obs, action, reward, next_obs, done, mask, next_mask):
        self.buffer.add(obs, action, reward, next_obs, done, mask, next_mask)

    def update(self) -> dict:
        if len(self.buffer) < self.min_buffer_size:
            return {"q_loss": 0.0, "mean_q": 0.0, "epsilon": self.epsilon()}

        batch = self.buffer.sample(self.batch_size, self._rng)
        obs        = torch.as_tensor(batch["obs"],        dtype=torch.float32, device=self.device)
        actions    = torch.as_tensor(batch["actions"],    dtype=torch.int64,   device=self.device)
        rewards    = torch.as_tensor(batch["rewards"],    dtype=torch.float32, device=self.device)
        next_obs   = torch.as_tensor(batch["next_obs"],   dtype=torch.float32, device=self.device)
        dones      = torch.as_tensor(batch["dones"],      dtype=torch.float32, device=self.device)
        next_masks = torch.as_tensor(batch["next_masks"], dtype=torch.bool,    device=self.device)

        # Current Q(s, a)
        q_values = self.network(obs)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target: r + γ * max_a' Q_target(s', a') with masking
        with torch.no_grad():
            q_next = self.target(next_obs)
            q_next_masked = q_next.masked_fill(~next_masks, -1e9)
            q_next_max = q_next_masked.max(dim=1).values
            targets = rewards + self.gamma * q_next_max * (1.0 - dones)

        loss = F.smooth_l1_loss(q_sa, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()

        self._train_step += 1
        if self._train_step % self.target_update_interval == 0:
            self.target.load_state_dict(self.network.state_dict())

        return {
            "q_loss": float(loss.item()),
            "mean_q": float(q_values.mean().item()),
            "epsilon": float(self.epsilon()),
        }

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "network_state": self.network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "arch": self.arch,
            "_env_step": self._env_step,
            "_train_step": self._train_step,
        }, path)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(ckpt["network_state"])
        self.target.load_state_dict(ckpt["network_state"])
        if "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])


# ── Routing adapter ───────────────────────────────────────────────────────────


class DQNRoutingAdapter:
    """
    Wraps a DQNAgent in a .route() interface compatible with the Evaluator.
    Mirrors PPORoutingAdapter design.
    """

    def __init__(self, dqn_agent: DQNAgent, n_nodes: int = 18,
                 sim_duration: float = 86400.0, use_aacr: bool = True) -> None:
        self.agent = dqn_agent
        self.n_nodes = n_nodes
        self.sim_duration = sim_duration
        self.use_aacr = use_aacr

    def route(self, bundle, contact_plan, current_node: int, current_time: float):
        from src.agents.rl_adapter import PPORoutingAdapter
        # Reuse PPORoutingAdapter's observation builder
        # Hack: create a temporary one and call its internal method
        if not hasattr(self, '_obs_builder'):
            class _Dummy: pass
            d = _Dummy()
            d.network = None
            self._obs_builder = PPORoutingAdapter(
                d, n_nodes=self.n_nodes,
                sim_duration=self.sim_duration,
                use_aacr=self.use_aacr,
            )
        obs, action_mask, contacts = self._obs_builder._build_obs(
            bundle, contact_plan, current_node, current_time
        )
        action = self.agent.select_action(obs, action_mask, deterministic=True)
        if action < 10 and action < len(contacts):
            return contacts[action]
        return None  # hold/drop → no contact

    def reset(self):
        pass

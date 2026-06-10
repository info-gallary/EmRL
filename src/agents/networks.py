"""
CA-PPO Networks: Contact-Aware PPO architecture for DTN routing.

Observation layout (76-dim):
  [0-5]   global state: node, dest, time, queue_util, hops, ttl_remaining
  [6-75]  K=10 contacts × 7 features: active, neighbor, start, duration,
                                        rate, energy, reliability

Action space: Discrete(K+2=12)
  0-9  : forward via contact k
  10   : hold / wait
  11   : drop (emergency)
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Categorical

# ── Hyper-parameters ──────────────────────────────────────────────────────────
D_MODEL: int = 128
N_HEADS: int = 4
N_ATTN_LAYERS: int = 2
K_CONTACTS: int = 10
CONTACT_FEATURES: int = 9   # +dest-distance +delivery-potential (contact-graph encoder)
GLOBAL_FEATURES: int = 6
DROPOUT: float = 0.1


# ── Building blocks ───────────────────────────────────────────────────────────

class ContactEncoder(nn.Module):
    """Per-contact MLP: (batch, K, 7) → (batch, K, d_model)."""

    def __init__(self, in_features: int = CONTACT_FEATURES, d_model: int = D_MODEL) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, K, 7)
        return self.net(x)  # (B, K, d_model)


class ContactAttentionBlock(nn.Module):
    """Transformer encoder block over K contacts."""

    def __init__(self, d_model: int = D_MODEL, n_heads: int = N_HEADS, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, K, d_model)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x  # (B, K, d_model)


class GlobalEncoder(nn.Module):
    """Encodes the 6-dim global state into a d_model embedding."""

    def __init__(self, in_features: int = GLOBAL_FEATURES, d_model: int = D_MODEL) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, 6)
        return self.net(x)  # (B, d_model)


# ── Shared contact backbone ───────────────────────────────────────────────────

class ContactBackbone(nn.Module):
    """Shared sub-network used by both actor and critic."""

    def __init__(
        self,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_ATTN_LAYERS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.global_enc = GlobalEncoder(GLOBAL_FEATURES, d_model)
        self.contact_enc = ContactEncoder(CONTACT_FEATURES, d_model)
        self.attn_layers = nn.ModuleList(
            [ContactAttentionBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )

    def forward(self, obs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            obs: (B, 76)  raw observation

        Returns:
            global_emb:   (B, d_model)
            contact_embs: (B, K, d_model)
        """
        global_feat = obs[:, :GLOBAL_FEATURES]                          # (B, 6)
        # K-agnostic: infer the number of contacts from the observation width.
        # All downstream layers (per-contact MLP, attention, pointer) are
        # length-independent, so the same weights work for any K.
        contact_feat = obs[:, GLOBAL_FEATURES:].reshape(
            obs.shape[0], -1, CONTACT_FEATURES
        )                                                                # (B, K, 7)

        global_emb = self.global_enc(global_feat)                       # (B, d_model)
        contact_embs = self.contact_enc(contact_feat)                   # (B, K, d_model)

        for layer in self.attn_layers:
            contact_embs = layer(contact_embs)                          # (B, K, d_model)

        return global_emb, contact_embs


# ── Actor ─────────────────────────────────────────────────────────────────────

class CAPPOActor(nn.Module):
    """
    Contact-Aware Actor Network.

    Pointer attention over contact embeddings produces K logits.
    A small MLP on global_emb produces 2 extra logits (hold, drop).
    Action masking zeroes out invalid choices.
    """

    def __init__(self, backbone: ContactBackbone, d_model: int = D_MODEL) -> None:
        super().__init__()
        self.backbone = backbone
        self.scale = math.sqrt(d_model)
        self.extra_head = nn.Linear(d_model, 2)   # hold + drop logits

    def forward(self, obs: Tensor, action_mask: Tensor) -> Categorical:
        """
        Args:
            obs:         (B, 76)
            action_mask: (B, K+2)  True = action is valid

        Returns:
            Categorical distribution over K+2 actions
        """
        global_emb, contact_embs = self.backbone(obs)   # (B, d), (B, K, d)

        # Pointer scores: (B, K)
        g = global_emb.unsqueeze(1)                      # (B, 1, d)
        contact_logits = (g @ contact_embs.transpose(1, 2)).squeeze(1) / self.scale  # (B, K)

        # Hold / drop logits: (B, 2)
        extra_logits = self.extra_head(global_emb)       # (B, 2)

        logits = torch.cat([contact_logits, extra_logits], dim=-1)  # (B, K+2)

        # Mask invalid actions
        logits = logits.masked_fill(~action_mask, -1e9)

        return Categorical(logits=logits)


# ── Critic ────────────────────────────────────────────────────────────────────

class CAPPOCritic(nn.Module):
    """
    Contact-Aware Critic Network.

    Aggregates global_emb and mean-pooled contact_embs, then regresses value.
    """

    def __init__(self, backbone: ContactBackbone, d_model: int = D_MODEL) -> None:
        super().__init__()
        self.backbone = backbone
        self.value_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, obs: Tensor) -> Tensor:
        """
        Args:
            obs: (B, 76)

        Returns:
            value: (B,)
        """
        global_emb, contact_embs = self.backbone(obs)          # (B, d), (B, K, d)
        pooled = contact_embs.mean(dim=1)                       # (B, d)
        combined = torch.cat([global_emb, pooled], dim=-1)      # (B, 2d)
        return self.value_head(combined).squeeze(-1)            # (B,)


# ── Unified network ───────────────────────────────────────────────────────────

class CAPPONetwork(nn.Module):
    """
    Wraps actor and critic with a shared contact backbone.

    Parameter count target: ~150 k (appropriate for embedded-systems paper).
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_ATTN_LAYERS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.backbone = ContactBackbone(d_model, n_heads, n_layers, dropout)
        self.actor = CAPPOActor(self.backbone, d_model)
        self.critic = CAPPOCritic(self.backbone, d_model)

    def forward(self, obs: Tensor, action_mask: Tensor) -> Tuple[Categorical, Tensor]:
        """
        Returns:
            dist:  Categorical distribution (masked) over K+2 actions
            value: (B,) state-value estimate
        """
        dist = self.actor(obs, action_mask)
        value = self.critic(obs)
        return dist, value

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

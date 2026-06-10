"""
rl_adapter.py
-------------
Adapts the Gymnasium-style PPOAgent for use in the store-carry-forward
_RoutingEnv evaluator, which expects agent.route(bundle, contact_plan,
current_node, current_time) -> Optional[Contact].
"""

from __future__ import annotations

import numpy as np
from typing import Optional

K_CONTACTS: int = 10
OBS_BASE_DIMS: int = 6
OBS_CONTACT_DIMS: int = 9   # +destination-distance +delivery-potential (graph encoder)
OBS_DIM: int = OBS_BASE_DIMS + K_CONTACTS * OBS_CONTACT_DIMS

MAX_DATA_RATE: float = 2_000_000.0
MAX_ENERGY_COST: float = 3.0
MAX_CONTACT_DURATION: float = 7200.0
MAX_QUEUE_UTIL: float = 1.0
MAX_HOPS: int = 20
MAX_TTL: float = 3600.0 * 24


class PPORoutingAdapter:
    """
    Wraps a trained PPOAgent so it can be used by the Evaluator's
    store-carry-forward simulator.

    Builds a DTNRoutingEnv-compatible 76-dim observation from the
    raw routing state and feeds it through the PPO policy.
    """

    def __init__(self, ppo_agent, n_nodes: int = 18, sim_duration: float = 86400.0,
                 use_aacr: bool = True, k_contacts: int = K_CONTACTS):
        self.ppo_agent = ppo_agent
        self.n_nodes = n_nodes
        self.sim_duration = sim_duration
        self.use_aacr = use_aacr  # when False, reliability obs feature is constant 1.0
        self.K = int(k_contacts)  # must match the env the agent was trained on

    # Hold step must match DTNRoutingEnv.TIME_STEP_HOLD (300 s)
    HOLD_STEP: float = 300.0

    def route(self, bundle, contact_plan, current_node: int, current_time: float) -> Optional[object]:
        """
        Route the bundle using the PPO policy.

        Mirrors the store-carry-forward hold semantics used during training:
        when the policy chooses action 10 (hold) or no contacts are available,
        advance internal time by HOLD_STEP and re-query the policy.  Only
        return None when the policy explicitly drops (action 11) or TTL expires.

        The returned contact may have start_time > current_time; the evaluator's
        _RoutingEnv handles this via send_time = max(current_time, contact.start_time).
        """
        sim_time = current_time
        max_hold_steps = max(1, int((bundle.deadline - current_time) / self.HOLD_STEP) + 1)

        for _ in range(max_hold_steps):
            if sim_time >= bundle.deadline:
                return None

            obs, action_mask, contacts = self._build_obs(
                bundle, contact_plan, current_node, sim_time
            )

            # No contacts in window — advance and retry (implicit hold)
            if not any(action_mask[:self.K]):
                sim_time += self.HOLD_STEP
                continue

            action, _lp, _v = self.ppo_agent.select_action(obs, action_mask, deterministic=True)

            if action < self.K and action_mask[action]:
                return contacts[action]
            elif action == self.K + 1:  # explicit drop
                return None
            else:  # hold (action == self.K)
                sim_time += self.HOLD_STEP

        return None  # TTL would expire during holds

    def _build_obs(self, bundle, contact_plan, current_node: int, current_time: float):
        # Must match DTNRoutingEnv._build_observation() exactly.
        N_minus1 = max(self.n_nodes - 1, 1)
        T = max(self.sim_duration, 1.0)

        ttl_remaining = max(0.0, bundle.deadline - current_time)
        queue_util = 0.0  # unknown outside training env; assume 0

        global_feats = np.array([
            current_node / N_minus1,
            bundle.destination / N_minus1,
            min(current_time / T, 1.0),
            queue_util,
            min(bundle.hops, MAX_HOPS) / MAX_HOPS,
            min(ttl_remaining / MAX_TTL, 1.0),
        ], dtype=np.float32)

        # Same TTL-aware window and reachability-guided selection as
        # DTNRoutingEnv._get_candidate_contacts()
        window = min(ttl_remaining, 7200.0)
        upcoming = contact_plan.get_contacts_from(current_node, current_time, window=window)
        if not upcoming and ttl_remaining > 0:
            upcoming = contact_plan.get_contacts_from(current_node, current_time, window=ttl_remaining)
        pot = contact_plan.delivery_potential_to(bundle.destination)
        upcoming = sorted(
            upcoming,
            key=lambda c: (c.start_time > current_time,
                           -pot.get(c.receiver, 0.0),
                           c.start_time, -c.data_rate)
        )[:self.K]

        contact_feats = np.zeros((self.K, OBS_CONTACT_DIMS), dtype=np.float32)
        action_mask = np.zeros(self.K + 2, dtype=bool)
        action_mask[self.K] = True      # hold always valid
        action_mask[self.K + 1] = True  # drop always valid

        # Destination-distance oracle (global routing signal); matches env.
        n_nodes = getattr(contact_plan, 'n_nodes', self.n_nodes)
        dist_to = contact_plan.hop_distance_to(bundle.destination)

        contacts: list = [None] * self.K
        for k, c in enumerate(upcoming):
            duration = max(c.end_time - c.start_time, 1.0)
            contact_feats[k] = [
                1.0,                                             # active flag
                c.receiver / N_minus1,                          # receiver node
                min(c.start_time / T, 1.0),                     # absolute start time (matches training)
                min(duration / MAX_CONTACT_DURATION, 1.0),      # duration
                min(c.data_rate / MAX_DATA_RATE, 1.0),          # data rate
                min(c.energy_cost / MAX_ENERGY_COST, 1.0),      # energy cost
                float(c.reliability) if self.use_aacr else 1.0, # AACR reliability (masked when use_aacr=False)
                min(dist_to.get(c.receiver, n_nodes) / max(n_nodes, 1), 1.0),  # destination-distance
                float(pot.get(c.receiver, 0.0)),                # delivery-potential (graph encoder)
            ]
            action_mask[k] = True
            contacts[k] = c

        obs = np.clip(np.concatenate([global_feats, contact_feats.flatten()]), 0.0, 1.0)
        return obs, action_mask, contacts

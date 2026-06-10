"""
oracle.py
---------
Oracle upper-bound agent for DTN routing evaluation.

OracleCGR has omniscient knowledge of the complete future contact schedule
and uses offline time-expanded Dijkstra to compute the globally optimal
routing decision for every bundle.

Its Bundle Delivery Ratio (BDR_oracle) serves as the ceiling against which
all learned and heuristic agents are compared::

    relative_performance = BDR_agent / BDR_oracle

Usage
-----
>>> oracle = OracleCGR()
>>> results = oracle.route_all_bundles(bundles, contact_plan)
>>> # results[bundle_id] == (delivered: bool, delay: float, energy: float)
"""

from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Shared data-structure imports (with lightweight stubs for standalone use)
# ---------------------------------------------------------------------------
try:
    from src.env.contact_plan import Contact, ContactPlan
    from src.env.bundle import Bundle
except ImportError:
    from dataclasses import dataclass as _dc, field as _field

    @_dc
    class Contact:
        contact_id: int
        sender: int
        receiver: int
        start_time: float
        end_time: float
        data_rate: float
        energy_cost: float
        reliability: float

    @_dc
    class Bundle:
        bundle_id: int
        source: int
        destination: int
        size_bits: int
        creation_time: float
        ttl: float
        hops: int = 0
        visited_nodes: set = _field(default_factory=set)
        total_energy: float = 0.0

        @property
        def deadline(self) -> float:
            return self.creation_time + self.ttl

        def is_expired(self, current_time: float) -> bool:
            return current_time >= self.deadline

    class ContactPlan:
        def get_contacts_from(
            self, node: int, current_time: float, window: float = 3600
        ) -> List[Contact]:
            return []


_INF = math.inf


# ---------------------------------------------------------------------------
# Internal helpers (duplicated from classical_cgr to keep modules independent)
# ---------------------------------------------------------------------------

def _tx_delay(bundle: Bundle, contact: Contact) -> float:
    if contact.data_rate <= 0:
        return _INF
    return bundle.size_bits / contact.data_rate


def _arr_time(current_time: float, bundle: Bundle, contact: Contact) -> float:
    send_t = max(current_time, contact.start_time)
    finish = send_t + _tx_delay(bundle, contact)
    if finish > contact.end_time:
        return _INF
    return finish


# ---------------------------------------------------------------------------
# OracleCGR
# ---------------------------------------------------------------------------

class OracleCGR:
    """
    Oracle upper bound — knows complete future contact schedule.

    Uses offline time-expanded Dijkstra over the *full* contact graph
    (contacts from every node, not just the current carrier) to determine
    the minimum-delay delivery path for each bundle.

    The oracle is non-causal: it can plan across contacts that have not yet
    occurred at the bundle's creation time, giving it information unavailable
    to any online agent.

    Parameters
    ----------
    optimise : str
        Objective to minimise on each path.
        ``"delay"``   — minimise end-to-end delivery delay (default).
        ``"energy"``  — minimise total energy cost along the path.
        ``"hops"``    — minimise hop count.
    large_window : float
        Time window (seconds) used when fetching contacts from the plan.
        Should comfortably cover the longest TTL in the bundle set.
        Default: 7 days (604 800 s).
    """

    def __init__(
        self,
        optimise: str = "delay",
        large_window: float = 604_800.0,
    ) -> None:
        if optimise not in ("delay", "energy", "hops"):
            raise ValueError("optimise must be 'delay', 'energy', or 'hops'")
        self.optimise = optimise
        self.large_window = large_window

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route_all_bundles(
        self,
        bundles: List[Bundle],
        contact_plan: ContactPlan,
    ) -> Dict[int, Tuple[bool, float, float]]:
        """
        Compute optimal routing outcome for every bundle.

        Parameters
        ----------
        bundles : list of Bundle
        contact_plan : ContactPlan

        Returns
        -------
        dict
            ``{bundle_id: (delivered, delay, energy)}``

            - ``delivered`` : *True* iff a feasible path existed.
            - ``delay``     : end-to-end delay in seconds (0.0 if not delivered).
            - ``energy``    : total energy along optimal path (0.0 if not delivered).
        """
        # Pre-fetch contacts once per unique source node / time pair to avoid
        # redundant plan queries when many bundles share the same source.
        results: Dict[int, Tuple[bool, float, float]] = {}

        for bundle in bundles:
            delivered, delay, energy = self._route_bundle(bundle, contact_plan)
            results[bundle.bundle_id] = (delivered, delay, energy)

        return results

    def route_bundle(
        self,
        bundle: Bundle,
        contact_plan: ContactPlan,
    ) -> Tuple[bool, float, float]:
        """
        Single-bundle convenience wrapper.

        Returns
        -------
        (delivered, delay, energy)
        """
        return self._route_bundle(bundle, contact_plan)

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------

    def _route_bundle(
        self,
        bundle: Bundle,
        contact_plan: ContactPlan,
    ) -> Tuple[bool, float, float]:
        """
        Time-expanded Dijkstra with full future knowledge.

        State : (node_id, earliest_arrival_time_at_node)
        Cost  : determined by ``self.optimise``

        Returns (delivered, delay, total_energy).
        """
        source = bundle.source
        dest = bundle.destination
        t0 = bundle.creation_time
        deadline = bundle.deadline

        if t0 >= deadline:
            return False, 0.0, 0.0

        # dist[node] = (cost, arrival_time, total_energy)
        dist: Dict[int, Tuple[float, float, float]] = {
            source: (0.0, t0, 0.0)
        }

        # heap entries: (cost, arrival_time, node, total_energy)
        heap: List[Tuple[float, float, int, float]] = [
            (0.0, t0, source, 0.0)
        ]

        # Lazily fetch contacts per node; use a large window rooted at t0
        # so the oracle sees all future contacts.
        fetched: set = set()
        node_contacts: Dict[int, List[Contact]] = {}

        def _fetch(node: int) -> None:
            if node in fetched:
                return
            fetched.add(node)
            node_contacts[node] = contact_plan.get_contacts_from(
                node, t0, window=self.large_window
            )

        _fetch(source)

        while heap:
            cost, arr_t, node, acc_energy = heapq.heappop(heap)

            # Check against best known cost (not just arrival time)
            best_cost, best_arr, _ = dist.get(node, (_INF, _INF, _INF))
            if cost > best_cost + 1e-9:
                continue

            if node == dest:
                delay = arr_t - t0
                return True, delay, acc_energy

            if arr_t >= deadline:
                continue

            _fetch(node)

            for c in node_contacts.get(node, []):
                if c.sender != node:
                    continue
                if c.end_time <= arr_t:
                    continue  # contact already over by arrival
                if c.start_time >= deadline:
                    continue  # contact starts after deadline

                new_arr = _arr_time(arr_t, bundle, c)
                if new_arr >= deadline or new_arr == _INF:
                    continue

                new_energy = acc_energy + c.energy_cost
                new_cost = self._edge_cost(
                    cost=cost,
                    arr_time=new_arr,
                    t0=t0,
                    energy_cost=c.energy_cost,
                    acc_energy=acc_energy,
                )

                prev_cost, _, _ = dist.get(c.receiver, (_INF, _INF, _INF))
                if new_cost < prev_cost - 1e-9:
                    dist[c.receiver] = (new_cost, new_arr, new_energy)
                    _fetch(c.receiver)
                    heapq.heappush(
                        heap, (new_cost, new_arr, c.receiver, new_energy)
                    )

        return False, 0.0, 0.0

    def _edge_cost(
        self,
        cost: float,
        arr_time: float,
        t0: float,
        energy_cost: float,
        acc_energy: float,
    ) -> float:
        """Return the path cost to use in the priority queue."""
        if self.optimise == "delay":
            return arr_time - t0
        if self.optimise == "energy":
            return acc_energy + energy_cost
        # hops
        return cost + 1.0

    # ------------------------------------------------------------------
    # Convenience: compute oracle BDR for a bundle set
    # ------------------------------------------------------------------

    def compute_oracle_bdr(
        self,
        bundles: List[Bundle],
        contact_plan: ContactPlan,
    ) -> float:
        """
        Return the oracle Bundle Delivery Ratio for *bundles*.

        BDR_oracle = number_delivered / total_bundles
        """
        if not bundles:
            return 0.0
        results = self.route_all_bundles(bundles, contact_plan)
        delivered = sum(1 for delivered, _, _ in results.values() if delivered)
        return delivered / len(bundles)

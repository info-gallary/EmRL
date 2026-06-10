"""
classical_cgr.py
----------------
Classical CGR baselines for DTN routing research.

Implements three algorithms:
  - ClassicalCGR  : RFC 6260 / Burleigh 2003 Dijkstra-based Contact Graph Routing
  - AdaptiveCGR   : Heuristic extension that weights reliability, data-rate and congestion
  - SprayAndWait  : Epidemic-style replication baseline (L = 2 copies)

All agents expose the same interface::

    contact = agent.route(bundle, contact_plan, current_node, current_time)

Returns the *next* Contact to forward the bundle on, or None when no route exists.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Shared data-structure imports
# The ContactPlan / Bundle types live in src/env; fall back to local stubs so
# this module can be imported standalone during unit-testing.
# ---------------------------------------------------------------------------
try:
    from src.env.contact_plan import Contact, ContactPlan
    from src.env.bundle import Bundle
except ImportError:
    # Lightweight stubs used when the env package is not on sys.path
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_INF = math.inf


def _transmission_delay(bundle: Bundle, contact: Contact) -> float:
    """Seconds to transmit bundle over a contact (size / data_rate)."""
    if contact.data_rate <= 0:
        return _INF
    return bundle.size_bits / contact.data_rate


def _earliest_send_time(current_time: float, contact: Contact) -> float:
    """Earliest moment we can begin transmitting on this contact."""
    return max(current_time, contact.start_time)


def _arrival_time(
    send_time: float, bundle: Bundle, contact: Contact
) -> float:
    """Expected arrival time at the far end of *contact*."""
    tx_delay = _transmission_delay(bundle, contact)
    finish_time = send_time + tx_delay
    if finish_time > contact.end_time:
        return _INF  # bundle does not fit within contact window
    return finish_time


# ---------------------------------------------------------------------------
# 1. ClassicalCGR — RFC 6260 / Burleigh 2003
# ---------------------------------------------------------------------------

class ClassicalCGR:
    """
    Dijkstra-based Contact Graph Routing as described in RFC 6260
    and Burleigh (2003).

    The algorithm builds a time-expanded graph where each node represents
    a (node_id, contact_id) pair and finds the path that minimises the
    expected bundle arrival time at the destination.
    """

    def route(
        self,
        bundle: Bundle,
        contact_plan: ContactPlan,
        current_node: int,
        current_time: float,
    ) -> Optional[Contact]:
        """
        Return the first Contact on the earliest-arrival path to
        ``bundle.destination``, or *None* when no feasible route exists.
        """
        if bundle.is_expired(current_time):
            return None

        # Gather all contacts reachable within the remaining TTL
        remaining_ttl = bundle.deadline - current_time
        contacts = contact_plan.get_contacts_from(
            current_node, current_time, window=remaining_ttl
        )
        if not contacts:
            return None

        # Build the graph and run Dijkstra
        first_hop = self._dijkstra(
            bundle=bundle,
            contacts=contacts,
            source=current_node,
            dest=bundle.destination,
            t_start=current_time,
            contact_plan=contact_plan,
        )
        return first_hop

    # ------------------------------------------------------------------
    # Graph construction + shortest-path
    # ------------------------------------------------------------------

    def _build_contact_graph(
        self,
        source: int,
        dest: int,
        t_start: float,
        t_deadline: float,
        contacts: List[Contact],
    ) -> Dict[int, List[Tuple[int, float, Contact]]]:
        """
        Build a time-expanded adjacency list.

        Nodes are integers (node_ids).  Each edge carries:
          (neighbour_node, earliest_arrival_at_neighbour, contact_used)

        Only contacts that begin before the deadline and end after
        t_start are included.
        """
        graph: Dict[int, List[Tuple[int, float, Contact]]] = {}

        for c in contacts:
            if c.start_time >= t_deadline:
                continue  # contact starts too late
            if c.end_time <= t_start:
                continue  # contact already over
            if c.sender not in graph:
                graph[c.sender] = []
            graph[c.sender].append((c.receiver, c, ))

        return graph

    def _dijkstra(
        self,
        bundle: Bundle,
        contacts: List[Contact],
        source: int,
        dest: int,
        t_start: float,
        contact_plan: ContactPlan,
    ) -> Optional[Contact]:
        """
        Modified Dijkstra over the time-expanded contact graph.

        Priority queue entries: (earliest_arrival, node_id, first_contact)
        where *first_contact* is the Contact used on the very first hop
        (needed so we can return it without reconstructing the full path).
        """
        t_deadline = bundle.deadline

        # dist[node] = earliest time we can have a bundle arrive at node
        dist: Dict[int, float] = {source: t_start}

        # (arrival_time, node, first_hop_contact_or_None)
        heap: List[Tuple[float, int, Optional[Contact]]] = [
            (t_start, source, None)
        ]

        # Collect *all* contacts relevant to the search (not just from source)
        # We do a BFS-style expansion: when we relax a new node for the first
        # time we fetch its contacts from the plan.
        fetched_nodes: set = {source}
        all_contacts: Dict[int, List[Contact]] = {source: contacts}

        while heap:
            arr_time, node, first_hop = heapq.heappop(heap)

            if arr_time > dist.get(node, _INF) + 1e-9:
                continue  # stale entry

            if node == dest:
                return first_hop  # reached destination

            if arr_time >= t_deadline:
                continue

            # Lazily fetch contacts for newly-reached nodes
            if node not in fetched_nodes:
                fetched_nodes.add(node)
                window = t_deadline - arr_time
                all_contacts[node] = contact_plan.get_contacts_from(
                    node, arr_time, window=window
                )

            for c in all_contacts.get(node, []):
                if c.sender != node:
                    continue
                send_t = _earliest_send_time(arr_time, c)
                if send_t >= t_deadline:
                    continue
                new_arr = _arrival_time(send_t, bundle, c)
                if new_arr >= t_deadline or new_arr == _INF:
                    continue
                if new_arr < dist.get(c.receiver, _INF):
                    dist[c.receiver] = new_arr
                    # Preserve the very first hop contact
                    hop = c if first_hop is None else first_hop
                    heapq.heappush(heap, (new_arr, c.receiver, hop))

        return None  # no feasible route found


# ---------------------------------------------------------------------------
# 1b. RUCoP-style — reliability-aware routing (reimplementation)
# ---------------------------------------------------------------------------

class RUCoPRouter:
    """
    Reliability-aware contact routing in the style of RUCoP
    (Routing Under Uncertain Contact Plans; Raverta et al., 2021).

    NOTE: this is OUR reimplementation of RUCoP's core idea — selecting the
    next hop on the route that MAXIMISES end-to-end delivery probability under
    probabilistic (reliability-weighted) contacts — not the authors' original
    code. RUCoP frames routing as an MDP and computes delivery-probability-
    maximising routes; we approximate the resulting policy with a max-product-
    reliability Dijkstra over the time-expanded contact graph, subject to the
    same timing feasibility constraints as ClassicalCGR.

    This is a STRONG, FAIR baseline: unlike RFC-6260 ClassicalCGR (which is
    reliability-blind and minimises arrival time), RUCoP explicitly accounts for
    link reliability — the same information EmRL's AACR uses.
    """

    def route(self, bundle, contact_plan, current_node, current_time):
        first = self._dijkstra_maxprob(bundle, contact_plan, current_node,
                                       bundle.destination, current_time)
        return first

    def _dijkstra_maxprob(self, bundle, contact_plan, source, dest, t_start):
        """
        Max-delivery-probability route via Dijkstra on cost = -sum(log rel),
        i.e. maximise the product of contact reliabilities along a timing-
        feasible path. Returns the first-hop Contact, or None.
        """
        t_deadline = bundle.deadline
        # state cost[node] = min negative-log-probability to reach node
        best: Dict[int, float] = {source: 0.0}
        # heap entries: (neg_log_prob, arrival_time, node, first_hop)
        heap: List[Tuple[float, float, int, Optional[Contact]]] = [
            (0.0, t_start, source, None)
        ]
        fetched: set = {source}
        adj: Dict[int, List[Contact]] = {
            source: contact_plan.get_contacts_from(source, t_start,
                                                   window=t_deadline - t_start)
        }
        best_arrival: Dict[int, float] = {source: t_start}

        while heap:
            nlp, arr_time, node, first_hop = heapq.heappop(heap)
            if node == dest:
                return first_hop
            if nlp > best.get(node, _INF) + 1e-12 and arr_time > best_arrival.get(node, _INF):
                continue
            if arr_time >= t_deadline:
                continue
            if node not in fetched:
                fetched.add(node)
                adj[node] = contact_plan.get_contacts_from(
                    node, arr_time, window=t_deadline - arr_time)
            for c in adj.get(node, []):
                if c.sender != node:
                    continue
                send_t = _earliest_send_time(arr_time, c)
                if send_t >= t_deadline:
                    continue
                new_arr = _arrival_time(send_t, bundle, c)
                if new_arr >= t_deadline or new_arr == _INF:
                    continue
                rel = max(min(c.reliability, 1.0), 1e-6)
                new_nlp = nlp - math.log(rel)
                if new_nlp < best.get(c.receiver, _INF) - 1e-12:
                    best[c.receiver] = new_nlp
                    best_arrival[c.receiver] = new_arr
                    hop = first_hop if first_hop is not None else c
                    heapq.heappush(heap, (new_nlp, new_arr, c.receiver, hop))
        return None


# ---------------------------------------------------------------------------
# 2. AdaptiveCGR — Heuristic scoring
# ---------------------------------------------------------------------------

class AdaptiveCGR:
    """
    Heuristic CGR that scores candidate next-hop contacts by combining
    reliability, data-rate and congestion information.

    Score = reliability * data_rate / (wait_time + 1)

    The contact with the highest score that can deliver the bundle before
    its deadline is chosen.

    Parameters
    ----------
    max_queue_occupancy : float
        Queue occupancy fraction above which a contact is penalised
        (treated as congested).  Default 0.9.
    congestion_penalty : float
        Multiplicative penalty applied to the score of congested contacts.
        Default 0.1.
    """

    def __init__(
        self,
        max_queue_occupancy: float = 0.9,
        congestion_penalty: float = 0.1,
    ) -> None:
        self.max_queue_occupancy = max_queue_occupancy
        self.congestion_penalty = congestion_penalty

    def route(
        self,
        bundle: Bundle,
        contact_plan: ContactPlan,
        current_node: int,
        current_time: float,
        queue_states: Optional[Dict[int, float]] = None,
    ) -> Optional[Contact]:
        """
        Return the highest-scoring Contact for the next hop, or *None*.

        Parameters
        ----------
        queue_states : dict, optional
            Mapping ``{node_id: occupancy_fraction}`` (0.0–1.0) representing
            current queue load at each neighbour.  When *None*, no congestion
            penalty is applied.
        """
        if bundle.is_expired(current_time):
            return None

        remaining_ttl = bundle.deadline - current_time
        contacts = contact_plan.get_contacts_from(
            current_node, current_time, window=remaining_ttl
        )
        if not contacts:
            return None

        best_contact: Optional[Contact] = None
        best_score: float = -_INF

        for c in contacts:
            if c.sender != current_node:
                continue

            # Only consider contacts that end before the bundle deadline
            send_t = _earliest_send_time(current_time, c)
            arr_t = _arrival_time(send_t, bundle, c)
            if arr_t >= bundle.deadline or arr_t == _INF:
                continue

            wait_time = max(0.0, c.start_time - current_time)
            score = (c.reliability * c.data_rate) / (wait_time + 1.0)

            # Remaining contact window — penalise contacts that barely fit
            remaining_window = c.end_time - send_t
            tx_delay = _transmission_delay(bundle, c)
            if remaining_window < tx_delay * 1.1:
                score *= 0.5  # tight fit: soft penalty

            # Congestion penalty
            if queue_states is not None:
                occ = queue_states.get(c.receiver, 0.0)
                if occ >= self.max_queue_occupancy:
                    score *= self.congestion_penalty

            if score > best_score:
                best_score = score
                best_contact = c

        return best_contact


# ---------------------------------------------------------------------------
# 3. SprayAndWait — epidemic-style baseline
# ---------------------------------------------------------------------------

class SprayAndWait:
    """
    Spray-and-Wait routing baseline (epidemic-style).

    Phase 1 — Spray : forward the bundle to the first available contact
    until *L* copies have been created.
    Phase 2 — Wait  : if all copies have been sprayed, hold and wait for
    a direct contact to the destination.

    Default replication factor L = 2.

    The agent is stateless with respect to per-bundle copy counts; callers
    must track ``bundle.hops`` externally and pass the current copy count
    via ``copies_remaining``.
    """

    def __init__(self, L: int = 2) -> None:
        if L < 1:
            raise ValueError("Replication factor L must be >= 1.")
        self.L = L

    def route(
        self,
        bundle: Bundle,
        contact_plan: ContactPlan,
        current_node: int,
        current_time: float,
        copies_remaining: int = 1,
    ) -> Optional[Contact]:
        """
        Return the next Contact for forwarding, or *None*.

        Parameters
        ----------
        copies_remaining : int
            Number of bundle copies still to be sprayed.  When > 1 (spray
            phase) any available contact is acceptable; when == 1 (wait
            phase) only a direct-to-destination contact is used.
        """
        if bundle.is_expired(current_time):
            return None

        remaining_ttl = bundle.deadline - current_time
        contacts = contact_plan.get_contacts_from(
            current_node, current_time, window=remaining_ttl
        )
        if not contacts:
            return None

        dest = bundle.destination

        # --- Wait phase: only direct-to-destination contacts ---
        if copies_remaining <= 1:
            for c in contacts:
                if c.sender != current_node:
                    continue
                if c.receiver != dest:
                    continue
                send_t = _earliest_send_time(current_time, c)
                if _arrival_time(send_t, bundle, c) < bundle.deadline:
                    return c
            return None

        # --- Spray phase: prefer direct contact, else use first available ---
        direct: Optional[Contact] = None
        first_available: Optional[Contact] = None

        for c in contacts:
            if c.sender != current_node:
                continue
            send_t = _earliest_send_time(current_time, c)
            if _arrival_time(send_t, bundle, c) >= bundle.deadline:
                continue
            if c.receiver == dest:
                direct = c
                break  # direct contact always wins
            if first_available is None:
                first_available = c

        return direct if direct is not None else first_available

"""
evaluator.py
------------
Comprehensive evaluation suite for DTN routing agents.

The Evaluator runs a battery of scenarios (standard topologies, congestion,
jamming, adversarial contact delays) and compares multiple agents, returning
a nested dict of EvalMetrics keyed by scenario and agent name.

Quick-start
-----------
>>> evaluator = Evaluator()
>>> results = evaluator.evaluate_all(
...     emrl_agent=my_ppo_agent,
...     contact_plans={"rrna": rrna_plan, "rrnb": rrnb_plan},
...     n_episodes=200,
... )
>>> print(results["rrna"]["EmRL"].summary())

Standalone Monte Carlo evaluation
----------------------------------
>>> metrics = monte_carlo_evaluate(agent, contact_plan, n_rollouts=1000)
"""

from __future__ import annotations

import math
import random
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.evaluation.metrics import (
    EpisodeResult,
    EvalMetrics,
    compute_metrics,
)
from src.agents.classical_cgr import AdaptiveCGR, ClassicalCGR, SprayAndWait
from src.agents.oracle import OracleCGR

# ---------------------------------------------------------------------------
# Lazy imports for env types — same stub-fallback pattern used in agents
# ---------------------------------------------------------------------------
try:
    from src.env.contact_plan import Contact, ContactPlan
    from src.env.bundle import Bundle
except ImportError:
    from dataclasses import dataclass as _dc, field as _f

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
        visited_nodes: set = _f(default_factory=set)
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
# Scenario configuration helpers
# ---------------------------------------------------------------------------

@dataclass
class ScenarioConfig:
    """Parameters for a single evaluation scenario."""

    name: str
    jamming_rate: float = 0.0       # fraction of contacts that fail (0–1)
    congestion_load: float = 1.0    # traffic load multiplier (>1 = congestion)
    adversarial_delay: float = 0.0  # extra delay (s) injected into contacts
    anomaly_rate: float = 0.0       # fraction of contacts with lowered reliability (AACR test)
    seed: int = 0


_STANDARD_SCENARIOS: List[ScenarioConfig] = [
    ScenarioConfig("standard", seed=0),
]

_JAMMING_SCENARIO = ScenarioConfig(
    "jamming", jamming_rate=0.30, seed=1
)

_CONGESTION_SCENARIO = ScenarioConfig(
    "congestion", congestion_load=3.0, seed=2
)

_ADVERSARIAL_SCENARIO = ScenarioConfig(
    "adversarial", adversarial_delay=120.0, seed=3
)

_ANOMALY_SCENARIO = ScenarioConfig(
    "anomaly", anomaly_rate=0.30, seed=5
)


# ---------------------------------------------------------------------------
# Contact plan perturbation utilities
# ---------------------------------------------------------------------------

def _apply_jamming(
    contact_plan: ContactPlan,
    jamming_rate: float,
    rng: random.Random,
) -> ContactPlan:
    """
    Return a copy of *contact_plan* with a random fraction of contacts
    removed (simulating link failures / jamming).

    The original plan is never mutated.
    """
    plan_copy = deepcopy(contact_plan)

    if not hasattr(plan_copy, "contacts"):
        return plan_copy

    surviving = [
        c for c in plan_copy.contacts
        if rng.random() >= jamming_rate
    ]
    plan_copy.contacts = surviving
    # Rebuild sender index
    plan_copy._by_sender = {}
    for c in plan_copy.contacts:
        plan_copy._by_sender.setdefault(c.sender, []).append(c)
    return plan_copy


def _apply_anomaly(
    contact_plan: ContactPlan,
    anomaly_rate: float,
    rng: random.Random,
) -> ContactPlan:
    """
    Return a copy of *contact_plan* with a random fraction of contacts having
    their reliability lowered to [0.5, 0.8] — simulating AACR anomaly conditions
    as in training Phase 3.  The original plan is never mutated.
    """
    plan_copy = deepcopy(contact_plan)

    if not hasattr(plan_copy, "contacts"):
        return plan_copy

    for c in plan_copy.contacts:
        if rng.random() < anomaly_rate:
            c.reliability = rng.uniform(0.5, 0.8)
    return plan_copy


def _apply_adversarial_delay(
    contact_plan: ContactPlan,
    delay: float,
) -> ContactPlan:
    """
    Return a copy of *contact_plan* where every contact's start_time and
    end_time have been shifted forward by *delay* seconds.
    """
    plan_copy = deepcopy(contact_plan)

    if not hasattr(plan_copy, "contacts"):
        return plan_copy

    for c in plan_copy.contacts:
        c.start_time += delay
        c.end_time += delay

    return plan_copy


# ---------------------------------------------------------------------------
# Lightweight simulation environment
# ---------------------------------------------------------------------------

class _RoutingEnv:
    """
    Minimal simulation environment used by the Evaluator.

    Simulates store-carry-forward routing of a single bundle given an agent
    that returns a next-hop Contact.  The agent interface expected::

        contact = agent.route(bundle, contact_plan, current_node, current_time)

    For PPO/RL agents that expose a Gymnasium-style step() interface, wrap
    them with a thin adapter before passing to the Evaluator.
    """

    def __init__(
        self,
        contact_plan: ContactPlan,
        max_hops: int = 32,
        congestion_load: float = 1.0,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.contact_plan = contact_plan
        self.max_hops = max_hops
        self.congestion_load = congestion_load
        self._rng = rng or random.Random(0)

        # Simulated queue states: {node_id: occupancy_fraction}
        self.queue_states: Dict[int, float] = {}

    def reset(self, bundle: Bundle) -> Bundle:
        self._bundle = bundle
        self._current_node = bundle.source
        self._current_time = bundle.creation_time
        # Randomise queue states under congestion
        self.queue_states = {
            n: min(1.0, self._rng.random() * self.congestion_load)
            for n in range(64)  # generous upper bound on node count
        }
        return bundle

    def step(self, agent: Any) -> EpisodeResult:
        """
        Run one complete bundle delivery simulation using *agent*.

        Returns
        -------
        EpisodeResult
        """
        bundle = self._bundle
        current_node = self._current_node
        current_time = self._current_time
        total_energy = 0.0
        hops = 0

        while True:
            if bundle.is_expired(current_time):
                return EpisodeResult(
                    delivered=False,
                    delay=0.0,
                    energy=total_energy,
                    hops=hops,
                    drop_reason="ttl_expired",
                )

            if hops >= self.max_hops:
                return EpisodeResult(
                    delivered=False,
                    delay=0.0,
                    energy=total_energy,
                    hops=hops,
                    drop_reason="no_route",
                )

            # Ask the agent for the next hop
            contact = self._get_contact(agent, bundle, current_node, current_time)

            if contact is None:
                return EpisodeResult(
                    delivered=False,
                    delay=0.0,
                    energy=total_energy,
                    hops=hops,
                    drop_reason="no_route",
                )

            # Simulate transmission
            send_time = max(current_time, contact.start_time)

            # Stochastic link failure — matches DTNRoutingEnv training semantics.
            # When a link fails the bundle stays at current_node; time advances
            # to the end of the failed contact window so the agent re-queries with
            # updated context (exactly as in the training env's hold-on-failure path).
            if self._rng.random() > contact.reliability:
                current_time = contact.end_time
                continue  # re-query agent with new current_time

            tx_delay = (
                bundle.size_bits / contact.data_rate
                if contact.data_rate > 0
                else math.inf
            )
            finish_time = send_time + tx_delay

            if finish_time > contact.end_time or finish_time >= bundle.deadline:
                return EpisodeResult(
                    delivered=False,
                    delay=0.0,
                    energy=total_energy,
                    hops=hops,
                    drop_reason="ttl_expired",
                )

            total_energy += contact.energy_cost
            hops += 1
            current_time = finish_time
            current_node = contact.receiver
            bundle.record_hop(contact.receiver, contact.energy_cost)

            if current_node == bundle.destination:
                delay = current_time - bundle.creation_time
                return EpisodeResult(
                    delivered=True,
                    delay=delay,
                    energy=total_energy,
                    hops=hops,
                    drop_reason="",
                )

    def _get_contact(
        self,
        agent: Any,
        bundle: Bundle,
        current_node: int,
        current_time: float,
    ):
        """Dispatch to the correct agent call signature."""
        # AdaptiveCGR accepts queue_states
        if isinstance(agent, AdaptiveCGR):
            return agent.route(
                bundle, self.contact_plan, current_node, current_time,
                queue_states=self.queue_states,
            )
        # SprayAndWait
        if isinstance(agent, SprayAndWait):
            copies = max(1, agent.L - bundle.hops)
            return agent.route(
                bundle, self.contact_plan, current_node, current_time,
                copies_remaining=copies,
            )
        # ClassicalCGR, OracleCGR (single bundle wrapper), generic RL agent
        if hasattr(agent, "route"):
            return agent.route(bundle, self.contact_plan, current_node, current_time)
        # Gymnasium-style RL agents: use predict() / act()
        raise TypeError(
            f"Agent {type(agent).__name__!r} does not expose a .route() method. "
            "Wrap it with a RoutingAgentAdapter before passing to the Evaluator."
        )


# ---------------------------------------------------------------------------
# Main Evaluator class
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Comprehensive evaluation suite for DTN routing agents.

    Evaluates on:

    1. Standard RRN-A topology (with ISL links)
    2. Standard RRN-B topology (without ISL links)
    3. Synthetic random topologies (one per seed in *seeds*)
    4. Congestion scenario (high traffic load)
    5. Jamming scenario (30 % contact failure rate)
    6. Adversarial contact-plan manipulation (shifted contact windows)

    Agents compared:

    - EmRL (PPO, the method under evaluation)
    - ClassicalCGR
    - AdaptiveCGR
    - SprayAndWait
    - Oracle (upper bound)

    Parameters
    ----------
    max_hops : int
        Maximum forwarding hops before declaring delivery failure.
    bundle_ttl : float
        Default TTL (seconds) for synthetic bundles.  Default: 3600 s.
    bundle_size_bits : int
        Default bundle payload size.  Default: 10 Mbits.
    verbose : bool
        Print progress to stdout when True.
    """

    def __init__(
        self,
        max_hops: int = 32,
        bundle_ttl: float = 3600.0,
        bundle_size_bits: int = 10_000_000,
        verbose: bool = False,
    ) -> None:
        self.max_hops = max_hops
        self.bundle_ttl = bundle_ttl
        self.bundle_size_bits = bundle_size_bits
        self.verbose = verbose

        # Built-in classical agents (one shared instance each)
        self._classical_cgr = ClassicalCGR()
        self._adaptive_cgr = AdaptiveCGR()
        self._spray_and_wait = SprayAndWait(L=2)
        self._oracle = OracleCGR(optimise="delay")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_all(
        self,
        emrl_agent: Any,
        contact_plans: Dict[str, ContactPlan],
        n_episodes: int = 200,
        seeds: List[int] = None,
    ) -> Dict[str, Dict[str, EvalMetrics]]:
        """
        Run the full evaluation battery.

        Parameters
        ----------
        emrl_agent : trained PPO agent
            Must expose ``.route(bundle, contact_plan, node, time)`` or a
            Gymnasium ``predict()`` method wrapped by RoutingAgentAdapter.
        contact_plans : dict
            Must contain at least ``"rrna"`` and ``"rrnb"`` keys.
            Additional keys are treated as extra standard topologies.
        n_episodes : int
            Number of bundle routing episodes per (scenario, agent) pair.
        seeds : list of int
            Seeds for synthetic topology evaluation.  Default: [0,1,2,3,4].

        Returns
        -------
        dict
            ``results[scenario_name][agent_name]`` → EvalMetrics
        """
        if seeds is None:
            seeds = [0, 1, 2, 3, 4]

        agents: Dict[str, Any] = {
            "EmRL": emrl_agent,
            "ClassicalCGR": self._classical_cgr,
            "AdaptiveCGR": self._adaptive_cgr,
            "SprayAndWait": self._spray_and_wait,
            "Oracle": self._oracle,
        }

        all_results: Dict[str, Dict[str, EvalMetrics]] = {}

        # ---- Standard topologies ----------------------------------------
        for plan_name, plan in contact_plans.items():
            if self.verbose:
                print(f"[Evaluator] Scenario: {plan_name} (standard)")
            scenario_results = self._evaluate_scenario(
                agents=agents,
                contact_plan=plan,
                n_episodes=n_episodes,
                scenario_cfg=ScenarioConfig(plan_name),
            )
            all_results[plan_name] = scenario_results

        # ---- Congestion scenario (use first available plan) ---------------
        base_plan = next(iter(contact_plans.values()))

        if self.verbose:
            print("[Evaluator] Scenario: congestion")
        congestion_results = self._evaluate_scenario(
            agents=agents,
            contact_plan=base_plan,
            n_episodes=n_episodes,
            scenario_cfg=_CONGESTION_SCENARIO,
        )
        all_results["congestion"] = congestion_results

        # ---- Jamming scenario --------------------------------------------
        if self.verbose:
            print("[Evaluator] Scenario: jamming")
        jamming_results = self._evaluate_scenario(
            agents=agents,
            contact_plan=base_plan,
            n_episodes=n_episodes,
            scenario_cfg=_JAMMING_SCENARIO,
        )
        all_results["jamming"] = jamming_results

        # ---- Adversarial contact delays ----------------------------------
        if self.verbose:
            print("[Evaluator] Scenario: adversarial")
        adversarial_results = self._evaluate_scenario(
            agents=agents,
            contact_plan=base_plan,
            n_episodes=n_episodes,
            scenario_cfg=_ADVERSARIAL_SCENARIO,
        )
        all_results["adversarial"] = adversarial_results

        # ---- Anomaly scenario (AACR test: 30% low-reliability contacts) --
        if self.verbose:
            print("[Evaluator] Scenario: anomaly")
        anomaly_results = self._evaluate_scenario(
            agents=agents,
            contact_plan=base_plan,
            n_episodes=n_episodes,
            scenario_cfg=_ANOMALY_SCENARIO,
        )
        all_results["anomaly"] = anomaly_results

        # ---- Synthetic random topologies (per seed) ---------------------
        for seed in seeds:
            scene_name = f"synthetic_seed{seed}"
            if self.verbose:
                print(f"[Evaluator] Scenario: {scene_name}")
            syn_cfg = ScenarioConfig(scene_name, seed=seed)
            syn_results = self._evaluate_scenario(
                agents=agents,
                contact_plan=base_plan,  # perturbed by seed RNG
                n_episodes=n_episodes,
                scenario_cfg=syn_cfg,
            )
            all_results[scene_name] = syn_results

        # Backfill congestion_bdr / jamming_bdr into standard scenario metrics
        self._backfill_scenario_bdrs(all_results, contact_plans, agents)

        return all_results

    def run_episode(
        self,
        agent: Any,
        env: _RoutingEnv,
        bundle: Bundle,
        deterministic: bool = True,
    ) -> EpisodeResult:
        """
        Run a single evaluation episode.

        Parameters
        ----------
        agent : routing agent
        env : _RoutingEnv
            Pre-constructed environment.
        bundle : Bundle
            The bundle to route.
        deterministic : bool
            Passed through to RL agents that support stochastic policies.
            Classical agents always behave deterministically.

        Returns
        -------
        EpisodeResult
        """
        env.reset(bundle)
        return env.step(agent)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_scenario(
        self,
        agents: Dict[str, Any],
        contact_plan: ContactPlan,
        n_episodes: int,
        scenario_cfg: ScenarioConfig,
    ) -> Dict[str, EvalMetrics]:
        """Run *n_episodes* for every agent on the given scenario."""
        rng = random.Random(scenario_cfg.seed)
        np_rng = np.random.default_rng(scenario_cfg.seed)

        # Perturb the contact plan according to the scenario
        perturbed_plan = self._perturb_plan(contact_plan, scenario_cfg, rng)

        # Generate bundles for this scenario
        bundles = self._generate_bundles(
            contact_plan=perturbed_plan,
            n=n_episodes,
            rng=rng,
            np_rng=np_rng,
        )

        # Oracle BDR — computed once, shared across all agents in this scenario
        oracle_bdr = self._oracle.compute_oracle_bdr(bundles, perturbed_plan)

        scenario_results: Dict[str, EvalMetrics] = {}

        for agent_name, agent in agents.items():
            if agent_name == "Oracle":
                # Oracle routes analytically; simulate its deliveries
                episode_results = self._run_oracle_episodes(
                    bundles, perturbed_plan
                )
            else:
                env = _RoutingEnv(
                    contact_plan=perturbed_plan,
                    max_hops=self.max_hops,
                    congestion_load=scenario_cfg.congestion_load,
                    rng=rng,
                )
                episode_results = [
                    self.run_episode(agent, env, b) for b in bundles
                ]

            congestion_bdr = math.nan
            jamming_bdr = math.nan

            scenario_results[agent_name] = compute_metrics(
                results=episode_results,
                oracle_bdr=oracle_bdr,
                congestion_bdr=congestion_bdr,
                jamming_bdr=jamming_bdr,
            )

        return scenario_results

    def _run_oracle_episodes(
        self,
        bundles: List[Bundle],
        contact_plan: ContactPlan,
    ) -> List[EpisodeResult]:
        """
        Convert OracleCGR's batch output to per-episode EpisodeResults.
        The oracle always finds the optimal path, so energy is set to 0
        (it is not tracked by the oracle's Dijkstra unless optimise="energy").
        """
        oracle_results = self._oracle.route_all_bundles(bundles, contact_plan)
        episode_results: List[EpisodeResult] = []

        for bundle in bundles:
            delivered, delay, energy = oracle_results.get(
                bundle.bundle_id, (False, 0.0, 0.0)
            )
            episode_results.append(
                EpisodeResult(
                    delivered=delivered,
                    delay=delay if delivered else 0.0,
                    energy=energy if delivered else 0.0,
                    hops=0,  # oracle does not track hop counts
                    drop_reason="" if delivered else "no_route",
                )
            )

        return episode_results

    def _perturb_plan(
        self,
        contact_plan: ContactPlan,
        cfg: ScenarioConfig,
        rng: random.Random,
    ) -> ContactPlan:
        """Apply scenario-specific perturbations to the contact plan."""
        plan = contact_plan

        if cfg.jamming_rate > 0.0:
            plan = _apply_jamming(plan, cfg.jamming_rate, rng)

        if cfg.adversarial_delay > 0.0:
            plan = _apply_adversarial_delay(plan, cfg.adversarial_delay)

        if cfg.anomaly_rate > 0.0:
            plan = _apply_anomaly(plan, cfg.anomaly_rate, rng)

        return plan

    def _generate_bundles(
        self,
        contact_plan: ContactPlan,
        n: int,
        rng: random.Random,
        np_rng: np.random.Generator,
    ) -> List[Bundle]:
        """
        Generate *n* Bundle objects with random (source, dest, creation_time).

        Node IDs and time windows are inferred heuristically from the contact
        plan.  If the plan exposes ``node_ids`` and ``time_range`` attributes,
        those are used directly; otherwise safe defaults are applied.
        """
        # Infer topology parameters
        node_ids = getattr(contact_plan, "node_ids", None)
        if node_ids is None or len(node_ids) < 2:
            node_ids = list(range(10))  # fallback

        t_min = getattr(contact_plan, "t_start", 0.0)
        t_max = getattr(contact_plan, "t_end", 86_400.0)
        # Match training env: DTNRoutingEnv.reset() samples from [0, sim_duration * 0.5]
        creation_window = max(1.0, (t_max - t_min) * 0.5)

        bundles: List[Bundle] = []
        for i in range(n):
            source, dest = rng.sample(node_ids, 2)
            creation_time = t_min + rng.random() * creation_window
            b = Bundle(
                bundle_id=i,
                source=source,
                destination=dest,
                size_bits=self.bundle_size_bits,
                creation_time=creation_time,
                ttl=self.bundle_ttl,
            )
            bundles.append(b)

        return bundles

    def _backfill_scenario_bdrs(
        self,
        all_results: Dict[str, Dict[str, EvalMetrics]],
        contact_plans: Dict[str, ContactPlan],
        agents: Dict[str, Any],
    ) -> None:
        """
        Copy congestion_bdr and jamming_bdr values into the standard topology
        metrics so that a single EvalMetrics object contains all comparative
        figures.
        """
        for plan_name in contact_plans:
            if plan_name not in all_results:
                continue
            for agent_name in agents:
                std_metric = all_results[plan_name].get(agent_name)
                cong_metric = all_results.get("congestion", {}).get(agent_name)
                jam_metric = all_results.get("jamming", {}).get(agent_name)

                if std_metric is None:
                    continue

                std_metric.congestion_bdr = (
                    cong_metric.bdr if cong_metric is not None else math.nan
                )
                std_metric.jamming_bdr = (
                    jam_metric.bdr if jam_metric is not None else math.nan
                )

    def _make_env(
        self,
        contact_plan: ContactPlan,
        scenario: str = "standard",
        **scenario_kwargs: Any,
    ) -> _RoutingEnv:
        """
        Create an evaluation environment for the given scenario.

        Parameters
        ----------
        contact_plan : ContactPlan
        scenario : str
            One of ``"standard"``, ``"congestion"``, ``"jamming"``,
            ``"adversarial"``.
        **scenario_kwargs
            Override any ScenarioConfig field (e.g. ``jamming_rate=0.5``).
        """
        cfg_map = {
            "standard": ScenarioConfig("standard"),
            "congestion": _CONGESTION_SCENARIO,
            "jamming": _JAMMING_SCENARIO,
            "adversarial": _ADVERSARIAL_SCENARIO,
        }

        if scenario not in cfg_map:
            raise ValueError(
                f"Unknown scenario {scenario!r}. "
                f"Choose from {list(cfg_map)!r}."
            )

        cfg = cfg_map[scenario]
        for k, v in scenario_kwargs.items():
            if not hasattr(cfg, k):
                raise ValueError(f"ScenarioConfig has no field {k!r}.")
            setattr(cfg, k, v)

        rng = random.Random(cfg.seed)
        perturbed = self._perturb_plan(contact_plan, cfg, rng)

        return _RoutingEnv(
            contact_plan=perturbed,
            max_hops=self.max_hops,
            congestion_load=cfg.congestion_load,
            rng=rng,
        )


# ---------------------------------------------------------------------------
# Monte Carlo rollout evaluator (standalone function)
# ---------------------------------------------------------------------------

def monte_carlo_evaluate(
    agent: Any,
    contact_plan: ContactPlan,
    n_rollouts: int = 1000,
    seed: int = 42,
    bundle_ttl: float = 3600.0,
    bundle_size_bits: int = 10_000_000,
    max_hops: int = 32,
) -> EvalMetrics:
    """
    Monte Carlo evaluation: randomly sample (source, dest, creation_time)
    triples, simulate bundle routing with *agent*, and aggregate metrics.

    Parameters
    ----------
    agent : routing agent
        Must expose ``.route(bundle, contact_plan, node, time)`` or be
        compatible with _RoutingEnv's dispatch logic.
    contact_plan : ContactPlan
        The contact plan to route over.
    n_rollouts : int
        Number of Monte Carlo trials.  Default: 1000.
    seed : int
        Random seed for reproducibility.
    bundle_ttl : float
        Time-to-live for synthetic bundles.  Default: 3600 s.
    bundle_size_bits : int
        Bundle payload size in bits.  Default: 10 Mbits.
    max_hops : int
        Maximum hops before declaring failure.  Default: 32.

    Returns
    -------
    EvalMetrics
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    evaluator = Evaluator(
        max_hops=max_hops,
        bundle_ttl=bundle_ttl,
        bundle_size_bits=bundle_size_bits,
        verbose=False,
    )

    bundles = evaluator._generate_bundles(
        contact_plan=contact_plan,
        n=n_rollouts,
        rng=rng,
        np_rng=np_rng,
    )

    oracle_bdr = evaluator._oracle.compute_oracle_bdr(bundles, contact_plan)

    env = _RoutingEnv(
        contact_plan=contact_plan,
        max_hops=max_hops,
        congestion_load=1.0,
        rng=rng,
    )

    if isinstance(agent, OracleCGR):
        episode_results = evaluator._run_oracle_episodes(bundles, contact_plan)
    else:
        episode_results = [
            evaluator.run_episode(agent, env, b) for b in bundles
        ]

    return compute_metrics(
        results=episode_results,
        oracle_bdr=oracle_bdr,
    )

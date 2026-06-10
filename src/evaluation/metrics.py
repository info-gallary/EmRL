"""
metrics.py
----------
Metric dataclasses and computation utilities for DTN routing evaluation.

Key types
---------
EpisodeResult  : outcome record for a single bundle routing episode.
EvalMetrics    : aggregated statistics over many episodes, including oracle
                 comparison and 95 % confidence intervals.

Key functions
-------------
compute_metrics(results, oracle_bdr)   -> EvalMetrics
compute_confidence_interval(values)    -> (lower, upper)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Per-episode result
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    """
    Outcome of routing a single bundle through a DTN environment.

    Attributes
    ----------
    delivered : bool
        True iff the bundle reached its destination before its deadline.
    delay : float
        End-to-end delay in seconds (measured from creation_time to
        arrival_time).  Meaningful only when ``delivered`` is True;
        set to 0.0 otherwise.
    energy : float
        Cumulative (normalised) energy consumed along the path taken.
        Set to 0.0 when the bundle was not delivered.
    hops : int
        Number of forwarding hops taken.  0 when not delivered.
    drop_reason : str
        Empty string on successful delivery.  One of:
          - ``"ttl_expired"``   : bundle exceeded its time-to-live.
          - ``"no_route"``      : no feasible next-hop was found.
          - ``"congestion"``    : dropped due to queue overflow.
    """

    delivered: bool
    delay: float
    energy: float
    hops: int
    drop_reason: str = ""  # "ttl_expired" | "no_route" | "congestion" | ""

    def __post_init__(self) -> None:
        valid_reasons = {"", "ttl_expired", "no_route", "congestion"}
        if self.drop_reason not in valid_reasons:
            raise ValueError(
                f"drop_reason must be one of {valid_reasons!r}, "
                f"got {self.drop_reason!r}"
            )
        if self.delivered and self.drop_reason:
            raise ValueError(
                "drop_reason must be empty when delivered=True."
            )


# ---------------------------------------------------------------------------
# Aggregated evaluation metrics
# ---------------------------------------------------------------------------

@dataclass
class EvalMetrics:
    """
    Aggregated evaluation metrics over a set of episodes.

    Attributes
    ----------
    bdr : float
        Bundle Delivery Ratio = delivered / total.
    mean_delay : float
        Mean end-to-end delay (seconds) over *delivered* bundles only.
        NaN when no bundle was delivered.
    mean_energy : float
        Mean normalised energy over *delivered* bundles only.
        NaN when no bundle was delivered.
    mean_hops : float
        Mean hop count over *delivered* bundles only.
        NaN when no bundle was delivered.
    oracle_bdr : float
        Oracle upper-bound BDR (passed in, not recomputed here).
    relative_performance : float
        ``bdr / oracle_bdr``.  NaN when ``oracle_bdr`` is 0.
    congestion_bdr : float
        BDR measured under a congestion scenario (filled in by Evaluator;
        defaults to NaN here).
    jamming_bdr : float
        BDR measured under a jamming scenario (filled in by Evaluator;
        defaults to NaN here).
    ci_95 : dict
        95 % confidence intervals keyed by metric name::

            {
                "bdr":        (lower, upper),
                "mean_delay": (lower, upper),
                "mean_energy":(lower, upper),
                "mean_hops":  (lower, upper),
            }

    n_episodes : int
        Total number of episodes used to compute these metrics.
    drop_reason_counts : dict
        Raw counts of each drop reason (including "" for delivered).
    episode_results : list of EpisodeResult, optional
        Per-episode outcomes preserved for paired statistical tests.
        Filled by compute_metrics() when keep_episodes=True.
    """

    bdr: float
    mean_delay: float
    mean_energy: float
    mean_hops: float
    oracle_bdr: float
    relative_performance: float
    congestion_bdr: float
    jamming_bdr: float
    ci_95: Dict[str, Tuple[float, float]]
    n_episodes: int
    drop_reason_counts: Dict[str, int] = field(default_factory=dict)
    episode_results: List["EpisodeResult"] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable one-line summary."""
        rp = (
            f"{self.relative_performance:.3f}"
            if not math.isnan(self.relative_performance)
            else "N/A"
        )
        return (
            f"BDR={self.bdr:.3f} | delay={self.mean_delay:.1f}s | "
            f"energy={self.mean_energy:.4f} | hops={self.mean_hops:.2f} | "
            f"oracle={self.oracle_bdr:.3f} | rel={rp} | n={self.n_episodes}"
        )


# ---------------------------------------------------------------------------
# Confidence interval helper
# ---------------------------------------------------------------------------

def compute_confidence_interval(
    values: List[float],
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Compute a two-sided confidence interval for the mean of *values*.

    Uses the Student-t distribution (appropriate for small samples) via
    ``scipy.stats.t.interval``.  Falls back to ``(NaN, NaN)`` when there
    are fewer than 2 observations.

    Parameters
    ----------
    values : list of float
        Sample values.  Must not be empty.
    confidence : float
        Confidence level in (0, 1).  Default 0.95.

    Returns
    -------
    (lower, upper) : Tuple[float, float]
        Bounds of the confidence interval.
    """
    n = len(values)
    if n < 2:
        if n == 1:
            v = float(values[0])
            return v, v
        return math.nan, math.nan

    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    se = float(sp_stats.sem(arr))  # standard error of the mean

    lower, upper = sp_stats.t.interval(
        confidence=confidence,
        df=n - 1,
        loc=mean,
        scale=se,
    )
    return float(lower), float(upper)


# ---------------------------------------------------------------------------
# Main aggregation function
# ---------------------------------------------------------------------------

def compute_metrics(
    results: List[EpisodeResult],
    oracle_bdr: float,
    congestion_bdr: float = math.nan,
    jamming_bdr: float = math.nan,
    confidence: float = 0.95,
) -> EvalMetrics:
    """
    Aggregate a list of EpisodeResult objects into EvalMetrics.

    Parameters
    ----------
    results : list of EpisodeResult
        One entry per evaluated bundle.
    oracle_bdr : float
        Oracle upper-bound BDR (computed externally via OracleCGR).
    congestion_bdr : float, optional
        BDR from congestion scenario.  NaN if not yet evaluated.
    jamming_bdr : float, optional
        BDR from jamming scenario.  NaN if not yet evaluated.
    confidence : float
        Confidence level for CIs.  Default 0.95.

    Returns
    -------
    EvalMetrics
    """
    if not results:
        nan = math.nan
        ci_empty: Dict[str, Tuple[float, float]] = {
            "bdr": (nan, nan),
            "mean_delay": (nan, nan),
            "mean_energy": (nan, nan),
            "mean_hops": (nan, nan),
        }
        return EvalMetrics(
            bdr=nan,
            mean_delay=nan,
            mean_energy=nan,
            mean_hops=nan,
            oracle_bdr=oracle_bdr,
            relative_performance=nan,
            congestion_bdr=congestion_bdr,
            jamming_bdr=jamming_bdr,
            ci_95=ci_empty,
            n_episodes=0,
        )

    n = len(results)
    n_delivered = sum(1 for r in results if r.delivered)
    bdr = n_delivered / n

    delivered_results = [r for r in results if r.delivered]

    if delivered_results:
        delays = [r.delay for r in delivered_results]
        energies = [r.energy for r in delivered_results]
        hops = [float(r.hops) for r in delivered_results]

        mean_delay = float(np.mean(delays))
        mean_energy = float(np.mean(energies))
        mean_hops = float(np.mean(hops))
    else:
        delays, energies, hops = [], [], []
        mean_delay = math.nan
        mean_energy = math.nan
        mean_hops = math.nan

    # Per-episode delivered flags for BDR CI
    delivered_flags = [1.0 if r.delivered else 0.0 for r in results]

    ci_95: Dict[str, Tuple[float, float]] = {
        "bdr": compute_confidence_interval(delivered_flags, confidence),
        "mean_delay": compute_confidence_interval(delays, confidence)
        if delays
        else (math.nan, math.nan),
        "mean_energy": compute_confidence_interval(energies, confidence)
        if energies
        else (math.nan, math.nan),
        "mean_hops": compute_confidence_interval(hops, confidence)
        if hops
        else (math.nan, math.nan),
    }

    relative_performance = (
        bdr / oracle_bdr
        if oracle_bdr > 0
        else math.nan
    )

    # Count drop reasons
    drop_reason_counts: Dict[str, int] = {}
    for r in results:
        key = r.drop_reason if not r.delivered else "delivered"
        drop_reason_counts[key] = drop_reason_counts.get(key, 0) + 1

    return EvalMetrics(
        bdr=bdr,
        mean_delay=mean_delay,
        mean_energy=mean_energy,
        mean_hops=mean_hops,
        oracle_bdr=oracle_bdr,
        relative_performance=relative_performance,
        congestion_bdr=congestion_bdr,
        jamming_bdr=jamming_bdr,
        ci_95=ci_95,
        n_episodes=n,
        drop_reason_counts=drop_reason_counts,
        episode_results=list(results),
    )

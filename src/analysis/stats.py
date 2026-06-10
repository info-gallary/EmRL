"""
Statistical analysis for the EmRL evaluation suite.

Provides:
  - paired_t_test(): paired t-test between two agents on per-episode BDR
  - cohen_d():      effect size for paired samples
  - holm_correction(): family-wise error rate correction for multiple comparisons
  - format_significance(): human-readable p-value formatting
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats


@dataclass
class PairedTestResult:
    """Result of a paired statistical comparison."""

    agent_a: str
    agent_b: str
    n_pairs: int
    mean_a: float
    mean_b: float
    mean_diff: float            # mean(a - b)
    std_diff: float
    t_statistic: float
    p_value: float
    cohens_d: float             # paired effect size
    ci_95_lo: float             # CI on mean_diff
    ci_95_hi: float
    significant: bool           # p < 0.05
    significant_holm: bool      # significant after Holm-Bonferroni

    def summary(self) -> str:
        marker = "***" if self.p_value < 0.001 else \
                 "**"  if self.p_value < 0.01  else \
                 "*"   if self.p_value < 0.05  else "ns"
        return (f"{self.agent_a} vs {self.agent_b}: "
                f"Δ={self.mean_diff:+.4f} [{self.ci_95_lo:+.4f}, {self.ci_95_hi:+.4f}] "
                f"t({self.n_pairs-1})={self.t_statistic:+.2f}, "
                f"p={format_significance(self.p_value)}, d={self.cohens_d:+.2f} {marker}")


def cohen_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cohen's d for paired samples = mean(a - b) / std(a - b).
    Interpretation: |d| ≥ 0.2 = small, ≥ 0.5 = medium, ≥ 0.8 = large.
    """
    diff = a - b
    if len(diff) < 2:
        return 0.0
    s = float(np.std(diff, ddof=1))
    if s < 1e-12:
        return 0.0
    return float(np.mean(diff) / s)


def paired_t_test(
    a_results: np.ndarray,
    b_results: np.ndarray,
    agent_a: str = "A",
    agent_b: str = "B",
) -> PairedTestResult:
    """
    Paired t-test on per-episode binary outcomes (delivered/not delivered)
    or continuous BDR. Both arrays must be the same length and ordered the
    same way (same bundle for index i).

    Args:
        a_results: per-episode outcomes for agent A (0/1 or float in [0,1])
        b_results: per-episode outcomes for agent B
        agent_a, agent_b: names for reporting

    Returns:
        PairedTestResult with full statistics.
    """
    a = np.asarray(a_results, dtype=np.float64)
    b = np.asarray(b_results, dtype=np.float64)
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    n = len(a)
    diff = a - b
    mean_d = float(np.mean(diff))
    std_d = float(np.std(diff, ddof=1))

    if n < 2 or std_d < 1e-12:
        t_stat, p_val = 0.0, 1.0
        ci_lo, ci_hi = mean_d, mean_d
    else:
        t_stat, p_val = stats.ttest_rel(a, b)
        t_stat, p_val = float(t_stat), float(p_val)
        sem = std_d / math.sqrt(n)
        t_crit = stats.t.ppf(0.975, n - 1)
        ci_lo = mean_d - t_crit * sem
        ci_hi = mean_d + t_crit * sem

    return PairedTestResult(
        agent_a=agent_a,
        agent_b=agent_b,
        n_pairs=n,
        mean_a=float(np.mean(a)),
        mean_b=float(np.mean(b)),
        mean_diff=mean_d,
        std_diff=std_d,
        t_statistic=t_stat,
        p_value=p_val,
        cohens_d=cohen_d_paired(a, b),
        ci_95_lo=ci_lo,
        ci_95_hi=ci_hi,
        significant=p_val < 0.05,
        significant_holm=False,    # filled in by holm_correction()
    )


def holm_correction(results: List[PairedTestResult], alpha: float = 0.05) -> None:
    """
    Apply Holm-Bonferroni step-down correction in-place to a list of test results.
    Updates each result's `significant_holm` flag.
    """
    if not results:
        return
    # Sort by p-value (ascending)
    sorted_results = sorted(enumerate(results), key=lambda x: x[1].p_value)
    m = len(results)
    for rank, (orig_idx, r) in enumerate(sorted_results):
        threshold = alpha / (m - rank)
        if r.p_value < threshold:
            results[orig_idx].significant_holm = True
        else:
            # Once we fail to reject, all remaining are also non-significant
            break


def format_significance(p: float) -> str:
    """Format p-value for journal-style reporting."""
    if p < 0.001:
        return "<0.001"
    elif p < 0.01:
        return f"{p:.3f}"
    else:
        return f"{p:.3f}"


def per_episode_outcomes_from_metrics(metrics) -> np.ndarray:
    """
    Extract per-episode delivery outcomes from EvalMetrics.
    Returns a 0/1 array of delivered/not for each episode.
    Falls back to mean BDR replicated if per-episode data unavailable.
    """
    # The Evaluator stores EpisodeResult lists; if EvalMetrics has them, use them
    if hasattr(metrics, 'episode_results') and metrics.episode_results:
        return np.array([1.0 if er.delivered else 0.0 for er in metrics.episode_results])
    # Fallback: simulate from BDR (much less informative)
    n = metrics.n_episodes
    n_delivered = int(round(metrics.bdr * n))
    arr = np.zeros(n)
    arr[:n_delivered] = 1.0
    return arr


def pairwise_comparison_table(
    per_episode_outcomes: Dict[str, np.ndarray],
    primary_agent: str = "EmRL",
) -> List[PairedTestResult]:
    """
    Run paired t-tests between `primary_agent` and every other agent,
    then apply Holm correction.
    """
    if primary_agent not in per_episode_outcomes:
        raise ValueError(f"{primary_agent} not in {list(per_episode_outcomes.keys())}")

    a_outcomes = per_episode_outcomes[primary_agent]
    results = []
    for agent, b_outcomes in per_episode_outcomes.items():
        if agent == primary_agent:
            continue
        if len(a_outcomes) != len(b_outcomes):
            # Truncate to shorter
            min_len = min(len(a_outcomes), len(b_outcomes))
            a_o, b_o = a_outcomes[:min_len], b_outcomes[:min_len]
        else:
            a_o, b_o = a_outcomes, b_outcomes
        r = paired_t_test(a_o, b_o, primary_agent, agent)
        results.append(r)

    holm_correction(results)
    return results

"""Statistical analysis utilities for EmRL evaluation."""
from .stats import (
    PairedTestResult,
    paired_t_test,
    cohen_d_paired,
    holm_correction,
    format_significance,
    per_episode_outcomes_from_metrics,
    pairwise_comparison_table,
)

__all__ = [
    "PairedTestResult",
    "paired_t_test",
    "cohen_d_paired",
    "holm_correction",
    "format_significance",
    "per_episode_outcomes_from_metrics",
    "pairwise_comparison_table",
]

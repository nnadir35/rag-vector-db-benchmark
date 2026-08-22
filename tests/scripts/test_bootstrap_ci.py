"""Unit tests for the bootstrap CI helpers used by the query-level analysis scripts."""

from scripts.paired_bootstrap_delta import paired_delta_ci
from scripts.query_level_bootstrap_ci import bootstrap_ci


def test_bootstrap_ci_brackets_sample_mean() -> None:
    values = [0.8, 0.9, 1.0, 0.7, 0.6, 0.85, 0.95, 1.0, 0.5, 0.75]
    mean, lo, hi = bootstrap_ci(values, n_samples=2000, seed=42)
    assert lo <= mean <= hi
    assert abs(mean - (sum(values) / len(values))) < 1e-9


def test_bootstrap_ci_empty_returns_nan() -> None:
    mean, lo, hi = bootstrap_ci([], n_samples=100, seed=1)
    assert mean != mean  # NaN
    assert lo != lo
    assert hi != hi


def test_bootstrap_ci_deterministic_with_fixed_seed() -> None:
    values = [1.0, 0.0, 1.0, 1.0, 0.0]
    a = bootstrap_ci(values, n_samples=500, seed=7)
    b = bootstrap_ci(values, n_samples=500, seed=7)
    assert a == b


def test_paired_delta_ci_inner_joins_on_query_id() -> None:
    values_a = {"q1": 1.0, "q2": 0.5, "q3": 0.0, "q_only_a": 1.0}
    values_b = {"q1": 0.0, "q2": 0.5, "q3": 1.0, "q_only_b": 0.0}

    result = paired_delta_ci(values_a, values_b, n_samples=500, seed=1)

    assert result["num_paired_queries"] == 3
    assert result["dropped_only_in_a"] == ["q_only_a"]
    assert result["dropped_only_in_b"] == ["q_only_b"]
    # mean delta over q1,q2,q3: (1-0) + (0.5-0.5) + (0-1) = 0 -> mean 0
    assert abs(result["mean_delta"] - 0.0) < 1e-9


def test_paired_delta_ci_no_common_queries_handled_gracefully() -> None:
    result = paired_delta_ci({"qa": 1.0}, {"qb": 0.0}, n_samples=100, seed=1)
    assert result["num_paired_queries"] == 0
    assert result["significant_at_alpha"] is False

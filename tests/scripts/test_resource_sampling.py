"""Unit tests for container-level resource sampling fallback (scripts/benchmark_db.py, item 8).

No Docker required: exercises the graceful "container not found / docker missing" path.
"""

import os

from scripts.benchmark_db import ResourceStats, run_with_resource_stats

# Importing scripts.benchmark_db runs its module-level load_dotenv(), which would
# otherwise leak this repo's .env CHROMA_HOST/CHROMA_PORT into the whole pytest process
# (other test modules import this one during collection, before any test runs) and break
# unrelated in-memory Chroma tests that assume no remote host override. Undo it here.
os.environ.pop("CHROMA_HOST", None)
os.environ.pop("CHROMA_PORT", None)


def test_run_with_resource_stats_without_container_name_is_process_scoped() -> None:
    result, stats = run_with_resource_stats(lambda: 1 + 1)
    assert result == 2
    assert isinstance(stats, ResourceStats)
    assert stats.resource_measurement_scope == "process"
    assert stats.container_peak_memory_mb is None
    assert stats.container_avg_cpu_percent is None
    assert stats.container_status is None


def test_run_with_resource_stats_missing_container_reports_unavailable_not_zero() -> None:
    result, stats = run_with_resource_stats(
        lambda: "done", container_name="definitely_not_a_real_container_xyz123"
    )
    assert result == "done"
    assert stats.resource_measurement_scope == "container"
    assert stats.container_status in ("unavailable", "error")
    # Never a fabricated 0.0 when the container can't be measured.
    assert stats.container_peak_memory_mb is None
    assert stats.container_avg_cpu_percent is None
    # The process-level (client-side) figures are still populated as before.
    assert stats.peak_memory_mb >= 0.0

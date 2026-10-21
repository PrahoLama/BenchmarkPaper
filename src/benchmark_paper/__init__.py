"""BenchmarkPaper package."""

from .audit import main as audit_main
from .benchmark import main as benchmark_main

__all__ = ["audit_main", "benchmark_main"]

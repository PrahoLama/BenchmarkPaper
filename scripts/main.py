"""Entry point for running the BenchmarkPaper pipeline."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark_paper.audit import main as audit_main
from benchmark_paper.benchmark import main as benchmark_main


def main() -> None:
    audit_main()
    benchmark_main()


if __name__ == "__main__":
    main()

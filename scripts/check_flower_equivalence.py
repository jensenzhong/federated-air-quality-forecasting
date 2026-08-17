"""Run the registered two- and twelve-client Flower equivalence gate."""

from pathlib import Path

from aqfl.federated.equivalence import run_equivalence_suite

if __name__ == "__main__":
    report = run_equivalence_suite(Path("artifacts/reports/flower_equivalence.json"))
    print(f"Flower equivalence passed: {len(report['cases'])} cases")

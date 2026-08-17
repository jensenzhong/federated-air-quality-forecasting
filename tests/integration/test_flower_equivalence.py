from __future__ import annotations

from aqfl.federated.equivalence import run_equivalence_suite


def test_sequential_flower_equivalence_suite() -> None:
    report = run_equivalence_suite()
    assert report["status"] == "passed"
    assert len(report["cases"]) == 4

# Engineering validation — 2026-08-17

This file records engineering evidence only. It contains no formal method-performance claim.

| Check | Result |
|---|---|
| Isolated installation | PASS (`uv sync --extra dev --frozen`; PyTorch 2.7.0+cpu, Flower 1.22.0, pandas 2.2.3) |
| Ruff | PASS |
| Mypy | PASS (38 source files) |
| Core tests | PASS (62) |
| Core coverage | 91.30% |
| Full UCI data test | PASS (12 stations, 420,768 rows) |
| Prepared cache | PASS (220 files, 1,236,279,706 bytes) |
| Two-client one-round Flower protocol | PASS (in-memory Grid integration) |
| Sequential Flower equivalence gate | PASS (2/12 clients × FedAvg/FedProx; max array/metric diff 0.0; report `flower_equivalence.json`) |
| Plaintext `sk-*` scan | PASS (0 hits in tracked text/config/code/docs) |
| Legacy deletion verification | PASS (0/9 targets remain) |
| Report dependency and validated-only filter | PASS (`tabulate` locked; empty registry emits no-results report; registry status is authoritative) |
| Screening queue | PASS (13 parameterized jobs; repeated planning is idempotent) |
| Flower launcher argument parsing | PASS (quoted two-field `node-config` accepted by Flower 1.22) |
| Flower failed-gate process safety | PASS (4.13 GB available vs 10 GB required; 0 Flower processes started) |
| 12-client full-concurrency resource gate | BLOCKED: 16 logical CPUs, latest launch snapshot 4.13 GB available RAM; 10 GB required |
| 12-client sequential FedProx feasibility | PASS, nonformal (1 round, 12/12 stations, 1.9 min, observed training process approximately 0.40 GB; run `fedprox-42-20260817T013258Z-8151de3`) |
| 12-client strict sequential Flower resource benchmark | PASS, nonformal (3 rounds, 328.94 s, peak RSS 0.392 GB, minimum available memory 1.329 GB; run `fedprox-42-20260817T020304Z-8151de3`) |
| Native Windows Ray simulation | UNAVAILABLE (object store startup timeout before client training); not retained as a project dependency |
| Validation-only Persistence smoke | Executed, then marked `invalid` for formal reporting because protocol was not frozen and split was not test |
| Validation-only baseline smoke | Seasonal Naive, Centralized GRU (1 epoch), and Local GRU (1 epoch) completed with all required artifacts; formal validator correctly rejects each |

Formal 12-client execution, model selection, five-seed comparisons, ablations, robustness and statistical claims remain `TBD`.

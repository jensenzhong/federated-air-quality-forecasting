# Engineering validation — 2026-08-13

This file records engineering evidence only. It contains no formal method-performance claim.

| Check | Result |
|---|---|
| Editable installation | PASS |
| Ruff | PASS |
| Mypy | PASS (38 source files) |
| Core tests | PASS (51) |
| Core coverage | 92.81% |
| Full UCI data test | PASS (12 stations, 420,768 rows) |
| Prepared cache | PASS (220 files, 1,236,279,706 bytes) |
| Two-client one-round Flower protocol | PASS (in-memory Grid integration) |
| Plaintext `sk-*` scan | PASS (0 hits in tracked text/config/code/docs) |
| Legacy deletion verification | PASS (0/9 targets remain) |
| 12-client full-concurrency resource gate | BLOCKED: 16 logical CPUs, approximately 1.33 GB available RAM; 10 GB required |
| Validation-only Persistence smoke | Executed, then marked `invalid` for formal reporting because protocol was not frozen and split was not test |

Formal 12-client execution, model selection, five-seed comparisons, ablations, robustness and statistical claims remain `TBD`.

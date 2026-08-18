# Continual SecAgg+ smoke evidence (2026-08-18)

This is a nonformal engineering/validation smoke, not a test-set result and not a multi-seed claim.

## Run

- Command: `python scripts/run_flower_sequential.py --method pafa_rule --seed 42 --rounds 12 --continual --continual-task-count 11 --continual-base-rounds 1 --continual-rounds-per-task 1 --evaluation-split val`
- Run artifact: `pafa_rule-42-20260818T043910Z-38c6a2f`
- Cohort: 12/12 clients in every round; 0 failures
- Protocol: Flower SecAgg+ four-stage workflow; client-local continual ledger; aggregate-only final task matrix
- Evaluation split: validation only

## Aggregate-only continual summary

The server decoded the fixed-size task matrix only after the final task and only after the minimum secure cohort gate:

| Metric | Value |
|---|---:|
| Average forgetting (AF) | -0.7457669576 |
| Average plasticity (AP) | 13.9780304649 |
| Average performance (AvgPerf) | 13.3000605034 |
| Task count | 11 |

The run also completed with `best_validation_macro_mae=13.0799293518` at round 10. These values are engineering evidence that the continual path executes; they are not a claim of superiority over the supplied benchmark or over FedAdam.

## Privacy/protocol interpretation

- Raw rows, residuals, timestamps, task scores, and the local ledger remained in each client context.
- Intermediate task-boundary rounds sent a zero continual-matrix slot; only the final task sent the fixed-length matrix through SecAgg+.
- The benchmark-compatible unobserved upper triangle is zero-filled locally at encoding time; observed lower-triangle cells must be complete.
- The server artifact contains cohort summaries only; no client metrics or station identifiers are persisted.
- Differential privacy is not enabled or claimed.


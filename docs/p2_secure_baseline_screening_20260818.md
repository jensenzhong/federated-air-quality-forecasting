# P2 secure PAFA development screening (2026-08-18)

This document records nonformal, validation-only, single-seed development runs. It is not a privacy certification, a test result, or a paper claim. All runs used the low-memory sequential Flower runtime with 12/12 clients and the SecAgg+ PAFA path.

## Comparable runs

| Method | Run ID | Best round | Best station-macro MAE ↓ | High-pollution MAE ↓ | Mean probe fraction | Mean local train seconds | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| PAFA FedProx | `pafa_fedprox-42-20260818T011153Z-28f691b` | 10 | 10.7875 | 29.3753 | 0.000 | 7.6330 | valid development baseline |
| PAFA FedAdam (`server_lr=0.1`) | `pafa_fedadam-42-20260818T015126Z-28f691b` | 10 | 10.2817 | 31.2001 | 0.000 | 8.3265 | valid development baseline |
| PAFA contextual bandit | `pafa_bandit-42-20260818T021014Z-28f691b` | 10 | 10.5097 | 29.8761 | 0.778 | 8.4404 | valid development control |
| PAFA rule proposer | `pafa_rule-42-20260818T022928Z-28f691b` | 9 | 10.7087 | 29.8815 | 0.800 | 8.3422 | valid development control |

Directive compliance was 1.0 on the valid runs, clipping-violation rate was 0, and each SecAgg+ round reported 12 results and 0 failures. The probe/action statistics are aggregate cohort summaries; no client identity or client trajectory was persisted on the server.

## Invalidated run

`pafa_fedadam-42-20260818T013129Z-28f691b` is excluded because the implementation hard-coded the secure server learning rate to `1.0` and ignored the runner value. The code now passes `--server-lr` through `run_secure_pafa`, and the corrected run above explicitly records `server_learning_rate=0.1`.

## Interpretation

- Bandit and Rule improve the mean MAE relative to the PAFA FedProx development run, but neither beats corrected PAFA FedAdam on mean MAE.
- Bandit and Rule have lower high-pollution MAE than corrected FedAdam, but this is a trade-off, not a uniformly superior result.
- Probe fractions near 0.8 mean that a budget-matched baseline and a probe-cost sensitivity analysis are still required.
- The active goal's 1% improvement and paired-statistics gate is not met by this single-seed screening. No formal multi-seed or test run is authorized yet.

## Next development gate

1. Add or verify an aggregate-only, probe-budget-matched FedProx control.
2. Tune the contextual controller against the corrected FedAdam baseline without changing the privacy boundary.
3. Re-run the resulting candidate on a frozen, unseen confirmation split; only then consider multi-seed statistics.
4. Run PAFA LLM only with a loopback/on-prem endpoint. The configured loopback port was not listening during this screening, so no LLM run was started.


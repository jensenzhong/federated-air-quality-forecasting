# P2 secure PAFA development screening (2026-08-18)

This document records nonformal, validation-only, single-seed development runs. It is not a privacy certification, a test result, or a paper claim. All runs used the low-memory sequential Flower runtime with 12/12 clients and the SecAgg+ PAFA path.

The LLM transport was separately exercised with the development-only localhost stub in `scripts/local_llm_stub_server.py`; that run is a protocol smoke, not an LLM quality result.

## Comparable runs

| Method | Run ID | Best round | Best station-macro MAE ↓ | High-pollution MAE ↓ | Mean probe fraction | Mean local train seconds | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| PAFA FedProx | `pafa_fedprox-42-20260818T011153Z-28f691b` | 10 | 10.7875 | 29.3753 | 0.000 | 7.6330 | valid development baseline |
| PAFA FedAdam (`server_lr=0.1`) | `pafa_fedadam-42-20260818T015126Z-28f691b` | 10 | 10.2817 | 31.2001 | 0.000 | 8.3265 | valid development baseline |
| PAFA FedProx (probe-budget matched) | `pafa_fedprox_budget_matched-42-20260818T031349Z-323da88` | 10 | 10.5280 | 29.6761 | 1.000 | 8.8835 | valid budget control |
| PAFA contextual bandit | `pafa_bandit-42-20260818T021014Z-28f691b` | 10 | 10.5097 | 29.8761 | 0.778 | 8.4404 | valid development control |
| PAFA rule proposer | `pafa_rule-42-20260818T022928Z-28f691b` | 9 | 10.7087 | 29.8815 | 0.800 | 8.3422 | valid development control |
| PAFA probe oracle | `pafa_probe_oracle-42-20260818T033649Z-37754fb` | 9 | 10.5626 | 29.3184 | 1.000 | 8.1681 | mechanism upper-bound control |

The controlled hybrid smoke `pafa_bandit_fedadam-42-20260818T025932Z-90200bf` was stopped after 3 rounds: best macro MAE `12.4955` (round 2), versus `11.4939` for the static FedAdam run at round 3. It is retained as a rejected development candidate; no 10-round result is claimed.

The actual local LLM proposer transport smoke `pafa_llm-42-20260818T040611Z-c8e2db6` completed one validation round with 12/12 clients, `source_rate_llm=1.0`, directive compliance `1.0`, and no client metrics persisted on the server. It used only the localhost development stub and must not be interpreted as evidence about a real LLM model.

The earlier `pafa_llm-42-20260818T040320Z-c8e2db6` attempt is invalid because the host HTTP proxy intercepted localhost and returned 502; it is excluded from all evidence.

Directive compliance was 1.0 on the valid runs, clipping-violation rate was 0, and each SecAgg+ round reported 12 results and 0 failures. The probe/action statistics are aggregate cohort summaries; no client identity or client trajectory was persisted on the server.

## Invalidated run

`pafa_fedadam-42-20260818T013129Z-28f691b` is excluded because the implementation hard-coded the secure server learning rate to `1.0` and ignored the runner value. The code now passes `--server-lr` through `run_secure_pafa`, and the corrected run above explicitly records `server_learning_rate=0.1`.

## Interpretation

- Bandit and Rule improve the mean MAE relative to the PAFA FedProx development run, but neither beats corrected PAFA FedAdam on mean MAE.
- Against the probe-budget-matched FedProx control, Bandit improves macro MAE by only about 0.17% on this seed and is worse on high-pollution MAE; this is below the pre-registered 1% gate and does not establish an agentic gain.
- Bandit and Rule have lower high-pollution MAE than corrected FedAdam, but this is a trade-off, not a uniformly superior result.
- The probe oracle reaches the best high-pollution MAE (`29.3184`) but still misses FedAdam on macro MAE; the frozen action space therefore appears more useful for tail-risk control than for average-accuracy gains.
- Probe fractions near 0.8 mean that a budget-matched baseline and a probe-cost sensitivity analysis are still required.
- The active goal's 1% improvement and paired-statistics gate is not met by this single-seed screening. No formal multi-seed or test run is authorized yet.

## Next development gate

The aggregate-only probe-budget-matched FedProx control has now been verified, and the reproducibility contract is frozen in `docs/13_v2_reproducibility_contract.md`. The remaining gates are:

1. Freeze an unseen confirmation split and establish institutional/client-process isolation without weakening SecAgg+.
2. Run the real `pafa_llm` proposer only against an approved loopback/on-prem endpoint; the localhost stub smoke is transport evidence, not LLM quality evidence.
3. Re-run the frozen candidate and strongest compatible baselines on the confirmation split; only if the 1%/CI/tail gates hold may the five-seed experiment begin.
4. If those gates fail, stop the accuracy-superiority claim and report the validated tail-risk/fairness conclusion instead.

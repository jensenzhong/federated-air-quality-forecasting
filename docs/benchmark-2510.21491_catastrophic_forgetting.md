# Benchmark protocol note: federated continual time-series forgetting

Source: Hallak & Kem, “Benchmarking Catastrophic Forgetting Mitigation Methods in Federated Time Series Forecasting”, arXiv:2510.21491v1 (2025), accepted at FLTA 2025. Stable sources: [arXiv abstract](https://arxiv.org/abs/2510.21491) and [HTML full text](https://ar5iv.labs.arxiv.org/html/2510.21491). The authors’ implementation is linked from the paper as [cf-federated-timeseries](https://github.com/khaledhallakk/cf-federated-timeseries).

The supplied notebook source was audited locally; the exact windows are frozen in `aqfl/data/continual_schedule.py` and the audit is recorded in `docs/benchmark-source-audit-2510.21491.md`.

## What is actually benchmarked

- Beijing Multi-Site Air Quality, 12 stations as clients.
- One offline base task covering March 2013–March 2014, followed by 11 chronological seasonal tasks through early 2017.
- Each task is disjoint and becomes unavailable after training unless a method explicitly stores replay data.
- LSTM, 12 hourly lags, six-step horizon, temporal 80/20 train/test split.
- Base phase: FedAvg, 500 rounds, one local epoch; each continual task: 30 rounds, one local epoch.
- Five random trials; method/target-specific grid search.
- Metrics use the task-performance matrix: average forgetting (AF), average plasticity (AP), and final average performance (AvgPerf), with lower values reported as better. The paper reports RMSE scaled by (10^3).

Reported Table II reference values (not results from this repository):

| Method | Temperature AF / AP / AvgPerf | PM2.5 AF / AP / AvgPerf | Wind Speed AF / AP / AvgPerf |
|---|---:|---:|---:|
| Replay | -0.63 / 42.60 / 42.35 | -0.03 / 79.19 / 75.31 | -0.16 / 135.61 / 132.21 |
| LwF (paper label: KD) | -0.06 / 43.62 / 43.91 | 0.02 / 80.43 / 76.88 | 0.08 / 136.74 / 133.28 |
| Online EWC | 0.07 / 44.21 / 44.49 | 0.04 / 79.44 / 75.63 | 0.08 / 137.03 / 133.58 |
| SI | 0.01 / 43.64 / 43.93 | 0.02 / 79.82 / 76.03 | 0.08 / 137.10 / 133.75 |

These values are a public-paper reference only. They cannot be used as a direct win/loss comparison against AQ-MAS-FL until the task schedule, target, horizon, model, scaling, communication budget, and five-seed protocol are reproduced.

## Compatibility audit for AQ-MAS-FL

The paper’s replay baseline stores representative historical samples locally. That is not a server privacy violation, but it is a different local-memory contract from a replay-free method and must be budgeted explicitly. The benchmark also reports client/task performance matrices, which cannot be sent to a server as single-client metrics under this project’s SecAgg+ boundary. AQ-MAS-FL therefore needs a secure aggregate-only continual evaluator: clients retain task scores locally and contribute only fixed-length task-indexed aggregate sums after a minimum cohort gate.

The current AQ-MAS-FL model is a GRU PM2.5 one-step predictor, while the paper uses an LSTM and three targets with a six-step horizon. Therefore the paper is an external continual-learning benchmark, not a directly comparable numerical baseline for the current static protocol. A valid comparison requires a separate benchmark adapter with:

1. frozen (T_0 + T_1\ldots T_{11}) seasonal manifests and hashes;
2. a six-step target/model configuration matching the paper, or an explicit application-specific claim limited to the current PM2.5 task;
3. equal 500-round base phase, 30-round/task continual phase, one local epoch, five seeds, and equal local-memory/probe budgets;
4. local task-performance matrices and secure cohort summaries for AF/AP/AvgPerf;
5. separate reporting of raw-data replay, parameter-only consolidation, and the proposed blackboard-coordinated controller.

## Research opportunity used in this project

The paper identifies Replay as the strongest reported method but also notes its memory and privacy cost. The defensible AQ-MAS-FL target is therefore not “LLM beats a reported number”; it is:

> under the same continual task schedule and local compute/memory budget, use private parameter consolidation plus SecAgg+ blackboard coordination to approach or exceed Replay-level retention without transmitting raw samples, and verify whether the LLM proposer adds value over the same-action-space bandit.

Until the adapter and frozen manifests exist, all benchmark-comparison results remain `TBD`.

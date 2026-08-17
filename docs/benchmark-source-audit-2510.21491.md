# Supplied benchmark source audit

Audited source: `C:/Users/23079/Downloads/khaledhallakk-cf-federated-timeseries-f20147b/khaledhallakk-cf-federated-timeseries-f20147b/CF-Federated-Time-Series.ipynb`.

The notebook is a single executable-style artifact rather than an installable package. Its source cells confirm:

- `TARGET_COL` defaults to `WSPM`, with `PM2.5` and `TEMP` as alternatives;
- `N_LAGS=12`, `PRED_LEN=6`, robust global 1st/99th-percentile normalization;
- base data `2013-05-01` through `2014-05-03 23:00`, one base-test day `2014-05-04`;
- exactly 11 continual windows, now frozen in `aqfl/data/continual_schedule.py`;
- one local LSTM predictor, FedAvg equal client weighting, 25 notebook rounds by default (the paper text describes 30 as the paper-grade protocol);
- Replay uses KMeans-selected representative windows, while EWC/Online-EWC use diagonal gradient-square Fisher estimates, KD uses the prior local predictor, and SI accumulates path contributions;
- the notebook computes lower-is-better legacy AF/AP/AvgPerf from a task checkpoint matrix.

The notebook also contains non-portable execution assumptions and a conversion failure under the current nbconvert toolchain due to indentation in a code cell. We therefore use it as a protocol oracle and baseline implementation reference, not as code to import into the secure Flower runtime.

The supplied source does not implement SecAgg+, cohort minimum-size gating, or aggregate-only task-matrix release. Any benchmark adapter in AQ-MAS-FL must keep task scores and replay buffers client-local and release only fixed-length SecAgg+ aggregates.

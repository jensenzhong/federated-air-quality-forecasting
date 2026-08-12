from __future__ import annotations

import numpy as np

from aqfl.data.preprocessing import fit_global_scaler, fit_station_imputer, transform_station


def test_mutating_future_does_not_change_train_fitted_states(raw_frame) -> None:
    train_mask = np.arange(len(raw_frame)) < 48
    left_state = fit_station_imputer("A", raw_frame.loc[train_mask])
    left_features, _ = transform_station(raw_frame, left_state)
    left_scaler = fit_global_scaler({"A": left_features}, {"A": raw_frame}, {"A": train_mask})

    mutated = raw_frame.copy()
    mutated.loc[~train_mask, "PM2.5"] = 999999
    mutated.loc[~train_mask, "TEMP"] = -999
    right_state = fit_station_imputer("A", mutated.loc[train_mask])
    right_features, _ = transform_station(mutated, right_state)
    right_scaler = fit_global_scaler({"A": right_features}, {"A": mutated}, {"A": train_mask})

    assert left_state == right_state
    assert left_scaler == right_scaler

# Data policy

Raw Beijing CSV files are not copied into this repository. Set `BEIJING_AQ_DATA_DIR` to a directory containing the 12 `PRSA_Data_*.csv` files. The preparation command records SHA-256 hashes in `data/manifest.json` and writes derived, ignored caches under `data/cache/`.

The loader deliberately ignores unrelated `data.csv` and `test.csv` files.

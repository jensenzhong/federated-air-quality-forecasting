$ErrorActionPreference = 'Stop'
if (-not $env:BEIJING_AQ_DATA_DIR) {
    throw 'Set BEIJING_AQ_DATA_DIR before preparing data.'
}
python -m aqfl.data.prepare --config configs/base.yaml

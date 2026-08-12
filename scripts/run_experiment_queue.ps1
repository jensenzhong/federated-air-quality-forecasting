param(
    [ValidateSet('smoke','screening','single_seed_full','formal_five_seed','ablation','robustness')]
    [string]$Stage = 'smoke',
    [switch]$Execute
)

$ErrorActionPreference = 'Stop'
$argsList = @('-m','aqfl.experiments.sweep','--plan','configs/experiments/formal.yaml','--stage',$Stage)
if ($Execute) { $argsList += '--execute' }
python @argsList

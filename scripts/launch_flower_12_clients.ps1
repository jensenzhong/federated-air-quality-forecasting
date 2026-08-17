param(
    [ValidateSet('fedavg','fedprox','qfedavg','fedadam','rule_mas','mas_llm','mas_llm_dynamic_only','mas_llm_no_fairness','fedprox_budget_matched')]
    [string]$Method = 'fedprox',
    [int]$Seed = 42,
    [int]$Rounds = 1,
    [string]$BudgetTrace = ''
)

$ErrorActionPreference = 'Stop'

$venvScripts = Join-Path (Resolve-Path '.').Path '.venv\Scripts'
$python = Join-Path $venvScripts 'python.exe'
$superlinkExecutable = Join-Path $venvScripts 'flower-superlink.exe'
$supernodeExecutable = Join-Path $venvScripts 'flower-supernode.exe'
$flwrExecutable = Join-Path $venvScripts 'flwr.exe'
foreach ($executable in @($python, $superlinkExecutable, $supernodeExecutable, $flwrExecutable)) {
    if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
        throw "Required project executable is missing: $executable. Run uv sync --extra dev --frozen."
    }
}

$snapshot = & $python -c "import json; from aqfl.config import load_config; from aqfl.federated.resources import enforce_resource_gate; print(json.dumps(enforce_resource_gate(load_config('configs/base.yaml'))))"
if ($LASTEXITCODE -ne 0) {
    throw "Resource gate failed with exit code $LASTEXITCODE; Flower processes were not started."
}
Write-Host "Resource gate passed: $snapshot"

$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'

$logs = Join-Path (Resolve-Path '.').Path 'artifacts\flower_runtime'
New-Item -ItemType Directory -Path $logs -Force | Out-Null

$superlink = Start-Process $superlinkExecutable -ArgumentList '--insecure','--fleet-api-type','grpc-rere','--fleet-api-address','127.0.0.1:9092','--control-api-address','127.0.0.1:9093' -PassThru -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logs 'superlink.out.log') -RedirectStandardError (Join-Path $logs 'superlink.err.log')
Start-Sleep -Seconds 3

$nodes = @()
for ($i = 0; $i -lt 12; $i++) {
    $nodeConfig = "partition-id=$i num-partitions=12"
    # Start-Process joins ArgumentList entries into one command line. Quote the
    # complete node-config value so both key/value pairs reach Flower as one arg.
    $nodeArguments = "--insecure --superlink 127.0.0.1:9092 --node-config `"$nodeConfig`""
    $nodes += Start-Process $supernodeExecutable -ArgumentList $nodeArguments -PassThru -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logs "supernode_$i.out.log") -RedirectStandardError (Join-Path $logs "supernode_$i.err.log")
}

Write-Host "Started SuperLink PID $($superlink.Id) and 12 SuperNodes."
$runConfig = "method=$Method seed=$Seed num-server-rounds=$Rounds"
if ($Method -eq 'fedprox_budget_matched') {
    if (-not $BudgetTrace) { throw 'FedProx budget matching requires -BudgetTrace.' }
    $runConfig = "$runConfig budget-trace='$BudgetTrace'"
}
$processRecords = @(
    [PSCustomObject]@{ Role = 'SuperLink'; Partition = ''; Id = $superlink.Id }
)
for ($i = 0; $i -lt $nodes.Count; $i++) {
    $processRecords += [PSCustomObject]@{ Role = 'SuperNode'; Partition = $i; Id = $nodes[$i].Id }
}
$processRecords | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $logs 'processes.json') -Encoding UTF8
Write-Host "Running: flwr run . local-12 --run-config `"$runConfig`""
try {
    & $flwrExecutable run . local-12 --run-config $runConfig
    if ($LASTEXITCODE -ne 0) { throw "Flower run failed with exit code $LASTEXITCODE" }
}
finally {
    foreach ($process in @($nodes) + @($superlink)) {
        if (-not $process.HasExited) { Stop-Process -Id $process.Id }
    }
}

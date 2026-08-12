$ErrorActionPreference = 'Stop'

ruff check .
if ($LASTEXITCODE -ne 0) { throw 'ruff failed' }

mypy aqfl
if ($LASTEXITCODE -ne 0) { throw 'mypy failed' }

pytest -m 'not full_data' --cov=aqfl --cov-report=term-missing
if ($LASTEXITCODE -ne 0) { throw 'core tests failed' }

if ($env:BEIJING_AQ_DATA_DIR) {
    pytest -m full_data -q
    if ($LASTEXITCODE -ne 0) { throw 'full-data tests failed' }
}

$scanRoots = @('aqfl','configs','docs','scripts','README.md','PROJECT_STATUS.md','ccfa.yaml')
$files = foreach ($root in $scanRoots) {
    if (Test-Path -LiteralPath $root -PathType Leaf) {
        Get-Item -LiteralPath $root
    } else {
        Get-ChildItem -LiteralPath $root -Recurse -File
    }
}
$hits = $files | Where-Object {
    $_.Extension -in '.py','.yaml','.yml','.json','.md','.ps1','.txt','.toml'
} | Select-String -Pattern 'sk-[A-Za-z0-9]{16,}' -AllMatches -ErrorAction SilentlyContinue
if ($hits) { throw "Secret scan found $(@($hits).Count) potential plaintext keys." }

python -c "from aqfl.config import load_config; from aqfl.federated.resources import resource_snapshot; print(resource_snapshot())"
Write-Host 'Audit passed. Formal Flower resource gate is checked separately before launch.'

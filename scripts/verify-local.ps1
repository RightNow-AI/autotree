[CmdletBinding()]
param(
    [switch]$Modeling
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $PSScriptRoot
$results = [System.Collections.Generic.List[object]]::new()

function Show-Summary {
    Write-Host ""
    Write-Host "Verification summary"
    Write-Host "--------------------"
    foreach ($result in $results) {
        Write-Host ("{0,-32} {1}" -f $result.Name, $result.Status)
    }
}

function Invoke-Native {
    $command = [string]$args[0]
    $arguments = [string[]]$args[1..($args.Count - 1)]

    & $command @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $command $($arguments -join ' ')"
    }
}

function Invoke-Gate {
    param(
        [Parameter(Mandatory)] [string]$Name,
        [Parameter(Mandatory)] [scriptblock]$Action
    )

    Write-Host ""
    Write-Host "==> $Name"
    try {
        & $Action
        $results.Add([pscustomobject]@{ Name = $Name; Status = "PASS" })
    }
    catch {
        $results.Add([pscustomobject]@{ Name = $Name; Status = "FAIL" })
        Show-Summary
        Write-Error $_
        exit 1
    }
}

Push-Location $repoRoot
try {
    Invoke-Gate "core: create Python 3.12 env" {
        Push-Location core
        try { Invoke-Native uv venv --python 3.12 --clear }
        finally { Pop-Location }
    }
    Invoke-Gate "core: install dev dependencies" {
        Push-Location core
        try { Invoke-Native uv pip install -e ".[dev]" }
        finally { Pop-Location }
    }
    Invoke-Gate "core: KV + kernel tests" {
        Push-Location core
        try { Invoke-Native uv run --no-sync pytest -q tests/kv tests/kernels }
        finally { Pop-Location }
    }
    Invoke-Gate "scheduler: fmt" {
        Push-Location scheduler
        try { Invoke-Native cargo fmt --check }
        finally { Pop-Location }
    }
    Invoke-Gate "scheduler: clippy" {
        Push-Location scheduler
        try { Invoke-Native cargo clippy --all-targets "--" -D warnings }
        finally { Pop-Location }
    }
    Invoke-Gate "scheduler: test" {
        Push-Location scheduler
        try { Invoke-Native cargo test }
        finally { Pop-Location }
    }
    Invoke-Gate "scheduler: python feature" {
        Push-Location scheduler
        try { Invoke-Native cargo check --features python }
        finally { Pop-Location }
    }

    if (Test-Path -LiteralPath (Join-Path $repoRoot "serve")) {
        Invoke-Gate "serve: create Python 3.12 env" {
            Push-Location serve
            try { Invoke-Native uv venv --python 3.12 --clear }
            finally { Pop-Location }
        }
        Invoke-Gate "serve: install dev dependencies" {
            Push-Location serve
            try { Invoke-Native uv pip install -e ".[dev]" }
            finally { Pop-Location }
        }
        Invoke-Gate "serve: tests" {
            Push-Location serve
            try { Invoke-Native uv run --no-sync pytest -q }
            finally { Pop-Location }
        }
    }
    else {
        Write-Host ""
        Write-Host "==> serve: tests"
        Write-Host "serve/ is absent; skipping the serve gate until that package merges."
        $results.Add([pscustomobject]@{ Name = "serve: tests"; Status = "SKIP (serve/ absent)" })
    }

    Invoke-Gate "workflow YAML parse" {
        $code = "import pathlib,yaml; files=[pathlib.Path('.github/workflows/ci.yml'),pathlib.Path('.github/workflows/gpu-parity.yml')]; [yaml.safe_load(path.read_text(encoding='utf-8')) for path in files]; print('Parsed workflow YAML:', ', '.join(map(str, files)))"
        Invoke-Native uv run --with pyyaml python -c $code
    }

    if ($Modeling) {
        Invoke-Gate "modeling: install dependencies" {
            Push-Location core
            try { Invoke-Native uv pip install -e ".[dev,modeling]" }
            finally { Pop-Location }
        }
        Invoke-Gate "modeling: core tests" {
            Push-Location core
            try {
                $env:HF_HOME = Join-Path ([System.IO.Path]::GetTempPath()) "autotree-huggingface"
                Invoke-Native uv run --no-sync pytest -q tests
            }
            finally { Pop-Location }
        }
    }
    else {
        $results.Add([pscustomobject]@{ Name = "modeling: scheduled/manual"; Status = "SKIP (use -Modeling)" })
    }

    Show-Summary
}
finally {
    Pop-Location
}

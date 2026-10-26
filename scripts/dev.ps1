# Windows equivalent of the Makefile.  Usage: .\scripts\dev.ps1 setup|init|run|zipnet-worker|clean|reset-data
param([Parameter(Mandatory = $true)][ValidateSet("setup", "init", "run", "zipnet-worker", "clean", "reset-data")][string]$Task)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$venvPython = Join-Path $root ".venv\Scripts\python.exe"
$env:PYTHONUTF8 = "1"

switch ($Task) {
    "setup" {
        python -m venv .venv
        & $venvPython -m pip install --upgrade pip
        & $venvPython -m pip install -r requirements.txt
        if (-not (Test-Path .env)) { Copy-Item .env.example .env }
        Write-Host "Edit .env and set LOCATENET_SECRET_KEY, then run: .\scripts\dev.ps1 init"
    }
    "init" { & $venvPython init_system.py }
    "run" { & $venvPython app.py }   # also starts the ZIPNET sync worker when ZIPNET_SYNC_ENABLED=1
    "zipnet-worker" { & $venvPython zipnet_worker.py }
    "clean" {
        Get-ChildItem -Recurse -Directory -Filter __pycache__ |
            Where-Object { $_.FullName -notlike "*\.venv\*" } |
            Remove-Item -Recurse -Force
        Remove-Item -Recurse -Force .pytest_cache, .ruff_cache -ErrorAction SilentlyContinue
    }
    "reset-data" {
        $answer = Read-Host "This deletes ALL local data. Type 'yes' to continue"
        if ($answer -eq "yes") { & $venvPython clean_system.py }
    }
}

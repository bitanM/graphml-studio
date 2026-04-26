$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r .\requirements.txt
& ".\.venv\Scripts\python.exe" .\evaluate_cora.py

Write-Host ""
Write-Host "Benchmark finished. Open:"
Write-Host "  graphml-studio\evaluation\cora\evaluation_report.md"

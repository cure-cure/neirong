$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:LAB_DEVICE = 'cpu'
$env:LAB_THREADS = '4'
$env:PYTHONIOENCODING = 'utf-8'
& (Join-Path $PSScriptRoot '.venv\Scripts\python.exe') (Join-Path $PSScriptRoot 'app.py')

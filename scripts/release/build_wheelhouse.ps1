$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "windows\build_wheelhouse.ps1") @args

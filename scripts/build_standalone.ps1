$ErrorActionPreference = "Stop"

function Get-PythonCandidates {
    $candidates = [System.Collections.Generic.List[string]]::new()

    function Add-Candidate {
        param([string]$Candidate)

        if (-not $Candidate) {
            return
        }
        if (-not (Test-Path $Candidate)) {
            return
        }

        $resolved = [System.IO.Path]::GetFullPath($Candidate)
        if (-not $candidates.Contains($resolved)) {
            $candidates.Add($resolved)
        }
    }

    $venvPython = if ($env:VIRTUAL_ENV) {
        Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"
    } else {
        $null
    }
    Add-Candidate $venvPython

    $condaPython = if ($env:CONDA_PREFIX) {
        Join-Path $env:CONDA_PREFIX "python.exe"
    } else {
        $null
    }
    Add-Candidate $condaPython

    if ($env:PYTHON) {
        Add-Candidate $env:PYTHON
    }

    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand -and $pythonCommand.Source) {
        Add-Candidate $pythonCommand.Source
    }

    $condaCommand = Get-Command conda -ErrorAction SilentlyContinue
    if ($condaCommand) {
        try {
            $condaEnvList = conda env list --json | ConvertFrom-Json
            foreach ($envRoot in $condaEnvList.envs) {
                Add-Candidate (Join-Path $envRoot "python.exe")
            }
        }
        catch {
        }
    }

    return $candidates
}

function Test-PyInstallerAvailable {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PythonExecutable
    )

    $quotedPython = '"' + $PythonExecutable.Replace('"', '""') + '"'
    cmd /c "$quotedPython -m PyInstaller --version >nul 2>nul"
    return ($LASTEXITCODE -eq 0)
}

function Resolve-PythonExecutable {
    $candidates = Get-PythonCandidates
    if ($candidates.Count -eq 0) {
        throw "Could not resolve a Python executable from the current environment"
    }

    foreach ($candidate in $candidates) {
        if (Test-PyInstallerAvailable -PythonExecutable $candidate) {
            return $candidate
        }
    }

    $candidateText = $candidates -join ", "
    throw "PyInstaller was not available in any candidate Python: $candidateText"
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$python = Resolve-PythonExecutable
$platformName = "windows"
$buildRoot = Join-Path $projectRoot "build\pyinstaller\$platformName"
$distRoot = Join-Path $projectRoot "dist\standalone\$platformName"
$specPath = Join-Path $projectRoot "packaging\tg-agent.spec"
$exeName = "tg-agent.exe"
$builtExe = Join-Path $distRoot $exeName
$packageRoot = Join-Path $distRoot "package"

if (Test-Path $buildRoot) {
    Remove-Item -LiteralPath $buildRoot -Recurse -Force
}
if (Test-Path $packageRoot) {
    Remove-Item -LiteralPath $packageRoot -Recurse -Force
}

Write-Host "Using Python: $python"

& $python -m PyInstaller `
    --noconfirm `
    --clean `
    --distpath $distRoot `
    --workpath $buildRoot `
    $specPath
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed"
}

New-Item -ItemType Directory -Path $packageRoot | Out-Null
Copy-Item -LiteralPath $builtExe -Destination (Join-Path $packageRoot $exeName) -Force
Copy-Item `
    -LiteralPath (Join-Path $scriptDir "install-tg-agent.ps1") `
    -Destination (Join-Path $packageRoot "install-tg-agent.ps1") `
    -Force
Copy-Item `
    -LiteralPath (Join-Path $projectRoot "tg_crab_worker.json") `
    -Destination (Join-Path $packageRoot "tg_crab_worker.json") `
    -Force

Write-Host "Standalone package ready: $packageRoot"

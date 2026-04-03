param(
    [string]$Version = "",
    [string]$Python = "",
    [string]$Arch = "x64",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

function Get-PythonCandidates {
    param(
        [string]$ExplicitPython = ""
    )

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

    if ($ExplicitPython) {
        Add-Candidate $ExplicitPython
        return $candidates
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
    param(
        [string]$ExplicitPython = ""
    )

    $candidates = Get-PythonCandidates -ExplicitPython $ExplicitPython
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

function Get-ProjectVersion {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PyprojectPath
    )

    $content = Get-Content -LiteralPath $PyprojectPath -Raw
    $match = [regex]::Match($content, '(?m)^version\s*=\s*"([^"]+)"')
    if (-not $match.Success) {
        throw "Could not find project version in $PyprojectPath"
    }
    return $match.Groups[1].Value
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$pythonExe = Resolve-PythonExecutable -ExplicitPython $Python
$resolvedVersion = if ($Version) { $Version } else { Get-ProjectVersion -PyprojectPath (Join-Path $projectRoot "pyproject.toml") }
$platformName = "windows"
$buildScript = Join-Path $scriptDir "build_standalone.ps1"
$packageRoot = Join-Path $projectRoot "dist\standalone\$platformName\package"
$releaseRoot = Join-Path $projectRoot "dist\release"
$releaseName = "tg-agent-$platformName-$Arch-v$resolvedVersion"
$releaseDir = Join-Path $releaseRoot $releaseName
$zipPath = Join-Path $releaseRoot "$releaseName.zip"
$readmePath = Join-Path $releaseDir "README.txt"

Write-Host "Preparing local release: $releaseName"
Write-Host "Using Python: $pythonExe"

if (-not $SkipBuild) {
    $env:PYTHON = $pythonExe
    & $buildScript
}

if (-not (Test-Path $packageRoot)) {
    throw "Package directory not found: $packageRoot"
}

if (Test-Path $releaseDir) {
    Remove-Item -LiteralPath $releaseDir -Recurse -Force
}
if (Test-Path $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}

New-Item -ItemType Directory -Path $releaseDir -Force | Out-Null
Copy-Item -Path (Join-Path $packageRoot "*") -Destination $releaseDir -Recurse -Force

@"
tg-agent local release package

Version: $resolvedVersion
Platform: $platformName
Architecture: $Arch

Contents:
- tg-agent.exe
- install-tg-agent.ps1
- tg_crab_worker.json

Install:
1. Extract this zip.
2. Run:
   powershell -ExecutionPolicy Bypass -File .\install-tg-agent.ps1

Direct run without install:
.\tg-agent.exe
"@ | Set-Content -LiteralPath $readmePath -Encoding UTF8

Compress-Archive -Path $releaseDir -DestinationPath $zipPath -CompressionLevel Optimal

Write-Host "Release directory: $releaseDir"
Write-Host "Release archive: $zipPath"

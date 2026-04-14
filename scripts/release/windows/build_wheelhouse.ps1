param(
    [string]$PythonExecutable = "",
    [string]$OutputDir = "",
    [switch]$Clean,
    [switch]$ProjectOnly
)

$ErrorActionPreference = "Stop"

function Get-RepoRoot {
    return [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\..\.."))
}

function Resolve-PythonExecutable {
    param([string]$ExplicitPython)

    $candidates = [System.Collections.Generic.List[string]]::new()

    function Normalize-CandidatePath {
        param([string]$Candidate)

        if (-not $Candidate) {
            return ""
        }

        return [Environment]::ExpandEnvironmentVariables($Candidate.Trim().Trim('"'))
    }

    function Add-Candidate {
        param([string]$Candidate)

        $normalized = Normalize-CandidatePath $Candidate
        if (-not $normalized) {
            return
        }

        if (Test-Path -LiteralPath $normalized -PathType Container) {
            $normalized = Join-Path $normalized "python.exe"
        }

        if (-not (Test-Path -LiteralPath $normalized -PathType Leaf)) {
            return
        }

        $resolved = [System.IO.Path]::GetFullPath($normalized)
        if (-not $candidates.Contains($resolved)) {
            $candidates.Add($resolved)
        }
    }

    function Add-CandidatesFromCommand {
        param([string]$CommandName)

        $commands = @(Get-Command $CommandName -All -ErrorAction SilentlyContinue)
        foreach ($command in $commands) {
            if ($command.Source) {
                Add-Candidate $command.Source
            }
        }
    }

    function Add-CandidatesFromWhere {
        param([string]$CommandName)

        try {
            $matches = & where.exe $CommandName 2>$null
            if ($LASTEXITCODE -ne 0) {
                return
            }

            foreach ($match in $matches) {
                $trimmed = $match.Trim()
                if (-not $trimmed) {
                    continue
                }
                if ($trimmed.StartsWith("INFO:", [System.StringComparison]::OrdinalIgnoreCase)) {
                    continue
                }

                Add-Candidate $trimmed
            }
        }
        catch {
        }
    }

    if ($ExplicitPython) {
        $explicitCandidate = Normalize-CandidatePath $ExplicitPython

        if (Test-Path -LiteralPath $explicitCandidate -PathType Container) {
            $explicitCandidate = Join-Path $explicitCandidate "python.exe"
        }

        Add-Candidate $explicitCandidate
        if ($candidates.Count -eq 0) {
            $explicitCommand = Get-Command $explicitCandidate -ErrorAction SilentlyContinue
            if ($explicitCommand -and $explicitCommand.Source) {
                Add-Candidate $explicitCommand.Source
            }
        }

        if ($candidates.Count -eq 0) {
            throw "Explicit Python executable was provided but not found: $ExplicitPython"
        }
    }
    else {
        if ($env:VIRTUAL_ENV) {
            Add-Candidate (Join-Path $env:VIRTUAL_ENV "Scripts\python.exe")
        }
        if ($env:CONDA_PREFIX) {
            Add-Candidate (Join-Path $env:CONDA_PREFIX "python.exe")
        }
        if ($env:CONDA_PYTHON_EXE) {
            Add-Candidate $env:CONDA_PYTHON_EXE
        }
        if ($env:PYTHON) {
            Add-Candidate $env:PYTHON
        }

        Add-CandidatesFromCommand "python"
        Add-CandidatesFromWhere "python"

        $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
        if ($pyLauncher -and $pyLauncher.Source) {
            try {
                $resolvedPy = & $pyLauncher.Source -3 -c "import sys; print(sys.executable)"
                if ($LASTEXITCODE -eq 0) {
                    Add-Candidate $resolvedPy.Trim()
                }
            }
            catch {
            }
        }
    }

    $explicitFailureMessage = ""
    foreach ($candidate in $candidates) {
        try {
            $probeOutput = @(& $candidate -c "import sys; print(sys.executable)" 2>&1)
            $resolvedCandidate = ($probeOutput | Select-Object -First 1 | Out-String).Trim()
            if ($LASTEXITCODE -eq 0) {
                if ($resolvedCandidate -and (Test-Path -LiteralPath $resolvedCandidate)) {
                    return [System.IO.Path]::GetFullPath($resolvedCandidate)
                }

                return $candidate
            }

            if ($ExplicitPython) {
                $probeMessage = ($probeOutput | ForEach-Object { $_.ToString().Trim() } | Where-Object { $_ }) -join " "
                if ($probeMessage) {
                    $explicitFailureMessage = "ExitCode=$LASTEXITCODE. $probeMessage"
                }
                else {
                    $explicitFailureMessage = "ExitCode=$LASTEXITCODE."
                }
            }
        }
        catch {
            if ($ExplicitPython) {
                $explicitFailureMessage = $_.Exception.Message
            }
        }
    }

    if ($ExplicitPython) {
        $message = "Explicit Python executable is not usable: $ExplicitPython"
        if ($explicitFailureMessage) {
            $message = "$message. $explicitFailureMessage"
        }
        throw $message
    }

    throw "Could not resolve a usable Python executable automatically. Pass -PythonExecutable explicitly, for example '-PythonExecutable C:\Python310\python.exe' or '-PythonExecutable %CONDA_PREFIX%\python.exe'."
}

function Test-PathUnderRoot {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Root
    )

    $fullPath = [System.IO.Path]::GetFullPath($Path).TrimEnd('\')
    $fullRoot = [System.IO.Path]::GetFullPath($Root).TrimEnd('\')
    return $fullPath -eq $fullRoot -or
        $fullPath.StartsWith("$fullRoot\", [System.StringComparison]::OrdinalIgnoreCase)
}

function Reset-Directory {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TargetPath,
        [Parameter(Mandatory = $true)]
        [string]$SafeRoot
    )

    if (Test-Path -LiteralPath $TargetPath) {
        if (-not (Test-PathUnderRoot -Path $TargetPath -Root $SafeRoot)) {
            throw "Refusing to delete path outside safe root: $TargetPath"
        }
        Remove-Item -LiteralPath $TargetPath -Recurse -Force
    }

    New-Item -ItemType Directory -Path $TargetPath -Force | Out-Null
}

function Assert-BuildBackendAvailable {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PythonExe
    )

    & $PythonExe -c "import hatchling" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Missing hatchling in $PythonExe. Run '$PythonExe -m pip install hatchling' first."
    }
}

function Build-Wheelhouse {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PythonExe,
        [Parameter(Mandatory = $true)]
        [string]$RepoRoot,
        [Parameter(Mandatory = $true)]
        [string]$TargetWheelhouseDir,
        [Parameter(Mandatory = $true)]
        [string]$TempRoot,
        [switch]$ProjectOnly
    )

    New-Item -ItemType Directory -Path $TempRoot -Force | Out-Null

    $previousTemp = $env:TEMP
    $previousTmp = $env:TMP
    $previousBuildTracker = $env:PIP_BUILD_TRACKER

    Push-Location $RepoRoot
    try {
        $env:TEMP = $TempRoot
        $env:TMP = $TempRoot
        $env:PIP_BUILD_TRACKER = Join-Path $TempRoot "pip-build-tracker"
        New-Item -ItemType Directory -Path $env:PIP_BUILD_TRACKER -Force | Out-Null

        $args = @("-m", "pip", "wheel", ".", "--wheel-dir", $TargetWheelhouseDir)
        if ($ProjectOnly) {
            $args += "--no-deps"
        }

        & $PythonExe @args
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to build wheelhouse with $PythonExe"
        }
    }
    finally {
        Pop-Location
        $env:TEMP = $previousTemp
        $env:TMP = $previousTmp
        $env:PIP_BUILD_TRACKER = $previousBuildTracker
    }
}

$repoRoot = Get-RepoRoot
$resolvedPythonExe = Resolve-PythonExecutable -ExplicitPython $PythonExecutable
$resolvedOutputDir = if ($OutputDir) {
    [System.IO.Path]::GetFullPath($OutputDir)
}
else {
    [System.IO.Path]::GetFullPath((Join-Path $repoRoot "dist\package\windows\wheelhouse"))
}
$outputParent = Split-Path -Parent $resolvedOutputDir
$tempRoot = Join-Path $outputParent ".wheelhouse_build_tmp"
$pythonVersion = (& $resolvedPythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')").Trim()

New-Item -ItemType Directory -Path $outputParent -Force | Out-Null

if ($Clean) {
    Reset-Directory -TargetPath $resolvedOutputDir -SafeRoot $outputParent
}
else {
    New-Item -ItemType Directory -Path $resolvedOutputDir -Force | Out-Null
    $existingEntries = Get-ChildItem -LiteralPath $resolvedOutputDir -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($existingEntries) {
        Write-Warning "Output directory already contains files. Use -Clean to avoid mixing wheelhouses from different Python versions."
    }
}

Assert-BuildBackendAvailable -PythonExe $resolvedPythonExe

Write-Host "[wheelhouse] repo root: $repoRoot"
Write-Host "[wheelhouse] python exe: $resolvedPythonExe"
Write-Host "[wheelhouse] python version: $pythonVersion"
Write-Host "[wheelhouse] output dir: $resolvedOutputDir"
Write-Host "[wheelhouse] mode: $(if ($ProjectOnly) { 'project-only' } else { 'project-and-dependencies' })"

Build-Wheelhouse `
    -PythonExe $resolvedPythonExe `
    -RepoRoot $repoRoot `
    -TargetWheelhouseDir $resolvedOutputDir `
    -TempRoot $tempRoot `
    -ProjectOnly:$ProjectOnly

if (Test-Path -LiteralPath $tempRoot) {
    Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

$builtWheels = Get-ChildItem -LiteralPath $resolvedOutputDir -Filter "*.whl" -ErrorAction SilentlyContinue
Write-Host "[wheelhouse] build complete"
Write-Host "[wheelhouse] wheel count: $($builtWheels.Count)"
Write-Host "[wheelhouse] output dir: $resolvedOutputDir"

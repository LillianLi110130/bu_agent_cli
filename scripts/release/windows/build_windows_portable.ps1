param(
    [string]$Version = "",
    [string]$PythonExecutable = "",
    [string]$PythonHome = "",
    [string]$SourceWheelhouse = "",
    [string]$OutputRoot = "",
    [switch]$AllowCondaPythonRuntime,
    [switch]$SkipZip,
    [switch]$SkipProjectWheelBuild
)

$ErrorActionPreference = "Stop"

function Get-RepoRoot {
    return [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\..\.."))
}

function Get-ProjectVersion {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PyprojectPath
    )

    $content = Get-Content -LiteralPath $PyprojectPath -Raw -Encoding UTF8
    if ($content -match '(?ms)^\[project\].*?^\s*version\s*=\s*"([^"]+)"') {
        return $Matches[1]
    }

    throw "Could not resolve version from $PyprojectPath"
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

function Resolve-PythonHome {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PythonExe,
        [string]$ExplicitPythonHome
    )

    if ($ExplicitPythonHome) {
        $resolved = [System.IO.Path]::GetFullPath($ExplicitPythonHome)
    }
    else {
        $resolved = (& $PythonExe -c "import sys; print(sys.base_prefix)").Trim()
    }

    if (-not $resolved) {
        throw "Could not resolve Python home from $PythonExe"
    }

    $resolved = [System.IO.Path]::GetFullPath($resolved)
    if (-not (Test-Path -LiteralPath (Join-Path $resolved "python.exe"))) {
        throw "python.exe not found in Python home: $resolved"
    }

    return $resolved
}

function Test-CondaPythonHome {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PythonHomeDir
    )

    return (Test-Path -LiteralPath (Join-Path $PythonHomeDir "conda-meta")) -or
        (Test-Path -LiteralPath (Join-Path $PythonHomeDir "condabin"))
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

        $attempts = 6
        $deleted = $false
        for ($attempt = 1; $attempt -le $attempts; $attempt++) {
            try {
                Remove-Item -LiteralPath $TargetPath -Recurse -Force -ErrorAction Stop
                $deleted = $true
                break
            }
            catch {
                if ($attempt -eq $attempts) {
                    $message = @(
                        "Failed to reset output directory because it is still in use: $TargetPath",
                        "Close any Explorer window, terminal, or running crab process that is using this bundle directory, then rerun the build.",
                        "You can also pass -OutputRoot to build into a different release directory."
                    ) -join " "
                    throw "$message Original error: $($_.Exception.Message)"
                }

                Start-Sleep -Milliseconds (250 * $attempt)
            }
        }

        if (-not $deleted -and (Test-Path -LiteralPath $TargetPath)) {
            throw "Failed to reset output directory: $TargetPath"
        }
    }

    New-Item -ItemType Directory -Path $TargetPath -Force | Out-Null
}

function Copy-AppPayload {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RepoRoot,
        [Parameter(Mandatory = $true)]
        [string]$TargetDir
    )

    $items = @(
        "agent_core",
        "cli",
        "tools",
        "tg_mem",
        "skills",
        "plugins",
        "config",
        "template",
        ".devagent",
        "tg_crab_main.py",
        "ralph_init.py",
        "ralph_loop.py",
        ".env",
        "tg_crab_worker.json"
    )

    foreach ($item in $items) {
        $source = Join-Path $RepoRoot $item
        if (-not (Test-Path -LiteralPath $source)) {
            throw "Required app payload is missing: $source"
        }
        Copy-Item -LiteralPath $source -Destination $TargetDir -Recurse -Force
    }
}

function Copy-FilteredPythonLib {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SourceLibDir,
        [Parameter(Mandatory = $true)]
        [string]$TargetLibDir
    )

    New-Item -ItemType Directory -Path $TargetLibDir -Force | Out-Null

    Get-ChildItem -LiteralPath $SourceLibDir -Force -File |
        ForEach-Object {
            Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $TargetLibDir $_.Name) -Force
        }

    Get-ChildItem -LiteralPath $SourceLibDir -Force -Directory |
        ForEach-Object {
            Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $TargetLibDir $_.Name) -Recurse -Force
        }

    $removePaths = @(
        (Join-Path $TargetLibDir "site-packages"),
        (Join-Path $TargetLibDir "test"),
        (Join-Path $TargetLibDir "tests"),
        (Join-Path $TargetLibDir "idlelib"),
        (Join-Path $TargetLibDir "tkinter"),
        (Join-Path $TargetLibDir "turtledemo"),
        (Join-Path $TargetLibDir "__pycache__")
    )

    foreach ($path in $removePaths) {
        if (Test-Path -LiteralPath $path) {
            Remove-Item -LiteralPath $path -Recurse -Force
        }
    }

    Get-ChildItem -LiteralPath $TargetLibDir -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Get-ChildItem -LiteralPath $TargetLibDir -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Extension -in @(".pyc", ".pyo") } |
        Remove-Item -Force -ErrorAction SilentlyContinue
}

function Copy-PythonRuntime {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SourcePythonHome,
        [Parameter(Mandatory = $true)]
        [string]$TargetRuntimeDir
    )

    $rootPatterns = @(
        "python*.exe",
        "python*.dll",
        "vcruntime*.dll",
        "ucrtbase.dll",
        "*.cfg",
        "LICENSE*",
        "pyvenv.cfg"
    )

    foreach ($pattern in $rootPatterns) {
        Get-ChildItem -LiteralPath $SourcePythonHome -Filter $pattern -File -ErrorAction SilentlyContinue |
            ForEach-Object {
                Copy-Item -LiteralPath $_.FullName -Destination $TargetRuntimeDir -Force
            }
    }

    $dirsToCopy = @("DLLs", "libs", "Lib")
    foreach ($dirName in $dirsToCopy) {
        $sourceDir = Join-Path $SourcePythonHome $dirName
        if (-not (Test-Path -LiteralPath $sourceDir)) {
            continue
        }

        $targetDir = Join-Path $TargetRuntimeDir $dirName
        if ($dirName -eq "Lib") {
            Copy-FilteredPythonLib -SourceLibDir $sourceDir -TargetLibDir $targetDir
        }
        else {
            Copy-Item -LiteralPath $sourceDir -Destination $targetDir -Recurse -Force
        }
    }
}

function Copy-Wheelhouse {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SourceWheelhouseDir,
        [Parameter(Mandatory = $true)]
        [string]$TargetWheelhouseDir
    )

    if (-not (Test-Path -LiteralPath $SourceWheelhouseDir)) {
        throw "Source wheelhouse directory not found: $SourceWheelhouseDir"
    }

    Copy-Item -Path (Join-Path $SourceWheelhouseDir "*") -Destination $TargetWheelhouseDir -Recurse -Force
}

function Build-ProjectWheel {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PythonExe,
        [Parameter(Mandatory = $true)]
        [string]$RepoRoot,
        [Parameter(Mandatory = $true)]
        [string]$TargetWheelhouseDir,
        [Parameter(Mandatory = $true)]
        [string]$TempRoot
    )

    Get-ChildItem -LiteralPath $TargetWheelhouseDir -Filter "tg_agent_cli-*.whl" -ErrorAction SilentlyContinue |
        Remove-Item -Force -ErrorAction SilentlyContinue

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
        & $PythonExe -m pip wheel . --no-deps --wheel-dir $TargetWheelhouseDir --no-build-isolation
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to build project wheel with $PythonExe. Install hatchling or pass a prebuilt wheelhouse."
        }
    }
    finally {
        Pop-Location
        $env:TEMP = $previousTemp
        $env:TMP = $previousTmp
        $env:PIP_BUILD_TRACKER = $previousBuildTracker
    }
}

function Build-FullWheelhouse {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PythonExe,
        [Parameter(Mandatory = $true)]
        [string]$RepoRoot,
        [Parameter(Mandatory = $true)]
        [string]$TargetWheelhouseDir,
        [Parameter(Mandatory = $true)]
        [string]$TempRoot
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
        & $PythonExe -m pip wheel . --wheel-dir $TargetWheelhouseDir
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to build full wheelhouse with $PythonExe. Ensure pip can resolve dependency wheels for this Python version."
        }
    }
    finally {
        Pop-Location
        $env:TEMP = $previousTemp
        $env:TMP = $previousTmp
        $env:PIP_BUILD_TRACKER = $previousBuildTracker
    }
}

function Write-DeployBatch {
    param(
        [Parameter(Mandatory = $true)]
        [string]$OutputPath
    )

    $content = @'
@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0win_deploy.ps1" %*
set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
    echo.
    echo crab deployment failed with exit code %EXITCODE%.
    echo Press any key to close this window.
    pause >nul
    exit /b %EXITCODE%
)
echo.
echo crab deployment completed successfully.
echo Press any key to close this window.
pause >nul
exit /b %EXITCODE%
'@
    Set-Content -LiteralPath $OutputPath -Value $content -Encoding ASCII
}

function Write-BundleReadme {
    param(
        [Parameter(Mandatory = $true)]
        [string]$OutputPath,
        [Parameter(Mandatory = $true)]
        [string]$VersionText
    )

    $content = @"
crab portable bundle

Version: $VersionText
Platform: windows
Architecture: x64

Contents:
- deploy.bat
- win_deploy.ps1
- crab-launcher.bat
- python-runtime\
- wheelhouse\
- app\

First run:
1. Extract this zip.
2. Run deploy.bat once.
3. Start crab-launcher.bat from the workspace you want to use.

Workspace behavior:
- Running launcher from a terminal uses the current working directory as the workspace.
- Double-clicking launcher from the bundle directory falls back to the Desktop directory.
- Global config stays in %USERPROFILE%\.tg_agent.
- After deploy.bat, `crab` works in cmd, PowerShell, and Git Bash.

Notes:
- deploy.bat copies the bundled Python runtime into %USERPROFILE%\.tg_agent\python-runtime and creates
  %USERPROFILE%\.tg_agent\.venv from that stable runtime.
- crab and dependencies are installed offline from wheelhouse\.
- deploy.bat also installs a `crab` command shim into %USERPROFILE%\.tg_agent\bin and adds that directory to the user PATH.
- deploy.bat removes stale crab/tg-agent command shim files left behind in %USERPROFILE%\.tg_agent\bin by older installs.
- If `crab` already exists from an older pip install, uninstall the older copy to avoid command precedence conflicts.
- Existing %USERPROFILE%\.tg_agent\.env and tg_crab_worker.json are preserved.
"@
    Set-Content -LiteralPath $OutputPath -Value $content -Encoding ASCII
}

function Write-WinDeployScript {
    param(
        [Parameter(Mandatory = $true)]
        [string]$OutputPath
    )

    $content = @'
param(
    [switch]$ForceRecreateVenv,
    [switch]$SkipDesktopShortcut
)

$ErrorActionPreference = "Stop"

$BundleRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RuntimeDir = Join-Path $BundleRoot "python-runtime"
$RuntimePython = Join-Path $RuntimeDir "python.exe"
$WheelhouseDir = Join-Path $BundleRoot "wheelhouse"
$AppDir = Join-Path $BundleRoot "app"
$LauncherPath = Join-Path $BundleRoot "crab-launcher.bat"
$UserProfileDir = if ($env:USERPROFILE) { $env:USERPROFILE } else { [Environment]::GetFolderPath("UserProfile") }
$InstallRoot = Join-Path $UserProfileDir ".tg_agent"
$InstalledRuntimeDir = Join-Path $InstallRoot "python-runtime"
$InstalledRuntimePython = Join-Path $InstalledRuntimeDir "python.exe"
$VenvDir = Join-Path $InstallRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$PyvenvConfig = Join-Path $VenvDir "pyvenv.cfg"
$BinDir = Join-Path $InstallRoot "bin"
$EntryShim = Join-Path $BinDir "crab-entry.py"
$CommandShim = Join-Path $BinDir "crab.cmd"
$BashCommandShim = Join-Path $BinDir "crab"
$LegacyCommandExe = Join-Path $BinDir "crab.exe"
$LegacyCommandScript = Join-Path $BinDir "crab-script.py"
$LegacyTgAgentEntryShim = Join-Path $BinDir "tg-agent-entry.py"
$LegacyTgAgentCommandShim = Join-Path $BinDir "tg-agent.cmd"
$LegacyTgAgentBashShim = Join-Path $BinDir "tg-agent"
$LegacyTgAgentCommandExe = Join-Path $BinDir "tg-agent.exe"
$LegacyTgAgentCommandScript = Join-Path $BinDir "tg-agent-script.py"
$EnvFile = Join-Path $InstallRoot ".env"
$WorkerConfig = Join-Path $InstallRoot "tg_crab_worker.json"
$PackagedEnvFile = Join-Path $AppDir ".env"
$PackagedWorkerConfig = Join-Path $AppDir "tg_crab_worker.json"

function Install-BundledRuntime {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SourceRuntimeDir,
        [Parameter(Mandatory = $true)]
        [string]$InstallRootDir,
        [Parameter(Mandatory = $true)]
        [string]$TargetRuntimeDir
    )

    if (Test-Path -LiteralPath $TargetRuntimeDir) {
        Remove-Item -LiteralPath $TargetRuntimeDir -Recurse -Force
    }

    Copy-Item -LiteralPath $SourceRuntimeDir -Destination $InstallRootDir -Recurse -Force
}

function Test-VenvUsesRuntime {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PyvenvConfigPath,
        [Parameter(Mandatory = $true)]
        [string]$ExpectedRuntimeDir
    )

    if (-not (Test-Path -LiteralPath $PyvenvConfigPath)) {
        return $false
    }

    $homeLine = Get-Content -LiteralPath $PyvenvConfigPath -ErrorAction SilentlyContinue |
        Where-Object { $_ -match '^\s*home\s*=' } |
        Select-Object -First 1
    if (-not $homeLine) {
        return $false
    }

    $configuredHome = [System.IO.Path]::GetFullPath(($homeLine -replace '^\s*home\s*=\s*', '').Trim())
    $expectedHome = [System.IO.Path]::GetFullPath($ExpectedRuntimeDir)
    return $configuredHome.TrimEnd('\') -ieq $expectedHome.TrimEnd('\')
}

function Add-UserPathEntry {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Entry
    )

    $normalizedEntry = $Entry.Trim().TrimEnd('\')
    if (-not $normalizedEntry) {
        return $false
    }

    $currentUserPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $existingEntries = @()
    if ($currentUserPath) {
        $existingEntries = @(
            $currentUserPath -split ';' |
                ForEach-Object { $_.Trim() } |
                Where-Object { $_ }
        )
    }

    $alreadyPresent = $existingEntries |
        Where-Object { $_.TrimEnd('\') -ieq $normalizedEntry } |
        Select-Object -First 1

    if (-not $alreadyPresent) {
        $newUserPath = if ($currentUserPath) {
            "$currentUserPath;$normalizedEntry"
        }
        else {
            $normalizedEntry
        }

        [Environment]::SetEnvironmentVariable("Path", $newUserPath, "User")
    }

    $sessionEntries = @()
    if ($env:PATH) {
        $sessionEntries = @(
            $env:PATH -split ';' |
                ForEach-Object { $_.Trim() } |
                Where-Object { $_ }
        )
    }

    $sessionHasEntry = $sessionEntries |
        Where-Object { $_.TrimEnd('\') -ieq $normalizedEntry } |
        Select-Object -First 1

    if (-not $sessionHasEntry) {
        $env:PATH = if ($env:PATH) {
            "$normalizedEntry;$env:PATH"
        }
        else {
            $normalizedEntry
        }
    }

    return (-not $alreadyPresent)
}

function Notify-EnvironmentChange {
    try {
        Add-Type -Namespace TgAgentPortable -Name NativeMethods -MemberDefinition @"
using System;
using System.Runtime.InteropServices;

public static class NativeMethods
{
    [DllImport("user32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern IntPtr SendMessageTimeout(
        IntPtr hWnd,
        uint Msg,
        UIntPtr wParam,
        string lParam,
        uint fuFlags,
        uint uTimeout,
        out UIntPtr lpdwResult);
}
"@ -ErrorAction SilentlyContinue | Out-Null

        $HWND_BROADCAST = [IntPtr]0xffff
        $WM_SETTINGCHANGE = 0x001A
        $SMTO_ABORTIFHUNG = 0x0002
        $result = [UIntPtr]::Zero

        [TgAgentPortable.NativeMethods]::SendMessageTimeout(
            $HWND_BROADCAST,
            $WM_SETTINGCHANGE,
            [UIntPtr]::Zero,
            "Environment",
            $SMTO_ABORTIFHUNG,
            5000,
            [ref]$result
        ) | Out-Null
    }
    catch {
    }
}

function Get-CommandCandidates {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CommandName
    )

    $candidates = @()

    try {
        $matches = & where.exe $CommandName 2>$null
        if ($LASTEXITCODE -ne 0) {
            return @()
        }

        foreach ($match in $matches) {
            $trimmed = $match.Trim()
            if (-not $trimmed) {
                continue
            }
            if ($trimmed.StartsWith("INFO:", [System.StringComparison]::OrdinalIgnoreCase)) {
                continue
            }

            $resolved = [System.IO.Path]::GetFullPath($trimmed)
            if (-not ($candidates | Where-Object { $_ -ieq $resolved } | Select-Object -First 1)) {
                $candidates += $resolved
            }
        }
    }
    catch {
    }

    return $candidates
}

function Get-PipUninstallHint {
    param(
        [string]$CommandPath
    )

    if (-not $CommandPath) {
        return ""
    }

    $commandDir = Split-Path -Parent $CommandPath
    if ((Split-Path -Leaf $commandDir) -ieq "Scripts") {
        $pythonRoot = Split-Path -Parent $commandDir
        $pythonExe = Join-Path $pythonRoot "python.exe"
        if (Test-Path -LiteralPath $pythonExe) {
            return "$pythonExe -m pip uninstall tg-agent-cli"
        }
    }

    return ""
}

if (-not (Test-Path -LiteralPath $RuntimePython)) {
    throw "Bundled Python runtime not found: $RuntimePython"
}
if (-not (Test-Path -LiteralPath $WheelhouseDir)) {
    throw "Bundled wheelhouse not found: $WheelhouseDir"
}

$projectWheel = Get-ChildItem -LiteralPath $WheelhouseDir -Filter "tg_agent_cli-*.whl" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if (-not $projectWheel) {
    throw "Project wheel not found in wheelhouse: $WheelhouseDir"
}

New-Item -ItemType Directory -Path $InstallRoot -Force | Out-Null
New-Item -ItemType Directory -Path $BinDir -Force | Out-Null
Install-BundledRuntime -SourceRuntimeDir $RuntimeDir -InstallRootDir $InstallRoot -TargetRuntimeDir $InstalledRuntimeDir

if (-not (Test-Path -LiteralPath $InstalledRuntimePython)) {
    throw "Installed Python runtime not found after copy: $InstalledRuntimePython"
}

if ($ForceRecreateVenv -and (Test-Path -LiteralPath $VenvDir)) {
    Remove-Item -LiteralPath $VenvDir -Recurse -Force
}

$venvNeedsRebuild = (-not (Test-Path -LiteralPath $VenvPython)) -or
    (-not (Test-VenvUsesRuntime -PyvenvConfigPath $PyvenvConfig -ExpectedRuntimeDir $InstalledRuntimeDir))
if ($venvNeedsRebuild -and (Test-Path -LiteralPath $VenvDir)) {
    Remove-Item -LiteralPath $VenvDir -Recurse -Force
}

if (-not (Test-Path -LiteralPath $VenvPython)) {
    & $InstalledRuntimePython -m venv $VenvDir
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $VenvPython)) {
        throw "Failed to create virtual environment: $VenvDir"
    }
}

& $VenvPython -m ensurepip --upgrade
if ($LASTEXITCODE -ne 0) {
    throw "Failed to bootstrap pip into virtual environment: $VenvDir"
}

& $VenvPython -m pip install --no-index --find-links $WheelhouseDir --upgrade --force-reinstall $projectWheel.FullName
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install crab from bundled wheelhouse"
}

$entryShimContent = @(
    'from tg_crab_main import cli_main',
    '',
    'if __name__ == "__main__":',
    '    cli_main()'
) -join "`r`n"
Set-Content -LiteralPath $EntryShim -Value $entryShimContent -Encoding UTF8

$commandShimContent = @(
    '@echo off',
    'setlocal',
    'set "INSTALL_ROOT=%USERPROFILE%\.tg_agent"',
    'set "VENV_PYTHON=%INSTALL_ROOT%\.venv\Scripts\python.exe"',
    'set "ENTRY_SHIM=%INSTALL_ROOT%\bin\crab-entry.py"',
    '',
    'if not exist "%VENV_PYTHON%" (',
    '    echo crab is not installed.',
    '    echo Run deploy.bat from the portable bundle first.',
    '    exit /b 1',
    ')',
    'if not exist "%ENTRY_SHIM%" (',
    '    echo crab entry shim is missing.',
    '    echo Run deploy.bat from the portable bundle first.',
    '    exit /b 1',
    ')',
    '',
    '"%VENV_PYTHON%" -u "%ENTRY_SHIM%" %*',
    'exit /b %ERRORLEVEL%'
) -join "`r`n"
Set-Content -LiteralPath $CommandShim -Value $commandShimContent -Encoding ASCII

$bashCommandShimContent = @(
    '#!/usr/bin/env bash',
    'set -euo pipefail',
    '',
    'install_root="${TG_AGENT_HOME:-${HOME}/.tg_agent}"',
    'if [ -n "${USERPROFILE:-}" ] && [ ! -d "$install_root" ]; then',
    '  install_root="${USERPROFILE}/.tg_agent"',
    'fi',
    '',
    'venv_python="${install_root}/.venv/Scripts/python.exe"',
    'entry_shim="${install_root}/bin/crab-entry.py"',
    '',
    'if [ ! -x "$venv_python" ]; then',
    '  echo "crab is not installed." >&2',
    '  echo "Run deploy.bat from the portable bundle first." >&2',
    '  exit 1',
    'fi',
    'if [ ! -f "$entry_shim" ]; then',
    '  echo "crab entry shim is missing." >&2',
    '  echo "Run deploy.bat from the portable bundle first." >&2',
    '  exit 1',
    'fi',
    '',
    'exec "$venv_python" -u "$entry_shim" "$@"'
) -join "`n"
Set-Content -LiteralPath $BashCommandShim -Value $bashCommandShimContent -Encoding ASCII

foreach ($legacyPath in @(
    $LegacyCommandExe,
    $LegacyCommandScript,
    $LegacyTgAgentEntryShim,
    $LegacyTgAgentCommandShim,
    $LegacyTgAgentBashShim,
    $LegacyTgAgentCommandExe,
    $LegacyTgAgentCommandScript
)) {
    if (Test-Path -LiteralPath $legacyPath) {
        try {
            Remove-Item -LiteralPath $legacyPath -Force -ErrorAction Stop
            Write-Host "[portable] removed stale command shim: $legacyPath"
        }
        catch {
            Write-Warning "Could not remove stale command shim: $legacyPath. $($_.Exception.Message)"
        }
    }
}

if ((Test-Path -LiteralPath $PackagedEnvFile) -and -not (Test-Path -LiteralPath $EnvFile)) {
    Copy-Item -LiteralPath $PackagedEnvFile -Destination $EnvFile -Force
}
if ((Test-Path -LiteralPath $PackagedWorkerConfig) -and -not (Test-Path -LiteralPath $WorkerConfig)) {
    Copy-Item -LiteralPath $PackagedWorkerConfig -Destination $WorkerConfig -Force
}

if (-not (Test-Path -LiteralPath $LauncherPath)) {
    throw "Launcher missing: $LauncherPath"
}

$existingCrabCommands = @(
    Get-CommandCandidates -CommandName "crab" |
        Where-Object { $_ -ine $CommandShim -and $_ -ine $BashCommandShim }
)

$pathUpdated = Add-UserPathEntry -Entry $BinDir
Notify-EnvironmentChange

if (-not $SkipDesktopShortcut) {
    $desktopDir = Join-Path $UserProfileDir "Desktop"
    if (Test-Path -LiteralPath $desktopDir) {
        $shortcutPath = Join-Path $desktopDir "Crab Portable.lnk"
        $shell = New-Object -ComObject WScript.Shell
        $shortcut = $shell.CreateShortcut($shortcutPath)
        $shortcut.TargetPath = $LauncherPath
        $shortcut.WorkingDirectory = $desktopDir
        $shortcut.Save()
        Write-Host "[portable] desktop shortcut created: $shortcutPath"
    }
}

Write-Host "[portable] install root: $InstallRoot"
Write-Host "[portable] installed runtime: $InstalledRuntimeDir"
Write-Host "[portable] venv ready: $VenvDir"
Write-Host "[portable] command shim: $CommandShim"
Write-Host "[portable] bash shim: $BashCommandShim"
if ($pathUpdated) {
    Write-Host "[portable] added to user PATH: $BinDir"
}
else {
    Write-Host "[portable] user PATH already contains: $BinDir"
}
Write-Host "[portable] launcher: $LauncherPath"
Write-Host "[portable] open a new cmd window from Explorer and run: crab"
Write-Host "[portable] in Git Bash, you can also run: crab"
Write-Host "[portable] if crab is still not found, sign out and sign back in once."
if ($existingCrabCommands.Count -gt 0) {
    $conflictingCommand = $existingCrabCommands[0]
    Write-Warning "Detected another crab command on this machine: $conflictingCommand"
    Write-Warning "The portable install does not overwrite older pip-based crab commands."

    $pipUninstallHint = Get-PipUninstallHint -CommandPath $conflictingCommand
    if ($pipUninstallHint) {
        Write-Warning "To avoid command conflicts, uninstall the older copy with: $pipUninstallHint"
    }
    else {
        Write-Warning "To avoid command conflicts, uninstall the older crab copy before using the PATH-based command."
    }

    Write-Warning "You can always launch the portable install directly with: $CommandShim"
}
Write-Host "[portable] start the launcher from your workspace directory."
'@

    Set-Content -LiteralPath $OutputPath -Value $content -Encoding UTF8
}

$repoRoot = Get-RepoRoot
$pyprojectPath = Join-Path $repoRoot "pyproject.toml"
$resolvedVersion = if ($Version) { $Version } else { Get-ProjectVersion -PyprojectPath $pyprojectPath }
$resolvedOutputRoot = if ($OutputRoot) {
    [System.IO.Path]::GetFullPath($OutputRoot)
}
else {
    [System.IO.Path]::GetFullPath((Join-Path $repoRoot "dist\release"))
}

$resolvedPythonExe = Resolve-PythonExecutable -ExplicitPython $PythonExecutable
$resolvedPythonHome = Resolve-PythonHome -PythonExe $resolvedPythonExe -ExplicitPythonHome $PythonHome
if ((Test-CondaPythonHome -PythonHomeDir $resolvedPythonHome) -and -not $AllowCondaPythonRuntime) {
    throw "Resolved Python home appears to be a conda runtime: $resolvedPythonHome. Use a standard CPython install or pass -AllowCondaPythonRuntime if you really want to continue."
}
$resolvedSourceWheelhouse = if ($SourceWheelhouse) {
    [System.IO.Path]::GetFullPath($SourceWheelhouse)
}
else {
    $defaultWheelhouse = Join-Path $repoRoot "dist\package\windows\wheelhouse"
    if (Test-Path -LiteralPath $defaultWheelhouse) {
        [System.IO.Path]::GetFullPath($defaultWheelhouse)
    }
    else {
        ""
    }
}

$bundleName = "tg-agent-windows-x64-v$resolvedVersion-portable"
$bundleDir = Join-Path $resolvedOutputRoot $bundleName
$appDir = Join-Path $bundleDir "app"
$runtimeDir = Join-Path $bundleDir "python-runtime"
$wheelhouseDir = Join-Path $bundleDir "wheelhouse"
$tempDir = Join-Path $bundleDir "tmp"
$launcherPath = Join-Path $bundleDir "crab-launcher.bat"
$deployBat = Join-Path $bundleDir "deploy.bat"
$deployPs1 = Join-Path $bundleDir "win_deploy.ps1"
$readmePath = Join-Path $bundleDir "README.txt"
$zipPath = Join-Path $resolvedOutputRoot "$bundleName.zip"

New-Item -ItemType Directory -Path $resolvedOutputRoot -Force | Out-Null
Reset-Directory -TargetPath $bundleDir -SafeRoot $resolvedOutputRoot
Reset-Directory -TargetPath $appDir -SafeRoot $bundleDir
Reset-Directory -TargetPath $runtimeDir -SafeRoot $bundleDir
Reset-Directory -TargetPath $wheelhouseDir -SafeRoot $bundleDir
Reset-Directory -TargetPath $tempDir -SafeRoot $bundleDir

Write-Host "[portable] repo root: $repoRoot"
Write-Host "[portable] python exe: $resolvedPythonExe"
Write-Host "[portable] python home: $resolvedPythonHome"
Write-Host "[portable] bundle dir: $bundleDir"

Write-Host "[portable] copying app payload..."
Copy-AppPayload -RepoRoot $repoRoot -TargetDir $appDir

Write-Host "[portable] copying bundled Python runtime..."
Copy-PythonRuntime -SourcePythonHome $resolvedPythonHome -TargetRuntimeDir $runtimeDir

if ($resolvedSourceWheelhouse) {
    Write-Host "[portable] copying source wheelhouse from $resolvedSourceWheelhouse"
    Copy-Wheelhouse -SourceWheelhouseDir $resolvedSourceWheelhouse -TargetWheelhouseDir $wheelhouseDir
}
else {
    Write-Host "[portable] no source wheelhouse provided; building dependency wheelhouse from the project"
    Build-FullWheelhouse -PythonExe $resolvedPythonExe -RepoRoot $repoRoot -TargetWheelhouseDir $wheelhouseDir -TempRoot $tempDir
}

if ($resolvedSourceWheelhouse -and -not $SkipProjectWheelBuild) {
    Write-Host "[portable] building project wheel..."
    Build-ProjectWheel -PythonExe $resolvedPythonExe -RepoRoot $repoRoot -TargetWheelhouseDir $wheelhouseDir -TempRoot $tempDir
}

if (Test-Path -LiteralPath $tempDir) {
    Remove-Item -LiteralPath $tempDir -Recurse -Force -ErrorAction SilentlyContinue
}

& (Join-Path $PSScriptRoot "render_launcher.ps1") -OutputPath $launcherPath -RuntimeDirName "python-runtime" -AppDirName "app"
Write-DeployBatch -OutputPath $deployBat
Write-WinDeployScript -OutputPath $deployPs1
Write-BundleReadme -OutputPath $readmePath -VersionText $resolvedVersion

if (-not $SkipZip) {
    if (Test-Path -LiteralPath $zipPath) {
        if (-not (Test-PathUnderRoot -Path $zipPath -Root $resolvedOutputRoot)) {
            throw "Refusing to overwrite archive outside output root: $zipPath"
        }
        Remove-Item -LiteralPath $zipPath -Force
    }

    Write-Host "[portable] creating zip archive..."
    Compress-Archive -Path $bundleDir -DestinationPath $zipPath -Force
}

Write-Host "[portable] build complete"
Write-Host "[portable] bundle: $bundleDir"
if (-not $SkipZip) {
    Write-Host "[portable] zip: $zipPath"
}

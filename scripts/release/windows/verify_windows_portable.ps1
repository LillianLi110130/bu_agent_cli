param(
    [string]$BundleDir = "",
    [switch]$RunSmoke
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

function Assert-PathExists {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Label
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing ${Label}: $Path"
    }
}

function Assert-PathMissing {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Label
    )

    if (Test-Path -LiteralPath $Path) {
        throw "Unexpected ${Label}: $Path"
    }
}

function Assert-WheelEntry {
    param(
        [Parameter(Mandatory = $true)]
        [string]$WheelPath,
        [Parameter(Mandatory = $true)]
        [string]$EntryPath
    )

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [System.IO.Compression.ZipFile]::OpenRead($WheelPath)
    try {
        $entry = $archive.Entries |
            Where-Object { $_.FullName -eq $EntryPath } |
            Select-Object -First 1
        if (-not $entry) {
            throw "Missing wheel entry ${EntryPath}: $WheelPath"
        }
    }
    finally {
        $archive.Dispose()
    }
}

function Invoke-SmokeRun {
    param(
        [Parameter(Mandatory = $true)]
        [string]$BundleRoot
    )

    $smokeRoot = Join-Path ([System.IO.Path]::GetTempPath()) "crab-smoke-$([guid]::NewGuid().ToString("N").Substring(0, 8))"
    $packageDir = Join-Path $smokeRoot "bundle"
    $homeDir = Join-Path $smokeRoot "home"
    $desktopDir = Join-Path $homeDir "Desktop"
    $workspaceDir = Join-Path $smokeRoot "workspace"
    $installRoot = Join-Path $homeDir ".tg_agent"
    $installedLauncher = Join-Path $installRoot "bin\crab-launcher.bat"
    $commandShim = Join-Path $installRoot "bin\crab.cmd"
    $browserHarnessCommandShim = Join-Path $installRoot "bin\browser-harness.cmd"
    $entryShim = Join-Path $installRoot "bin\crab-entry.py"
    $activeReleaseFile = Join-Path $installRoot "active-release.txt"
    $occupiedProcess = $null

    New-Item -ItemType Directory -Path $desktopDir -Force | Out-Null
    New-Item -ItemType Directory -Path $workspaceDir -Force | Out-Null
    Copy-Item -LiteralPath $BundleRoot -Destination $packageDir -Recurse -Force
    $deployScript = Join-Path $packageDir "win_deploy.ps1"

    $previousUserProfile = $env:USERPROFILE
    $previousHome = $env:HOME
    $previousSkipUpdateCheck = $env:CRAB_SKIP_UPDATE_CHECK

    try {
        $env:USERPROFILE = $homeDir
        $env:HOME = $homeDir
        $env:CRAB_SKIP_UPDATE_CHECK = "1"

        & $deployScript -Update -SkipDesktopShortcut -SkipProtocolRegistration
        if ($LASTEXITCODE -ne 0) {
            throw "win_deploy.ps1 smoke run failed with exit code $LASTEXITCODE"
        }

        Assert-PathExists -Path $installedLauncher -Label "installed launcher"
        Assert-PathExists -Path $commandShim -Label "installed crab command shim"
        Assert-PathExists -Path $browserHarnessCommandShim -Label "installed browser-harness command shim"
        Assert-PathExists -Path $activeReleaseFile -Label "active release pointer"
        $firstRelease = (Get-Content -LiteralPath $activeReleaseFile -Raw -Encoding ASCII).Trim()
        $firstReleaseDir = Join-Path (Join-Path $installRoot "releases") $firstRelease
        Assert-PathExists -Path $firstReleaseDir -Label "first installed release"
        Assert-PathExists -Path (Join-Path $firstReleaseDir ".venv\Scripts\python.exe") `
            -Label "crab virtual environment Python"
        Assert-PathExists -Path (Join-Path $firstReleaseDir ".browser-harness-venv\Scripts\python.exe") `
            -Label "browser-harness virtual environment Python"

        & $deployScript -Update -SkipDesktopShortcut -SkipProtocolRegistration
        if ($LASTEXITCODE -ne 0) {
            throw "win_deploy.ps1 update smoke run failed with exit code $LASTEXITCODE"
        }

        $secondRelease = (Get-Content -LiteralPath $activeReleaseFile -Raw -Encoding ASCII).Trim()
        if ($secondRelease -eq $firstRelease) {
            throw "Update smoke run did not switch the active release"
        }
        Assert-PathExists -Path $firstReleaseDir -Label "retained previous release"

        $occupiedReleaseDir = Join-Path (Join-Path $installRoot "releases") "occupied-smoke-release"
        New-Item -ItemType Directory -Path $occupiedReleaseDir -Force | Out-Null
        $occupiedExecutable = Join-Path $occupiedReleaseDir "ping.exe"
        Copy-Item -LiteralPath (Join-Path $env:SystemRoot "System32\ping.exe") `
            -Destination $occupiedExecutable `
            -Force
        (Get-Item -LiteralPath $occupiedReleaseDir).LastWriteTimeUtc = [DateTime]::UtcNow.AddDays(-10)
        $occupiedProcess = Start-Process `
            -FilePath $occupiedExecutable `
            -ArgumentList @("-n", "600", "127.0.0.1") `
            -WindowStyle Hidden `
            -PassThru

        & $deployScript -Update -SkipDesktopShortcut -SkipProtocolRegistration
        if ($LASTEXITCODE -ne 0) {
            throw "win_deploy.ps1 cleanup smoke run failed with exit code $LASTEXITCODE"
        }

        $thirdRelease = (Get-Content -LiteralPath $activeReleaseFile -Raw -Encoding ASCII).Trim()
        if ($thirdRelease -eq $secondRelease) {
            throw "Cleanup smoke run did not switch the active release"
        }
        Assert-PathMissing -Path $firstReleaseDir -Label "unused stale release"
        Assert-PathExists -Path (Join-Path (Join-Path $installRoot "releases") $secondRelease) `
            -Label "rollback release"
        Assert-PathExists -Path $occupiedReleaseDir -Label "in-use stale release"

        Stop-Process -Id $occupiedProcess.Id -Force -ErrorAction SilentlyContinue
        $occupiedProcess = $null

        Set-Content -LiteralPath $entryShim -Value "raise SystemExit(0)" -Encoding ASCII
        Set-Content -LiteralPath (Join-Path $installRoot "settings.json") `
            -Value "{`"default_workspace`":`"$workspaceDir`"}" `
            -Encoding UTF8
        Remove-Item -LiteralPath $packageDir -Recurse -Force

        Push-Location $workspaceDir
        try {
            cmd /c "`"$installedLauncher`" --help"
            if ($LASTEXITCODE -ne 0) {
                throw "installed launcher smoke run failed with exit code $LASTEXITCODE"
            }

            cmd /c "`"$commandShim`" --help"
            if ($LASTEXITCODE -ne 0) {
                throw "installed crab command smoke run failed with exit code $LASTEXITCODE"
            }

            cmd /c "`"$browserHarnessCommandShim`" --version"
            if ($LASTEXITCODE -ne 0) {
                throw "installed browser-harness command smoke run failed with exit code $LASTEXITCODE"
            }
        }
        finally {
            Pop-Location
        }
    }
    finally {
        if ($null -ne $occupiedProcess) {
            Stop-Process -Id $occupiedProcess.Id -Force -ErrorAction SilentlyContinue
        }
        $env:USERPROFILE = $previousUserProfile
        $env:HOME = $previousHome
        $env:CRAB_SKIP_UPDATE_CHECK = $previousSkipUpdateCheck
        if (Test-Path -LiteralPath $smokeRoot) {
            Remove-Item -LiteralPath $smokeRoot -Recurse -Force
        }
    }
}

$repoRoot = Get-RepoRoot
$projectVersion = Get-ProjectVersion -PyprojectPath (Join-Path $repoRoot "pyproject.toml")
$resolvedBundleDir = if ($BundleDir) {
    [System.IO.Path]::GetFullPath($BundleDir)
}
else {
    [System.IO.Path]::GetFullPath((Join-Path $repoRoot "dist\release\tg-agent-windows-x64-v$projectVersion-portable"))
}

$appDir = Join-Path $resolvedBundleDir "app"
$runtimeDir = Join-Path $resolvedBundleDir "python-runtime"
$wheelhouseDir = Join-Path $resolvedBundleDir "wheelhouse"
$shortcutIcon = Join-Path $resolvedBundleDir "crab.ico"

Assert-PathExists -Path $resolvedBundleDir -Label "bundle directory"
Assert-PathExists -Path (Join-Path $resolvedBundleDir "deploy.bat") -Label "deploy.bat"
Assert-PathExists -Path (Join-Path $resolvedBundleDir "win_deploy.ps1") -Label "win_deploy.ps1"
Assert-PathExists -Path (Join-Path $resolvedBundleDir "crab-launcher.bat") -Label "crab-launcher.bat"
Assert-PathExists -Path (Join-Path $resolvedBundleDir "README.txt") -Label "README.txt"
Assert-PathExists -Path $appDir -Label "app directory"
Assert-PathExists -Path $runtimeDir -Label "python-runtime directory"
Assert-PathExists -Path (Join-Path $runtimeDir "python.exe") -Label "bundled python.exe"
Assert-PathExists -Path (Join-Path $runtimeDir "Lib") -Label "bundled Lib"
Assert-PathExists -Path (Join-Path $runtimeDir "DLLs") -Label "bundled DLLs"
Assert-PathExists -Path $wheelhouseDir -Label "wheelhouse directory"
Assert-PathExists -Path (Join-Path $appDir "tg_crab_main.py") -Label "packaged tg_crab_main.py"
Assert-PathExists -Path (Join-Path $appDir "agent_core") -Label "packaged agent_core"
Assert-PathExists -Path (Join-Path $appDir "agent_core\updater.py") -Label "packaged updater"
Assert-PathExists -Path (Join-Path $appDir "cli") -Label "packaged cli"
Assert-PathExists -Path (Join-Path $appDir "tools") -Label "packaged tools"
Assert-PathExists -Path (Join-Path $appDir ".env") -Label "packaged .env"
Assert-PathExists -Path (Join-Path $appDir "tg_crab_worker.json") -Label "packaged worker config"

$projectWheel = Get-ChildItem -LiteralPath $wheelhouseDir -Filter "tg_agent_cli-*.whl" -ErrorAction SilentlyContinue |
    Select-Object -First 1
if (-not $projectWheel) {
    throw "Project wheel not found in bundled wheelhouse: $wheelhouseDir"
}
Assert-WheelEntry -WheelPath $projectWheel.FullName -EntryPath "tg_agent_cli/agent_core/updater.py"

Write-Host "[portable] structure verification passed: $resolvedBundleDir"
if (Test-Path -LiteralPath $shortcutIcon) {
    Write-Host "[portable] shortcut icon detected: $shortcutIcon"
}

if ($RunSmoke) {
    Invoke-SmokeRun -BundleRoot $resolvedBundleDir
    Write-Host "[portable] smoke verification passed"
}

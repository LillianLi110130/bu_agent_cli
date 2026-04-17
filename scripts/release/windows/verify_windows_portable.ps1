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

function Invoke-SmokeRun {
    param(
        [Parameter(Mandatory = $true)]
        [string]$BundleRoot
    )

    $deployScript = Join-Path $BundleRoot "win_deploy.ps1"
    $launcher = Join-Path $BundleRoot "crab-launcher.bat"
    $smokeRoot = Join-Path $BundleRoot ".tmp_portable_smoke"
    $homeDir = Join-Path $smokeRoot "home"
    $desktopDir = Join-Path $homeDir "Desktop"
    $workspaceDir = Join-Path $smokeRoot "workspace"

    if (Test-Path -LiteralPath $smokeRoot) {
        Remove-Item -LiteralPath $smokeRoot -Recurse -Force
    }

    New-Item -ItemType Directory -Path $desktopDir -Force | Out-Null
    New-Item -ItemType Directory -Path $workspaceDir -Force | Out-Null

    $previousUserProfile = $env:USERPROFILE
    $previousHome = $env:HOME

    try {
        $env:USERPROFILE = $homeDir
        $env:HOME = $homeDir

        & $deployScript -SkipDesktopShortcut
        if ($LASTEXITCODE -ne 0) {
            throw "win_deploy.ps1 smoke run failed with exit code $LASTEXITCODE"
        }

        Push-Location $workspaceDir
        try {
            cmd /c "`"$launcher`" --help"
            if ($LASTEXITCODE -ne 0) {
                throw "launcher smoke run failed with exit code $LASTEXITCODE"
            }
        }
        finally {
            Pop-Location
        }
    }
    finally {
        $env:USERPROFILE = $previousUserProfile
        $env:HOME = $previousHome
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
Assert-PathExists -Path (Join-Path $appDir "cli") -Label "packaged cli"
Assert-PathExists -Path (Join-Path $appDir "tools") -Label "packaged tools"
Assert-PathExists -Path (Join-Path $appDir ".env") -Label "packaged .env"
Assert-PathExists -Path (Join-Path $appDir "tg_crab_worker.json") -Label "packaged worker config"

$projectWheel = Get-ChildItem -LiteralPath $wheelhouseDir -Filter "tg_agent_cli-*.whl" -ErrorAction SilentlyContinue |
    Select-Object -First 1
if (-not $projectWheel) {
    throw "Project wheel not found in bundled wheelhouse: $wheelhouseDir"
}

Write-Host "[portable] structure verification passed: $resolvedBundleDir"

if ($RunSmoke) {
    Invoke-SmokeRun -BundleRoot $resolvedBundleDir
    Write-Host "[portable] smoke verification passed"
}

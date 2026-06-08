param(
    [Parameter(Mandatory = $true)]
    [string]$OutputPath,
    [string]$RuntimeDirName = "python-runtime",
    [string]$AppDirName = "app"
)

$ErrorActionPreference = "Stop"

$batContent = @'
@echo off
setlocal
set "POWERSHELL_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%POWERSHELL_EXE%" set "POWERSHELL_EXE=%WINDIR%\System32\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%POWERSHELL_EXE%" set "POWERSHELL_EXE=powershell.exe"
"%POWERSHELL_EXE%" -NoProfile -ExecutionPolicy Bypass -File "%~dpn0.ps1" %*
set "EXITCODE=%ERRORLEVEL%"
exit /b %EXITCODE%
'@

$ps1Content = @'
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CliArgs
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Decode-Text {
    param([string]$Base64)

    return [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($Base64))
}

$bundleDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$installRoot = Join-Path $env:USERPROFILE ".tg_agent"
$activeReleaseFile = Join-Path $installRoot "active-release.txt"
$venvPython = Join-Path $installRoot ".venv\Scripts\python.exe"
if (Test-Path -LiteralPath $activeReleaseFile) {
    $activeRelease = (Get-Content -LiteralPath $activeReleaseFile -Raw -Encoding ASCII).Trim()
    if (-not [string]::IsNullOrWhiteSpace($activeRelease)) {
        $venvPython = Join-Path $installRoot "releases\$activeRelease\.venv\Scripts\python.exe"
    }
}
$entryShim = Join-Path $installRoot "bin\crab-entry.py"
$settingsPath = Join-Path $installRoot "settings.json"
$workspace = (Get-Location).Path
$defaultWorkspace = ""
$manualWorkspaceSelected = $false

function Save-DefaultWorkspace {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SettingsFilePath,
        [Parameter(Mandatory = $true)]
        [string]$WorkspacePath
    )

    $settings = @{}
    if (Test-Path -LiteralPath $SettingsFilePath) {
        try {
            $loaded = Get-Content -LiteralPath $SettingsFilePath -Raw -Encoding UTF8 | ConvertFrom-Json -AsHashtable
            if ($loaded -is [hashtable]) {
                $settings = $loaded
            }
        }
        catch {
        }
    }

    $settings["default_workspace"] = [System.IO.Path]::GetFullPath($WorkspacePath)
    $settingsDir = Split-Path -Parent $SettingsFilePath
    if ($settingsDir) {
        New-Item -ItemType Directory -Path $settingsDir -Force | Out-Null
    }

    $json = $settings | ConvertTo-Json -Depth 10
    [System.IO.File]::WriteAllText($SettingsFilePath, $json + [Environment]::NewLine, [System.Text.UTF8Encoding]::new($false))
}

function Get-DefaultWorkspace {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SettingsFilePath
    )

    if (-not (Test-Path -LiteralPath $SettingsFilePath)) {
        return ""
    }

    try {
        $settings = Get-Content -LiteralPath $SettingsFilePath -Raw -Encoding UTF8 | ConvertFrom-Json
        $value = [string]$settings.default_workspace
        if (-not [string]::IsNullOrWhiteSpace($value)) {
            return $value.Trim()
        }
    }
    catch {
    }

    return ""
}

$msgPortableNotInstalled = Decode-Text "Y3JhYiDkvr/mkLrnjq/looPlsJrmnKrlronoo4U="
$msgRunDeployFirst = Decode-Text "6K+35YWI5Zyo5L6/5pC65YyF5qC555uu5b2V6L+Q6KGMIGRlcGxveS5iYXQ="
$msgEntryShimMissing = Decode-Text "Y3JhYiDlkK/liqjlhaXlj6PnvLrlpLE="
$msgCurrentStartupDir = Decode-Text "5b2T5YmN5ZCv5Yqo55uu5b2VOiB7MH0="
$msgWorkspacePrompt = Decode-Text "6K+36L6T5YWl5bel5L2c5Yy66Lev5b6EKOebtOaOpeWbnui9puS9v+eUqOW9k+WJjeebruW9lSk="
$msgWorkspaceMissing = Decode-Text "5bel5L2c5Yy655uu5b2V5LiN5a2Y5ZyoOiB7MH0="
$msgPressAnyKey = Decode-Text "6K+35oyJ5Lu75oSP6ZSu5YWz6Zet5q2k56qX5Y+j"
$msgStartingCrab = Decode-Text "5q2j5Zyo5ZCv5YqoIGNyYWIuLi4="
$msgWorkspaceLabel = Decode-Text "5bel5L2c5Yy6OiB7MH0="
$msgCannotSwitchWorkspace = Decode-Text "5peg5rOV5YiH5o2i5Yiw5bel5L2c5Yy655uu5b2VOiB7MH0="
$msgExitCode = Decode-Text "Y3JhYiDlt7LpgIDlh7osIOmAgOWHuueggTogezB9"
$msgConfirmDefaultWorkspace = Decode-Text "5piv5ZCm5bCG6K+l6Lev5b6E6K6+5Li66buY6K6k5bel5L2c6Lev5b6E77yf5Lul5ZCO5omT5byA5bCP6J6D6J+56YO95bCG6L+Z5Liq6Lev5b6E5L2c5Li65bel5L2c6Lev5b6EICh5L04p"
$msgDefaultWorkspaceSaved = Decode-Text "5bey5bCG6K+l6Lev5b6E5L+d5a2Y5Li66buY6K6k5bel5L2c6Lev5b6E"
$msgDefaultWorkspaceUnchanged = Decode-Text "5pyq5L+u5pS56buY6K6k5bel5L2c6Lev5b6E"
$msgDefaultWorkspaceSaveFailed = Decode-Text "5L+d5a2Y6buY6K6k5bel5L2c6Lev5b6E5aSx6LSlOiB7MH0="

if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host $msgPortableNotInstalled
    Write-Host $msgRunDeployFirst
    exit 1
}

if (-not (Test-Path -LiteralPath $entryShim)) {
    Write-Host $msgEntryShimMissing
    Write-Host $msgRunDeployFirst
    exit 1
}

$defaultWorkspace = Get-DefaultWorkspace -SettingsFilePath $settingsPath
if ([string]::IsNullOrWhiteSpace($defaultWorkspace)) {
    Write-Host ($msgCurrentStartupDir -f $workspace)
    $userWorkspaceInput = Read-Host $msgWorkspacePrompt
    if (-not [string]::IsNullOrWhiteSpace($userWorkspaceInput)) {
        $workspace = $userWorkspaceInput.Trim().Trim('"')
        $manualWorkspaceSelected = $true
    }
}
else {
    $workspace = $defaultWorkspace
}

if (-not (Test-Path -LiteralPath $workspace -PathType Container)) {
    Write-Host ($msgWorkspaceMissing -f $workspace)
    Write-Host $msgPressAnyKey
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

if ($manualWorkspaceSelected) {
    $saveChoice = Read-Host $msgConfirmDefaultWorkspace
    if ($saveChoice -match '^(?i:y|yes)$') {
        try {
            Save-DefaultWorkspace -SettingsFilePath $settingsPath -WorkspacePath $workspace
            Write-Host $msgDefaultWorkspaceSaved
        }
        catch {
            Write-Host ($msgDefaultWorkspaceSaveFailed -f $_.Exception.Message)
        }
    }
    else {
        Write-Host $msgDefaultWorkspaceUnchanged
    }
}

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONDONTWRITEBYTECODE = "1"

if ($env:CRAB_SKIP_UPDATE_CHECK -ne "1") {
    & $venvPython -m agent_core.updater check-before-launch
    $updateExitCode = $LASTEXITCODE
    if ($updateExitCode -eq 20) {
        $pendingUpdate = Join-Path $installRoot "updates\pending_update.ps1"
        $powerShellExe = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
        if (-not (Test-Path -LiteralPath $powerShellExe)) {
            $powerShellExe = "powershell.exe"
        }
        & $powerShellExe -NoProfile -ExecutionPolicy Bypass -File $pendingUpdate
        $pendingExitCode = $LASTEXITCODE
        if ($pendingExitCode -ne 0) {
            exit $pendingExitCode
        }
    }
    elseif ($updateExitCode -ne 0) {
        exit $updateExitCode
    }
}

Write-Host $msgStartingCrab
Write-Host ($msgWorkspaceLabel -f $workspace)
Write-Host ""

$previousLocation = Get-Location
try {
    Set-Location -LiteralPath $workspace
}
catch {
    Write-Host ($msgCannotSwitchWorkspace -f $workspace)
    Write-Host $msgPressAnyKey
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

try {
    & $venvPython -u $entryShim @CliArgs
    $exitCode = $LASTEXITCODE
}
finally {
    Set-Location -LiteralPath $previousLocation
}

if ($exitCode -ne 0) {
    Write-Host ""
    Write-Host ($msgExitCode -f $exitCode)
    Write-Host $msgPressAnyKey
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}

exit $exitCode
'@

$outputDir = Split-Path -Parent $OutputPath
if ($outputDir) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}

$launcherPs1Path = [System.IO.Path]::ChangeExtension($OutputPath, ".ps1")
$utf8WithBom = New-Object System.Text.UTF8Encoding($true)
[System.IO.File]::WriteAllText($OutputPath, $batContent, [System.Text.Encoding]::ASCII)
[System.IO.File]::WriteAllText($launcherPs1Path, $ps1Content, $utf8WithBom)
Write-Host "Launcher written to $OutputPath"
Write-Host "PowerShell launcher written to $launcherPs1Path"

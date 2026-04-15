# One-click build: console -> conda-pack -> NSIS .exe. Run from repo root.
# Requires: conda, node/npm (for console), NSIS (makensis) on PATH.

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot
Set-Location $RepoRoot
Write-Host "[build_win] REPO_ROOT=$RepoRoot"
$PackDir = $PSScriptRoot
$Dist = $PackDir
$Archive = Join-Path $Dist "cmb-pal-env.zip"
$Unpacked = Join-Path $Dist "CMBPal-unpacked"
$NsiPath = Join-Path $PackDir "copaw_desktop.nsi"

# Packages affected by conda-unpack bug on Windows (conda-pack Issue #154)
# conda-unpack corrupts Python string escaping when replacing path prefixes.
# Example: "\\\\?\\" (correct) -> "\\" (SyntaxError)
# Solution: Reinstall these packages after conda-unpack to restore correct files.
# See: issue.md, scripts/pack/WINDOWS_FIX.md
$CondaUnpackAffectedPackages = @(
  "huggingface_hub"  # Uses Windows extended-length path prefix (\\?\)
)

Write-Host "== Unpacking env =="
#if (Test-Path $Unpacked) { Remove-Item -Recurse -Verbose -Force $Unpacked }
#New-Item -ItemType Directory -Path $Unpacked -Force | Out-Null
#Expand-Archive -Path $Archive -DestinationPath $Unpacked -Force
#tar -vxf $Archive -C $Unpacked
$unpackedRoot = Get-ChildItem -Path $Unpacked -ErrorAction SilentlyContinue | Measure-Object
Write-Host "[build_win] Unpacked entries in $Unpacked : $($unpackedRoot.Count)"

# Resolve env root: conda-pack usually puts python.exe at archive root; allow one nested dir.
$EnvRoot = $Unpacked
if (-not (Test-Path (Join-Path $EnvRoot "python.exe"))) {
  $found = Get-ChildItem -Path $Unpacked -Directory -ErrorAction SilentlyContinue |
    Where-Object { Test-Path (Join-Path $_.FullName "python.exe") } |
    Select-Object -First 1
  if ($found) { $EnvRoot = $found.FullName; Write-Host "[build_win] Env root: $EnvRoot" }
}
if (-not (Test-Path (Join-Path $EnvRoot "python.exe"))) {
  throw "python.exe not found in unpacked env (checked $Unpacked and one level down)."
}
if (-not [System.IO.Path]::IsPathRooted($EnvRoot)) {
  $EnvRoot = Join-Path $RepoRoot $EnvRoot
}
Write-Host "[build_win] python.exe found at env root: $EnvRoot"
# Rewrite prefix in packed env so paths point to current location (required after move).
$CondaUnpack = Join-Path $EnvRoot "Scripts\conda-unpack.exe"
if (Test-Path $CondaUnpack) {
  Write-Host "[build_win] Running conda-unpack..."
  & $CondaUnpack
  if ($LASTEXITCODE -ne 0) { throw "conda-unpack failed with exit code $LASTEXITCODE" }

  # Fix conda-unpack bug: it corrupts Python string escaping on Windows
  # See: issue.md and https://github.com/conda/conda-pack/issues/154
  # Solution: Reinstall affected packages using cached wheels
  Write-Host "[build_win] Fixing conda-unpack corruption by reinstalling affected packages..."
  $WheelsCache = Join-Path $RepoRoot "cache\conda_unpack_wheels"
  if (Test-Path $WheelsCache) {
    $pythonExe = Join-Path $EnvRoot "python.exe"

    foreach ($pkg in $CondaUnpackAffectedPackages) {
      Write-Host "  Reinstalling $pkg..."
      & $pythonExe -m pip install --force-reinstall --no-deps `
        --find-links $WheelsCache --no-index $pkg
      if ($LASTEXITCODE -ne 0) {
        Write-Host "  WARN: Failed to reinstall $pkg (exit code: $LASTEXITCODE)" -ForegroundColor Yellow
      }
    }

    # Verify the fix worked
    Write-Host "[build_win] Verifying fix..."
    & $pythonExe -c "from huggingface_hub import file_download; print('✓ huggingface_hub import OK')"
    if ($LASTEXITCODE -ne 0) {
      throw "CRITICAL: huggingface_hub still has import errors after reinstall. See issue.md"
    }
    Write-Host "[build_win] conda-unpack corruption fixed successfully."
  } else {
    Write-Host "[build_win] WARN: wheels_cache not found at $WheelsCache" -ForegroundColor Yellow
    Write-Host "[build_win] WARN: Cannot fix conda-unpack corruption. App may fail to start." -ForegroundColor Yellow
  }
} else {
  Write-Host "[build_win] WARN: conda-unpack.exe not found at $CondaUnpack, skipping."
}
$LauncherBat = Join-Path $PSScriptRoot "cmbPal_launcher.bat"
@"
@echo off
cd /d "%~dp0"

REM Preserve system PATH for accessing system commands
REM Prepend packaged env to PATH so packaged Python takes precedence
set "NODE_PATH=%~dp0node-v22.18.0-win-x64"
set "PYTHON_PATH=%~dp0CMBPal-unpacked\Scripts;%~dp0CMBPal-unpacked"

set "PATH=%NODE_PATH%;%PYTHON_PATH%;%PATH%"
set "COPAW_CMB_SKILLS_HUB_BASE_URL=http://skills.paas.cmbchina.cn"
REM Log level: env var COPAW_LOG_LEVEL or default to "info"
if not defined COPAW_LOG_LEVEL set "COPAW_LOG_LEVEL=info"

REM Set SSL certificate paths for packaged environment
REM Use temp file to avoid for /f blocking issue in bat scripts
set "CERT_TMP=%TEMP%\copaw_cert_%RANDOM%.txt"
"%~dp0python.exe" -u -c "import certifi; print(certifi.where())" > "%CERT_TMP%" 2>nul
set /p CERT_FILE=<"%CERT_TMP%"
del "%CERT_TMP%" 2>nul
if defined CERT_FILE (
  if exist "%CERT_FILE%" (
    set "SSL_CERT_FILE=%CERT_FILE%"
    set "REQUESTS_CA_BUNDLE=%CERT_FILE%"
    set "CURL_CA_BUNDLE=%CERT_FILE%"
  )
)

set "DEFAULT_PORT=8088"
echo Input port (Press Enter to use default %DEFAULT_PORT%)
set /p PORT=
if "%PORT%"=="" set "PORT=%DEFAULT_PORT%"
echo Using port: %PORT%

if not exist "%USERPROFILE%\.copaw\workspaces\default\agent.json" (
  "%~dp0CMBPal-unpacked\python.exe" -u -m copaw init --defaults --accept-security
)
"%~dp0CMBPal-unpacked\python.exe" -u -m copaw app --port %PORT% --log-level %COPAW_LOG_LEVEL%
"@ | Set-Content -Path $LauncherBat -Encoding ASCII
$WshShell = New-Object -ComObject WScript.Shell

# 快捷方式路径（桌面）
$ShortcutPath = "$env:USERPROFILE\Desktop\cmbPal_launcher.lnk"

# 目标 bat 路径（当前目录）
$TargetPath = $LauncherBat

# 创建快捷方式
$Shortcut = $WshShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $TargetPath

# 工作目录（很重要）
$Shortcut.WorkingDirectory = $PSScriptRoot

# 图标（可选）
$Shortcut.IconLocation = Join-Path $PSScriptRoot "CmbPal.ico"

# 保存
$Shortcut.Save()

Write-Host "Shortcut created at $ShortcutPath"

Write-Host "Press any key to exit..."
$x = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
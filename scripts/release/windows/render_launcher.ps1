param(
    [Parameter(Mandatory = $true)]
    [string]$OutputPath,
    [string]$RuntimeDirName = "python-runtime",
    [string]$AppDirName = "app"
)

$ErrorActionPreference = "Stop"

$launcherContent = @'
@echo off
setlocal EnableExtensions

set "BUNDLE_DIR=%~dp0"
if "%BUNDLE_DIR:~-1%"=="\" set "BUNDLE_DIR=%BUNDLE_DIR:~0,-1%"

set "APP_DIR=%BUNDLE_DIR%\__APP_DIR__"
set "INSTALL_ROOT=%USERPROFILE%\.tg_agent"
set "VENV_PYTHON=%INSTALL_ROOT%\.venv\Scripts\python.exe"
set "ENTRY_SHIM=%INSTALL_ROOT%\bin\crab-entry.py"
set "ORIGINAL_CWD=%CD%"
set "WORKSPACE=%CD%"
set "USER_WORKSPACE_INPUT="

if not exist "%VENV_PYTHON%" (
    echo crab portable environment is not installed.
    echo Run deploy.bat in the bundle root first.
    set "EXITCODE=1"
    goto :finish
)

if not exist "%ENTRY_SHIM%" (
    echo crab entry shim is missing.
    echo Run deploy.bat in the bundle root first.
    set "EXITCODE=1"
    goto :finish
)

echo Current startup directory: %WORKSPACE%
set /p "USER_WORKSPACE_INPUT=Enter workspace path (press Enter to use current directory): "
if defined USER_WORKSPACE_INPUT (
    set "WORKSPACE=%USER_WORKSPACE_INPUT:"=%"
)

if not exist "%WORKSPACE%" (
    echo Workspace directory does not exist: %WORKSPACE%
    set "EXITCODE=1"
    goto :finish
)

set "PYTHONIOENCODING=utf-8"
set "PYTHONDONTWRITEBYTECODE=1"

echo Starting crab...
echo Workspace: %WORKSPACE%
echo.

pushd "%WORKSPACE%" >nul
"%VENV_PYTHON%" -u "%ENTRY_SHIM%" %*
set "EXITCODE=%ERRORLEVEL%"
popd >nul

:finish
if not defined EXITCODE set "EXITCODE=0"

if not "%EXITCODE%"=="0" (
    echo.
    echo crab exited with code %EXITCODE%.
    echo Press any key to close this window.
    pause >nul
    exit /b %EXITCODE%
)

exit /b %EXITCODE%
'@

$launcherContent = $launcherContent.Replace("__APP_DIR__", $AppDirName)
$launcherContent = $launcherContent.Replace("__RUNTIME_DIR__", $RuntimeDirName)

$outputDir = Split-Path -Parent $OutputPath
if ($outputDir) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}

Set-Content -LiteralPath $OutputPath -Value $launcherContent -Encoding ASCII
Write-Host "Launcher written to $OutputPath"

param(
    [string]$SourceExe = (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "tg-agent.exe"),
    [string]$InstallRoot = $(if ($env:TG_AGENT_HOME) { $env:TG_AGENT_HOME } else { Join-Path $HOME ".tg_agent" }),
    [switch]$SkipConfig
)

$ErrorActionPreference = "Stop"

function Get-DefaultEnvContent {
    return @(
        "# tg-agent runtime configuration",
        "# Fill in OPENAI_API_KEY after install if your shell or workspace does not already provide it.",
        "OPENAI_API_KEY=",
        "LLM_MODEL=GLM-4.7",
        "LLM_BASE_URL=https://open.bigmodel.cn/api/coding/paas/v4"
    ) -join "`r`n"
}

$installRootResolved = [System.IO.Path]::GetFullPath($InstallRoot)
$binDir = Join-Path $installRootResolved "bin"
$targetExe = Join-Path $binDir "tg-agent.exe"
$envFile = Join-Path $installRootResolved ".env"
$workerConfig = Join-Path $installRootResolved "tg_crab_worker.json"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$configTemplate = Join-Path $scriptDir "tg_crab_worker.json"

New-Item -ItemType Directory -Path $binDir -Force | Out-Null
Copy-Item -LiteralPath $SourceExe -Destination $targetExe -Force

if ((Test-Path $configTemplate) -and -not (Test-Path $workerConfig)) {
    Copy-Item -LiteralPath $configTemplate -Destination $workerConfig -Force
}

if (-not $SkipConfig -and -not (Test-Path $envFile)) {
    Set-Content -LiteralPath $envFile -Value (Get-DefaultEnvContent) -Encoding UTF8
}

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
$pathEntries = @()
if ($userPath) {
    $pathEntries = $userPath.Split(";") | Where-Object { $_ }
}
if ($pathEntries -notcontains $binDir) {
    $newPath = if ($userPath) { "$userPath;$binDir" } else { $binDir }
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
}

Write-Host "Installed tg-agent to $targetExe"
Write-Host "User config directory: $installRootResolved"
Write-Host "Config file: $envFile"
Write-Host "Restart your terminal to pick up PATH changes."

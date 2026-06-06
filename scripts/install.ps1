$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

Write-Host "==> Installing droneai-security..."
python -m pip install --upgrade pip
python -m pip install .

Write-Host "==> Creating default config..."
droneai init

Write-Host ""
Write-Host "Installed. Quick start:"
Write-Host "  droneai test --offline"
Write-Host "  droneai"
Write-Host "  droneai fetch-video"

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

Write-Host "==> Installing build dependencies..."
python -m pip install --upgrade pip
python -m pip install ".[dev]"

Write-Host "==> Building executables with PyInstaller..."
python -m PyInstaller droneai.spec --noconfirm --clean

$Dist = Join-Path $Root "dist\droneai-security"
if (-not (Test-Path $Dist)) {
    throw "Build failed: dist\droneai-security not found"
}

Copy-Item -Force "config.example.yaml" (Join-Path $Dist "config.example.yaml")
Copy-Item -Force "README.md" (Join-Path $Dist "README.txt")
Copy-Item -Force "LICENSE" (Join-Path $Dist "LICENSE")

foreach ($weight in @("weapon_detection_custom.pt", "yolov8n.pt")) {
    if (Test-Path (Join-Path $Root $weight)) {
        Copy-Item -Force (Join-Path $Root $weight) (Join-Path $Dist $weight)
        Write-Host "  Bundled $weight"
    }
}

$Zip = Join-Path $Root "dist\droneai-security-windows.zip"
if (Test-Path $Zip) { Remove-Item $Zip -Force }
Compress-Archive -Path $Dist -DestinationPath $Zip

Write-Host ""
Write-Host "Done."
Write-Host "  Folder: $Dist"
Write-Host "  Zip:    $Zip"
Write-Host ""
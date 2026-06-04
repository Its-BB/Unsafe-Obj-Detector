$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
py download_kaggle_dataset.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
py prepare_dataset.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
py train_max_accuracy.py @args
exit $LASTEXITCODE

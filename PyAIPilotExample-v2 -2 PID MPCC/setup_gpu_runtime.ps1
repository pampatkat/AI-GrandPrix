$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
$runtimeRoot = Join-Path $projectRoot ".runtime"
$gpuEnvironment = Join-Path $projectRoot ".venv-gpu"
$gpuPython = Join-Path $gpuEnvironment "Scripts\python.exe"

New-Item -ItemType Directory -Force -Path `
    (Join-Path $runtimeRoot "temp"), `
    (Join-Path $runtimeRoot "pip-cache") | Out-Null

$env:TEMP = Join-Path $runtimeRoot "temp"
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = Join-Path $runtimeRoot "pip-cache"

python -m venv --system-site-packages $gpuEnvironment
& $gpuPython -m pip install --upgrade pip
& $gpuPython -m pip install "onnxruntime-gpu[cuda,cudnn]==1.23.2"

Write-Host "GPU runtime installed on E:. Run .\run_gpu.ps1"

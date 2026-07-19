$ErrorActionPreference = "Stop"

$env:GATE_ONNX_PROVIDER = "cuda"
if (-not $env:FPV_DISPLAY) {
    $env:FPV_DISPLAY = "0"
}
if (-not $env:INITIAL_HOVER_THRUST) {
    $env:INITIAL_HOVER_THRUST = "0.28"
}
$gpuPython = Join-Path $PSScriptRoot ".venv-gpu\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $gpuPython)) {
    throw "GPU runtime is missing. Run .\setup_gpu_runtime.ps1 first."
}

& $gpuPython (Join-Path $PSScriptRoot "main.py")

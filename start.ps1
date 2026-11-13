# VR180 SBS 换脸工作台 —— 启动（Windows）
# 前置：已运行 setup.ps1 完成安装。
# 用法：powershell -ExecutionPolicy Bypass -File start.ps1

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:FACEFUSION_HOME = Join-Path $repoRoot '.facefusion'

if (-not (Test-Path (Join-Path $env:FACEFUSION_HOME 'facefusion.py'))) {
    Write-Host "未找到 FaceFusion（$env:FACEFUSION_HOME）。请先运行 setup.ps1。" -ForegroundColor Red
    exit 1
}

function Find-CondaExe {
    if ($env:CONDA_EXE -and (Test-Path $env:CONDA_EXE)) { return $env:CONDA_EXE }
    $cmd = Get-Command conda -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source) { return $cmd.Source }
    foreach ($base in @("$env:USERPROFILE\anaconda3", "$env:USERPROFILE\miniconda3")) {
        $exe = Join-Path $base 'Scripts\conda.exe'
        if (Test-Path $exe) { return $exe }
    }
    return $null
}

$conda = Find-CondaExe
$condaRoot = Split-Path -Parent (Split-Path -Parent $conda)
$envPython = Join-Path $condaRoot 'envs\facefusion\python.exe'

if (-not (Test-Path $envPython)) {
    Write-Host "未找到 facefusion 环境的 Python：$envPython" -ForegroundColor Red
    Write-Host "请先运行 setup.ps1。" -ForegroundColor Red
    exit 1
}

Write-Host "启动工作台：http://127.0.0.1:8080/"
& $envPython (Join-Path $repoRoot 'workbench\server.py') --port 8080

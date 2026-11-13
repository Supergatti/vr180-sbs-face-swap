# VR180 SBS 换脸工作台 —— 一键安装（Windows）
# 前置：已安装 Miniconda/Anaconda 与 Git；ffmpeg / ffprobe 在 PATH。
# 用法：powershell -ExecutionPolicy Bypass -File setup.ps1

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

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
if (-not $conda) {
    Write-Host "未找到 conda。请先安装 Miniconda 或 Anaconda。" -ForegroundColor Red
    exit 1
}
Write-Host "conda: $conda"

$envName = 'facefusion'
$condaRoot = Split-Path -Parent (Split-Path -Parent $conda)
$envPython = Join-Path $condaRoot "envs\$envName\python.exe"

if (-not (Test-Path $envPython)) {
    Write-Host "创建 conda 环境 $envName (python 3.10) ..."
    & $conda create -n $envName python=3.10 -y
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "使用环境 Python: $envPython"
& $envPython (Join-Path $repoRoot 'scripts\bootstrap.py')
exit $LASTEXITCODE

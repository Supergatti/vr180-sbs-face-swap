#!/usr/bin/env python3
"""一键安装引导：拉取 FaceFusion、安装依赖、下载权重、打补丁。

本脚本应使用 `facefusion` conda 环境里的 python 运行（setup.ps1 会处理好），
它会在仓库根目录下完成：
  1. 浅克隆 FaceFusion 3.1.2 到 `.facefusion/`
  2. 用 FaceFusion 自带的 install.py 安装依赖（含 onnxruntime-gpu）
  3. 下载本工作台专有的 VR180 检测模型
  4. 应用进度回传补丁（可选，失败不阻断）
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FACEFUSION_DIR = REPO_ROOT / '.facefusion'
FACEFUSION_REPO = 'https://github.com/facefusion/facefusion.git'
FACEFUSION_TAG = '3.1.2'


def run(cmd, cwd=None):
    print('+', ' '.join(map(str, cmd)), flush=True)
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode:
        sys.exit(result.returncode)


def main() -> int:
    if not shutil.which('git'):
        print('未找到 git，请先安装 Git 并加入 PATH。', file=sys.stderr)
        return 1

    # 1. 拉取 FaceFusion（浅克隆，固定到 3.1.2，与进度补丁匹配）
    if not (FACEFUSION_DIR / 'facefusion.py').is_file():
        FACEFUSION_DIR.parent.mkdir(parents=True, exist_ok=True)
        run(['git', 'clone', '--depth', '1', '--branch', FACEFUSION_TAG, FACEFUSION_REPO, str(FACEFUSION_DIR)], cwd=REPO_ROOT)
    else:
        print(f'FaceFusion 已存在：{FACEFUSION_DIR}')

    # 2. 安装 FaceFusion 依赖（含 onnxruntime-gpu 1.20.1）
    run([sys.executable, 'install.py', '--onnxruntime', 'cuda', '--skip-conda'], cwd=FACEFUSION_DIR)

    # 3. 下载本工作台专有的 VR180 检测模型
    run([sys.executable, str(REPO_ROOT / 'scripts' / 'download-models.py')])

    # 4. 应用进度回传补丁（可选，失败不阻断）
    progress_mod = REPO_ROOT / 'facefusion' / 'workbench_progress.py'
    patch = REPO_ROOT / 'facefusion' / 'workbench-progress.patch'
    shutil.copy2(progress_mod, FACEFUSION_DIR / 'facefusion' / 'workbench_progress.py')
    applied = subprocess.run(['git', 'apply', str(patch)], cwd=FACEFUSION_DIR)
    if applied.returncode == 0:
        print('进度回传补丁已应用。')
    else:
        print('提示：进度回传补丁未应用（可能是 FaceFusion 版本差异）。不影响换脸，仅缺实时进度。')

    print()
    print('安装完成。启动：')
    print('  powershell -ExecutionPolicy Bypass -File start.ps1')
    print('首次换脸时 FaceFusion 会自动下载其模型（inswapper / gfpgan 等）。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

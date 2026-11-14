#!/usr/bin/env python3
"""一键安装引导：拉取 FaceFusion、安装依赖、下载权重、打补丁。

本脚本应使用 `facefusion` conda 环境里的 python 运行（setup.ps1 会处理好），
它会在仓库根目录下完成：
  1. 浅克隆 FaceFusion 3.1.2 到 `.facefusion/`
  2. 用 FaceFusion 自带的 install.py 安装依赖（含 onnxruntime-gpu）
  3. 下载 VR180 检测权重与默认 FaceFusion 模型
  4. 应用工作台兼容补丁
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


def apply_patch(name, optional=False):
    patch = REPO_ROOT / 'facefusion' / name
    base = ['git', 'apply']
    if subprocess.run([*base, '--reverse', '--check', str(patch)], cwd=FACEFUSION_DIR, capture_output=True).returncode == 0:
        return
    if subprocess.run([*base, str(patch)], cwd=FACEFUSION_DIR).returncode:
        if optional:
            print(f'提示：未能应用 {name}，分步测试的进度回传可能不可用。')
        else:
            raise SystemExit(f'未能应用 {name}。需要 FaceFusion {FACEFUSION_TAG}。')


def download_facefusion_models():
    sys.path.insert(0, str(FACEFUSION_DIR))
    from facefusion import choices, content_analyser, face_classifier, face_detector, face_landmarker, face_masker, face_recognizer, logger, state_manager
    from facefusion.processors.modules import face_enhancer, face_swapper

    settings = {
        'download_providers': choices.download_providers, 'log_level': 'info',
        'face_detector_model': 'yoloface', 'face_landmarker_model': '2dfan4',
        'face_swapper_model': 'inswapper_128_fp16', 'face_enhancer_model': 'gfpgan_1.4',
        'face_occluder_model': 'xseg_1', 'face_parser_model': 'bisenet_resnet_34',
    }
    for key, value in settings.items():
        state_manager.init_item(key, value)
    logger.init('info')
    for module in (face_detector, face_landmarker, face_recognizer, face_classifier, face_swapper,
                   face_enhancer, face_masker, content_analyser):
        if not module.pre_check():
            raise SystemExit(f'无法准备 {module.__name__} 权重，请检查网络后重试。')


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

    apply_patch('workbench-analyser.patch')
    shutil.copy2(REPO_ROOT / 'facefusion' / 'workbench_progress.py',
                 FACEFUSION_DIR / 'facefusion' / 'workbench_progress.py')
    apply_patch('workbench-progress.patch', optional=True)
    download_facefusion_models()

    print()
    print('安装完成。启动：')
    print('  powershell -ExecutionPolicy Bypass -File start.ps1')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

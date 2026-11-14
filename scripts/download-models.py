#!/usr/bin/env python3
"""下载本工作台专有的 VR180 粗检测模型。

FaceFusion 自身的模型由 bootstrap.py 准备。这里只负责 FaceFusion 没有的
`yoloface_12n-vr180.onnx`。

来源：https://github.com/Glat0s/yolo-face/releases/tag/vr
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / '.assets' / 'models'
MODEL_NAME = 'yoloface_12n-vr180.onnx'
MODEL_URL = 'https://github.com/Glat0s/yolo-face/releases/download/vr/' + MODEL_NAME
MODEL_SHA256 = '8d5e9187fe5aa8f95be7a3267802639299fb39df969407d22bc379eed9aaf1e1'

CHUNK = 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    target = MODEL_DIR / MODEL_NAME
    if target.is_file() and sha256(target) == MODEL_SHA256:
        print(f'已存在且校验通过：{target}')
        return 0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + '.part')
    print(f'下载 {MODEL_URL}')
    print(f'  到 {target}')
    try:
        with urllib.request.urlopen(MODEL_URL) as response, part.open('wb') as stream:
            downloaded = 0
            total = int(response.headers.get('Content-Length') or 0)
            while chunk := response.read(CHUNK):
                stream.write(chunk)
                downloaded += len(chunk)
                if total:
                    sys.stdout.write(f'\r  {downloaded / 1e6:6.1f} MB / {total / 1e6:.1f} MB')
                    sys.stdout.flush()
            sys.stdout.write('\n')
        if sha256(part) != MODEL_SHA256:
            part.unlink(missing_ok=True)
            print('校验失败：SHA-256 不匹配，已删除下载文件。', file=sys.stderr)
            return 1
        part.replace(target)
        print(f'完成，已写入 {target}')
        return 0
    except Exception as exc:
        part.unlink(missing_ok=True)
        print(f'下载失败：{exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

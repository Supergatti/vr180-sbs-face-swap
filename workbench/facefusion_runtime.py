from __future__ import annotations

import os
import sys
from pathlib import Path


HOME = Path(os.environ.get('FACEFUSION_HOME') or Path(__file__).resolve().parents[1] / '.facefusion').resolve()


def configure():
    if not (HOME / 'facefusion.py').is_file():
        raise RuntimeError('未找到 FaceFusion。请运行 setup.ps1，或设置 FACEFUSION_HOME。')
    if str(HOME) not in sys.path:
        sys.path.insert(0, str(HOME))

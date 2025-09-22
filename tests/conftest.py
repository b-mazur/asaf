from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))

try:
    import asaf.utils as asaf_utils
except ModuleNotFoundError:
    pass
else:
    sys.modules.setdefault("utils", asaf_utils)

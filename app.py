"""Hugging Face Spaces entry point — delegates to demo/app.py."""

import sys
from pathlib import Path

# Ensure demo/ is importable (matches demo/app.py's sys.path setup)
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))

from demo.app import demo  # noqa: E402

if __name__ == "__main__":
    demo.launch()

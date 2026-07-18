"""Make V3's gaira_core importable in tests regardless of pytest rootdir."""
import sys
from pathlib import Path

DEMO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DEMO_ROOT))
sys.path.insert(0, str(DEMO_ROOT / "tests"))

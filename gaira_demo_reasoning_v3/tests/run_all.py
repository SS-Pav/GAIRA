"""Lightweight, dependency-free test runner (venv has no pytest).

Discovers tests/test_*.py, runs every top-level `test_*` function, reports
PASS/FAIL. Tests remain pytest-compatible; this is just a runner.

Usage: python tests/run_all.py
Exit code: 0 if all pass, 1 otherwise.
"""
import importlib.util
import sys
import traceback
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
DEMO_ROOT = TESTS_DIR.parent
sys.path.insert(0, str(DEMO_ROOT))
sys.path.insert(0, str(TESTS_DIR))


def load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    files = sorted(p for p in TESTS_DIR.glob("test_*.py"))
    total = passed = failed = 0
    fails = []
    for f in files:
        try:
            mod = load(f)
        except Exception as e:
            print(f"  IMPORT-FAIL {f.name}: {e}")
            failed += 1
            fails.append(f"{f.name}::<import>")
            continue
        fns = [n for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n))]
        for n in fns:
            total += 1
            try:
                getattr(mod, n)()
                print(f"  PASS  {f.name}::{n}")
                passed += 1
            except Exception:
                print(f"  FAIL  {f.name}::{n}")
                traceback.print_exc()
                failed += 1
                fails.append(f"{f.name}::{n}")
    print("-" * 60)
    print(f"total={total} passed={passed} failed={failed}")
    if fails:
        print("FAILURES:", ", ".join(fails))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

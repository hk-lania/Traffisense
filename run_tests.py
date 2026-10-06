"""Run the whole test suite without needing pytest installed.

    python run_tests.py            # all tests
    python run_tests.py -v         # verbose, one line per test

Picks up both unittest-style classes (metrics tests) and plain
`def test_...()` functions (controller tests).
"""

import importlib
import inspect
import sys
import unittest
from pathlib import Path


def plain_function_tests(start_dir: str = "tests") -> unittest.TestSuite:
    """Wrap module-level `test_*` functions so unittest runs them too."""
    suite = unittest.TestSuite()
    for path in sorted(Path(start_dir).glob("test_*.py")):
        module = importlib.import_module(f"{start_dir}.{path.stem}")
        for name, obj in vars(module).items():
            if (name.startswith("test_") and inspect.isfunction(obj)
                    and obj.__module__ == module.__name__):
                suite.addTest(unittest.FunctionTestCase(obj, description=f"{path.stem}.{name}"))
    return suite


def main() -> int:
    verbosity = 2 if "-v" in sys.argv else 1
    suite = unittest.defaultTestLoader.discover(start_dir="tests", top_level_dir=".")
    suite.addTests(plain_function_tests())
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())

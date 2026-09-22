"""Run the whole test suite without needing pytest installed.

    python run_tests.py            # all tests
    python run_tests.py -v         # verbose, one line per test
"""

import sys
import unittest


def main() -> int:
    verbosity = 2 if "-v" in sys.argv else 1
    suite = unittest.defaultTestLoader.discover(start_dir="tests", top_level_dir=".")
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())

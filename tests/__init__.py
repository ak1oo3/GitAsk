"""Test package bootstrap.

The project ships no installable package metadata (no pyproject/setup), so make
``import gitask`` work when pytest is run from the repo root by putting the
``src/`` layout on ``sys.path``. This runs before any test module imports
``gitask`` because Python executes the package ``__init__`` first.
"""

import os
import sys

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

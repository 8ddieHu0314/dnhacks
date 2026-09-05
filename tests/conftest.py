"""Shared test setup. Ensures the repo root is importable so tests can
`import app` and `from core...` regardless of pytest's invocation directory.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

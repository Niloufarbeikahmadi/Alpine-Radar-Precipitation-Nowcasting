"""Make ``src/`` importable when the package is not pip-installed, and use a headless backend."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

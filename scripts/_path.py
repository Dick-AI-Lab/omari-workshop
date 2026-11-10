"""Make `omari_us` importable when running scripts from a fresh checkout without `pip install -e python`."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

"""
tests/conftest.py
-----------------
Shared pytest configuration and fixtures.
"""
import sys
from pathlib import Path

# Ensure project root is on the Python path for all tests
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

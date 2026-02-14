"""Root conftest.py - ensures the project root is on sys.path for all tests."""
import sys
import os

# Add project root to path so all imports resolve correctly
sys.path.insert(0, os.path.dirname(__file__))

"""Root conftest.py - ensures the project root is on sys.path for all tests."""
import sys
import os

# Add project root to path so all imports resolve correctly
sys.path.insert(0, os.path.dirname(__file__))

# Exclude non-test files and directories from collection
collect_ignore = [
    "test_results.txt",
    "run_comprehensive_hil_test.py",
]

collect_ignore_glob = [
    "phase5/ai_models/tests/*",
    "phase5/ai_models/l4_autonomous/tests/*",
]

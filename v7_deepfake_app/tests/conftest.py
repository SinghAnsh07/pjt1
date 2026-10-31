"""
Pytest configuration and environment fixtures for V7 Deepfake App tests.
"""

import os
import sys

# Ensure v7_deepfake_app root is in sys.path
TEST_DIR = os.path.dirname(__file__)
APP_ROOT = os.path.abspath(os.path.join(TEST_DIR, ".."))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

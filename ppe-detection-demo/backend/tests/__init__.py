# Test package
import os
import sys

# Ensure backend root is on sys.path so tests can import detector, app, etc.
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

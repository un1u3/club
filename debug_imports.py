import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

print(f"PYTHONPATH: {sys.path[:3]}")
print(f"CWD: {os.getcwd()}")

modules = ["study_agents", "core", "knowmyschool"]
for m in modules:
    try:
        __import__(m)
        print(f"SUCCESS: imported {m}")
    except ImportError as e:
        print(f"FAILURE: {m} - {e}")

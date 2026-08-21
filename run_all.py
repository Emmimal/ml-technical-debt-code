"""
Runs every script in this repo, in the order the article presents them, and
prints each one's output under a clear header. Run this from inside the
ml-technical-debt-code/ folder (the same folder this file lives in).

Usage: python run_all.py
"""
import subprocess
import sys

SCRIPTS = [
    "01_entanglement.py",
    "02_feedback_loop.py",
    "03_audit_tool.py",
    "04_scoring_framework.py",
    "05_characterization_test.py",
    "06_characterization_test_catches_bug.py",
]

for script in SCRIPTS:
    print("\n" + "=" * 70)
    print(f"RUNNING: {script}")
    print("=" * 70)
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"--- {script} exited with an error ---")
        print(result.stderr)

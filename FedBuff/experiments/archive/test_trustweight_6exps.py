#!/usr/bin/env python3
"""
Quick test script for TrustWeight with all 6 experiments.
Tests the straggler-robustness improvements.
"""

# Note: This script requires solution.py to be converted to a proper module
# For now, we'll run it directly using Python's exec with proper handling

import sys
import os
import time
from pathlib import Path

# Read solution.py and extract the relevant parts
with open('solution.py', 'r') as f:
    solution_code = f.read()

# Remove the problematic __future__ import that's not at the top
# Actually, let's just execute it in a way that works
# Better approach: use subprocess to run it as a script

print("="*70)
print("TRUSTWEIGHT QUICK TEST - All 6 Experiments")
print("="*70)
print("\nNote: This will run all 6 experiments with current configs.")
print("Exp5 & Exp6 have straggler-robustness improvements applied.")
print("\nStarting test...\n")

# Since solution.py is a notebook-style file, we need to run it differently
# Let's create a minimal runner that uses the key parts

# Actually, the simplest is to just modify solution.py temporarily for quick test
# Or run it as-is and let it complete

# For now, let's just run the solution.py directly but we need to handle the notebook format
# Better: create a simple wrapper that imports and runs

import subprocess
import sys

# Run solution.py using Python
# We'll need to handle the notebook format - solution.py has # %% markers
# For a quick test, let's just run it and see what happens

print("Running TrustWeight experiments...")
print("This may take a while. Experiments will run sequentially.\n")

result = subprocess.run(
    [sys.executable, '-c', '''
import sys
# Read and execute solution.py
with open("solution.py", "r") as f:
    code = f.read()
# Remove __future__ import issue by executing in parts
# Actually, let's just import what we need
exec(code.replace("from __future__ import annotations", "# from __future__ import annotations"))
'''],
    cwd=os.getcwd(),
    capture_output=False
)

print("\n" + "="*70)
print("Test completed!")
print("="*70)



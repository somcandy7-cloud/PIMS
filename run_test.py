#!/usr/bin/env python
"""Quick test runner to verify implementation."""
import sys
import subprocess

result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/test_llm_backends.py", "-v"],
    cwd="c:/Users/somca/내문서/project/PIMS",
    capture_output=True,
    text=True,
)

print("STDOUT:")
print(result.stdout)
print("\nSTDERR:")
print(result.stderr)
print(f"\nReturn code: {result.returncode}")

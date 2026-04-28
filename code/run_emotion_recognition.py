#!/usr/bin/env python3
"""Top-level launcher for the emotion recognition evaluation.

Usage:
  python run_emotion_recognition.py --input-dir ... --backbone-checkpoint ... --lstm-checkpoint ... --output-json ...

This script ensures the `code/src` folder is on sys.path and calls the package's main() entry.
"""
import sys
from pathlib import Path

# Ensure code/src is on path so we can import the package
HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from emotion_recognition.emotion_recognition_script import main
except Exception as exc:
    print("Failed to import emotion_recognition package:", exc)
    raise

if __name__ == "__main__":
    raise SystemExit(main())

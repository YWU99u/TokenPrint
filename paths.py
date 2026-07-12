"""Shared path configuration for J-Space release.

When running from the release directory, paths resolve automatically.
Override with environment variables if needed:
  JSPACE_ROOT, JSPACE_FP_DIR, JSPACE_OUT_DIR, JSPACE_RESULTS
"""
import os

ROOT = os.environ.get("JSPACE_ROOT", os.path.dirname(os.path.abspath(__file__)))
FP_DIR = os.environ.get("JSPACE_FP_DIR",
    os.path.join(ROOT, "fingerprints", "calibration"))
OUT_DIR = os.environ.get("JSPACE_OUT_DIR",
    os.path.join(ROOT, "fingerprints", "output"))
CKPT_DIR = os.path.join(ROOT, "fingerprints", "checkpoints")
CTRL_DIR = os.path.join(ROOT, "fingerprints", "controls")
WITNESS_DIR = os.path.join(ROOT, "fingerprints", "witnesses")
RESULTS_DIR = os.environ.get("JSPACE_RESULTS",
    os.path.join(ROOT, "results"))

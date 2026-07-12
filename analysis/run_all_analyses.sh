#!/usr/bin/env bash
# Reproduce all analysis results from pre-extracted fingerprints.
# Run from the release/ root directory:
#   cd release && bash analysis/run_all_analyses.sh
#
# No GPU needed. Requires: numpy, matplotlib, scikit-learn.
# Total runtime: ~5 minutes on a laptop.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== J-Space: reproducing all analysis results ==="
echo "Working directory: $(pwd)"
echo ""

# Set up paths for analysis scripts
export JSPACE_ROOT="$(pwd)"
export JSPACE_FP_DIR="$(pwd)/fingerprints/calibration"
export JSPACE_OUT_DIR="$(pwd)/fingerprints/output"
export JSPACE_RESULTS="$(pwd)/results"

# Step 1: Core calibration (ladder, matrix, family stats)
echo ">>> [1/10] Calibration ladder + family structure"
python analysis/final_analysis.py

# Step 2: Robustness suite (leave-one-out, permutation, jackknife)
echo ">>> [2/10] Robustness suite"
python analysis/robustness_suite.py

# Step 3: Reviewer requirements (tokenizer stratification, domain leave-out, margins)
echo ">>> [3/10] Reviewer analyses"
python analysis/reviewer_musts.py

# Step 4: PhyloLM-style baseline
echo ">>> [4/10] PhyloLM baseline"
python analysis/phylolm_baseline.py

# Step 5: Tokenizer control (Qwen subpool)
echo ">>> [5/10] Tokenizer control"
python analysis/tokenizer_control.py

# Step 6: Depth ablation (30-model pool)
echo ">>> [6/10] Depth ablation (30 models)"
python analysis/depth_ablation_30.py

# Step 7: Witness pool analysis
echo ">>> [7/10] Witness pool (Pile ecosystem)"
python analysis/e0a_witness.py

# Step 8: Training trajectory
echo ">>> [8/10] Training trajectory (Pythia checkpoints)"
python analysis/e0b_analyze.py

# Step 9: Open-set ancestry
echo ">>> [9/10] Open-set ancestry identification"
python analysis/openset_ancestry.py

# Step 10: Metadata baseline + confound regression
echo ">>> [10/10] Metadata baseline + confound regression"
python analysis/metadata_confound.py

echo ""
echo "=== All analyses complete. Results in results/ ==="
ls -la results/*.json

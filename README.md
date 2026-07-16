# J-Space: A Calibrated Token-Space Fingerprint for Language-Model Provenance

Code, data, and fingerprints for the AAAI-27 submission.

## What's in this release

```
release/
├── probes.py                   # 250 probe prompts (frozen, never modified after authoring)
├── capability_gold.py          # 121 gold-key probes for capability scoring
├── jlens.py                    # Logit-lens extraction (GPT-NeoX, GPT-2, LlamaForCausalLM, etc.)
├── fingerprint_v2.py           # Fingerprint extraction + pairwise similarity + family AUC
├── output_baseline.py          # Output-level (true logits) fingerprint extraction + analysis
│
├── analysis/                   # All analysis scripts (run on pre-extracted fingerprints)
│   ├── final_analysis.py       # Calibration ladder, family stats, matrix
│   ├── robustness_suite.py     # Leave-one-out, block removal, jackknife, permutation test
│   ├── reviewer_musts.py       # Tokenizer stratification, domain leave-out, margin bootstrap
│   ├── phylolm_baseline.py     # PhyloLM-style Bhattacharyya baseline (26-model subpool)
│   ├── tokenizer_control.py    # Qwen shared-vocab subpool AUC
│   ├── depth_ablation_30.py    # Depth ablation on full 30-model pool
│   ├── e0a_witness.py          # Witness pool (Pile ecosystem) analysis
│   ├── e0b_analyze.py          # Training trajectory analysis (Pythia checkpoints)
│   ├── openset_ancestry.py     # Open-set ancestry identification
│   ├── metadata_confound.py    # B3 metadata baseline + B4 confound regression
│   └── run_all_analyses.sh     # Reproduce all analysis results in order
│
├── extraction/                 # Scripts for fingerprint extraction (requires GPU)
│   ├── extract_calibration.sh  # Extract 30 calibration-pool models
│   ├── extract_witnesses.sh    # Extract witness-pool models
│   ├── extract_checkpoints.sh  # Extract Pythia training checkpoints
│   └── cloud_extract_v2.py     # Standalone extractor for rented GPUs
│
├── figures/
│   └── make_figures_jspace.py  # Generate all paper figures from results/
│
├── fingerprints/               # Pre-extracted fingerprints (all models)
│   ├── calibration/            # 30 calibration-pool models (_fp.json)
│   ├── output/                 # Output-level fingerprints (_out.json)
│   ├── witnesses/              # Witness-pool models
│   ├── controls/               # Base/deduped/quantized controls
│   └── checkpoints/            # Pythia training checkpoints
│
└── results/                    # All computed results (JSON)
    ├── final_analysis.json
    ├── robustness_suite.json
    ├── reviewer_musts.json
    ├── ...
    └── (17 result files total)
```

## Reproducing the results

### From pre-extracted fingerprints (no GPU needed)

All fingerprints are included. To reproduce every number in the paper:

```bash
cd release
bash analysis/run_all_analyses.sh
```

This runs all analysis scripts in dependency order and takes ~5 minutes on a laptop.
Output goes to `results/`; diff against the shipped `results/` to verify.

### From scratch (requires GPU)

To re-extract fingerprints from model weights:

```bash
pip install torch transformers accelerate safetensors sentencepiece

# 1. Download models to models_dl/ (not included due to size)
# 2. Set MODELS_DIR and DEVICE in extraction scripts
# 3. Run:
bash extraction/extract_calibration.sh   # ~2 GPU-hours, single A100
bash extraction/extract_witnesses.sh     # ~1 GPU-hour
bash extraction/extract_checkpoints.sh   # ~2 GPU-hours (downloads ~110GB)

# Then run analyses:
bash analysis/run_all_analyses.sh
```

Numerical precision: fp32 and bf16 extractions agree at Jaccard 0.98.

## Key files for specific claims

| Paper claim | Script | Input | Output |
|---|---|---|---|
| Calibration ladder (Table 1) | `final_analysis.py` | `fingerprints/calibration/` | `results/final_analysis.json` |
| Family AUC 0.872 / robustness | `robustness_suite.py` | same | `results/robustness_suite.json` |
| Witness pool (Table 5) | `e0a_witness.py` | `fingerprints/witnesses/` + `output/` | `results/e0a_witness_output.json` |
| Training trajectory (Fig 5) | `e0b_analyze.py` | `fingerprints/checkpoints/` | `results/e0b_trajectory.json` |
| Ancestry identification (Table 3) | `openset_ancestry.py` | `fingerprints/calibration/` | `results/openset_ancestry.json` |
| Depth ablation (Fig 3) | `depth_ablation_30.py` | calibration + output | `results/depth_ablation_30.json` |
| PhyloLM baseline | `phylolm_baseline.py` | `fingerprints/output/` | `results/phylolm_baseline.json` |
| Metadata / confound (B3/B4) | `metadata_confound.py` | `fingerprints/calibration/` | `results/metadata_confound.json` |
| Quantization envelope | `depth_ablation_30.py` | `fingerprints/controls/` | `results/quant_envelope.json` |
| API truncation (k=5) | `depth_ablation_30.py` | `fingerprints/output/` | `results/api_truncation.json` |
| Capability scores | `capability_gold.py` | `fingerprints/calibration/` | (stdout) |
| All figures | `make_figures_jspace.py` | `results/` | `paper/figures/` |

## Requirements

```
numpy
matplotlib
scikit-learn  # metadata_confound.py only
```

No GPU needed for analysis. Extraction requires `torch`, `transformers`, `accelerate`.

## License

Code: MIT. Fingerprint data: CC-BY-4.0. Probe suite: CC-BY-4.0.

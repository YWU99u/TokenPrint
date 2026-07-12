#!/usr/bin/env python3
"""
E0-b analysis: same-data signal along the Pythia training trajectory.

For each available step t:
  pair      excess-S( 1.4b@t , 6.9b@t )     <- the curve that matters
  ref-1.4b  excess-S( 1.4b@t , OPT-6.7B )   <- unrelated reference curve
  ref-6.9b  excess-S( 6.9b@t , OPT-6.7B )
  self-*    raw-S( size@t , size@final )    <- drift toward final checkpoint

All excess = matched-probe mean Jaccard@20 minus mismatched-probe null
(200 random i!=j probe pairs), near-final depth — same procedure as the
paper. Raw values are reported alongside; the trajectory claim must be
read on EXCESS (early checkpoints emit generic high-frequency tokens that
inflate raw overlap for trivial reasons).

Uses whatever checkpoints e0b_pipeline.py has finished. step143000 = the
existing main fingerprints in results/fingerprints_v2/.

Output: results/e0b_trajectory.json
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from paths import ROOT, FP_DIR, OUT_DIR, CKPT_DIR, CTRL_DIR, WITNESS_DIR, RESULTS_DIR
sys.path.insert(0, ROOT)

import json
import numpy as np
K = 20
N_NULL = 200
rng = np.random.default_rng(42)

STEPS = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 143000]
# Two references with documented data relationships (paper, Robustness):
#   opt-6.7b    : partial overlap (OPT corpus includes a Pile subset, 0.34-0.37 raw)
#   Qwen2.5-7B  : no documented overlap (0.14-0.18 raw) — the clean unrelated ref
REFS = {"opt": "opt-6.7b", "qwen": "Qwen2.5-7B"}


def nf_key(fp):
    ls_, nn = fp["target_layers"], fp["num_layers"]
    return str(min(ls_, key=lambda l: abs(l - (nn - 2))))


def jac(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def load_sets(path):
    fp = json.load(open(path))
    lk = nf_key(fp)
    return {p: frozenset(t for t, s in d[lk][:K]) for p, d in fp["probes"].items()
            if not (isinstance(d, dict) and "error" in d) and lk in d}


def sim(sa, sb):
    """(raw, null, excess) over common probes."""
    common = sorted(set(sa) & set(sb))
    raw = float(np.mean([jac(sa[p], sb[p]) for p in common]))
    n = len(common)
    nulls = []
    for _ in range(N_NULL):
        i, j = rng.integers(0, n), rng.integers(0, n)
        while j == i:
            j = rng.integers(0, n)
        nulls.append(jac(sa[common[i]], sb[common[j]]))
    null = float(np.mean(nulls))
    return raw, null, raw - null


def _find_fp(name):
    """Search FP_DIR, then CTRL_DIR, then WITNESS_DIR for a fingerprint file."""
    for d in (FP_DIR, CTRL_DIR, WITNESS_DIR):
        p = os.path.join(d, f"{name}_fp.json")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{name}_fp.json not found in calibration/controls/witnesses")


def fp_path(size, step):
    if step == 143000:
        return _find_fp(size)
    return os.path.join(CKPT_DIR, f"{size}@step{step}_fp.json")


def main():
    ref_sets = {tag: load_sets(_find_fp(name))
                for tag, name in REFS.items()}
    final = {s: load_sets(fp_path(s, 143000)) for s in ("pythia-1.4b", "pythia-6.9b")}

    out = {"steps": {}}
    print(f"{'step':>7s} {'pair_raw':>9s} {'pair_null':>9s} {'pair_EXC':>9s} "
          f"{'opt14':>7s} {'opt69':>7s} {'qwen14':>7s} {'qwen69':>7s} "
          f"{'self14':>7s} {'self69':>7s}   (ref cols = EXCESS)")
    for step in STEPS:
        p14, p69 = fp_path("pythia-1.4b", step), fp_path("pythia-6.9b", step)
        if not (os.path.exists(p14) and os.path.exists(p69)):
            print(f"{step:>7d}   [checkpoints not extracted yet]")
            continue
        s14, s69 = load_sets(p14), load_sets(p69)
        raw, null, exc = sim(s14, s69)
        refs = {}
        for tag in REFS:
            _, _, refs[f"{tag}14"] = sim(s14, ref_sets[tag])
            _, _, refs[f"{tag}69"] = sim(s69, ref_sets[tag])
        self14 = float(np.mean([jac(s14[p], final["pythia-1.4b"][p])
                                for p in set(s14) & set(final["pythia-1.4b"])]))
        self69 = float(np.mean([jac(s69[p], final["pythia-6.9b"][p])
                                for p in set(s69) & set(final["pythia-6.9b"])]))
        out["steps"][step] = {
            "pair_raw": round(raw, 4), "pair_null": round(null, 4),
            "pair_excess": round(exc, 4),
            **{f"ref_{k}_excess": round(v, 4) for k, v in refs.items()},
            "self_1.4b_vs_final": round(self14, 4),
            "self_6.9b_vs_final": round(self69, 4)}
        print(f"{step:>7d} {raw:9.3f} {null:9.3f} {exc:9.3f} "
              f"{refs['opt14']:7.3f} {refs['opt69']:7.3f} "
              f"{refs['qwen14']:7.3f} {refs['qwen69']:7.3f} "
              f"{self14:7.3f} {self69:7.3f}")

    with open(os.path.join(RESULTS_DIR, "e0b_trajectory.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\nSaved -> results/e0b_trajectory.json")
    print("Read the claim on pair_EXCESS relative to ref curves, not raw.")


if __name__ == "__main__":
    main()

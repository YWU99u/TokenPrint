#!/usr/bin/env python3
"""
Tokenizer confound control for J-Space fingerprinting.

Two controls:
1. Mismatched-probe null: Jaccard between model A's top-k on probe i and
   model B's top-k on probe j (i != j). Captures baseline agreement from
   shared tokenizer + generic frequent tokens, independent of probe content.
   Excess similarity = matched - mismatched.
2. Vocab-overlap partial correlation: regress pair similarity on
   family-match controlling for tokenizer vocabulary overlap.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from paths import ROOT, FP_DIR, OUT_DIR, CKPT_DIR, CTRL_DIR, WITNESS_DIR, RESULTS_DIR
sys.path.insert(0, ROOT)

import json
import numpy as np
from collections import defaultdict

from fingerprint_v2 import FAMILY_MAP
K = 20
N_NULL = 200  # mismatched pairs sampled per model pair
rng = np.random.default_rng(42)


def get_layer_key(fp):
    layers, n = fp["target_layers"], fp["num_layers"]
    return str(min(layers, key=lambda l: abs(l - 3 * n // 4)))


def load_topk_sets(fp):
    """prompt -> frozenset of top-K token strings at 3/4 depth."""
    lk = get_layer_key(fp)
    out = {}
    for prompt, data in fp["probes"].items():
        if isinstance(data, dict) and "error" in data:
            continue
        if lk not in data:
            continue
        out[prompt] = frozenset(t for t, s in data[lk][:K])
    return out


def jac(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def main():
    fps = {}
    for f in sorted(os.listdir(FP_DIR)):
        if f.endswith("_fp.json") and not f.startswith("cloud-test"):
            name = f.replace("_fp.json", "")
            with open(os.path.join(FP_DIR, f)) as fh:
                fps[name] = json.load(fh)
    models = sorted(fps.keys())
    print(f"Loaded {len(models)} fingerprints")

    sets = {m: load_topk_sets(fps[m]) for m in models}

    # Common prompts across all models
    common = sorted(set.intersection(*[set(s.keys()) for s in sets.values()]))
    P = len(common)
    print(f"Common prompts: {P}")

    # ── Control 1: mismatched-probe null ──────────────────────────
    results = {}
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            ma, mb = models[i], models[j]
            matched = np.mean([jac(sets[ma][p], sets[mb][p]) for p in common])
            # Mismatched: random probe pairs (p != q)
            null_vals = []
            for _ in range(N_NULL):
                pi, qi = rng.choice(P, 2, replace=False)
                null_vals.append(jac(sets[ma][common[pi]], sets[mb][common[qi]]))
            mismatched = np.mean(null_vals)
            results[(ma, mb)] = {
                "matched": matched,
                "mismatched": mismatched,
                "excess": matched - mismatched,
            }

    # Family stats on raw vs excess
    def family_stats(key):
        intra_by_fam, all_inter = defaultdict(list), []
        for (ma, mb), r in results.items():
            fa, fb = FAMILY_MAP.get(ma, ma), FAMILY_MAP.get(mb, mb)
            if fa == fb:
                intra_by_fam[fa].append(r[key])
            else:
                all_inter.append(r[key])
        return intra_by_fam, all_inter

    print(f"\n{'':16s} {'--- raw Jaccard@20 ---':^28s} {'--- excess (confound-corrected) ---':^36s}")
    print(f"{'Family':16s} {'intra':>8s} {'inter':>8s} {'ratio':>8s}   {'intra':>8s} {'inter':>8s} {'ratio':>8s}")
    raw_intra, raw_inter = family_stats("matched")
    exc_intra, exc_inter = family_stats("excess")
    ri_mean, ei_mean = np.mean(raw_inter), np.mean(exc_inter)
    for fam in sorted(raw_intra):
        if len(raw_intra[fam]) < 1:
            continue
        rm, em = np.mean(raw_intra[fam]), np.mean(exc_intra[fam])
        print(f"{fam:16s} {rm:8.3f} {ri_mean:8.3f} {rm/ri_mean:7.2f}x   {em:8.3f} {ei_mean:8.3f} {em/ei_mean:7.2f}x")

    # Baseline magnitude: how much of raw similarity is tokenizer/frequency artifact?
    mism_same_tok, mism_diff_tok = [], []
    # Rough same-tokenizer groups: Qwen* + DS-Qwen* share Qwen tokenizer; Llama3* + DS-Llama share Llama3; Pythia share GPTNeoX
    def tok_group(m):
        if "Qwen" in m and "Llama" not in m: return "qwen"
        if "Llama" in m: return "llama3"  # includes DS-Distill-Llama
        if "pythia" in m: return "neox"
        if "gemma" in m: return "gemma"
        if "Mistral" in m or "Ministral" in m: return "mistral"
        if "Phi" in m or "phi" in m: return "phi"
        if "internlm" in m: return "internlm"
        return m
    for (ma, mb), r in results.items():
        (mism_same_tok if tok_group(ma) == tok_group(mb) else mism_diff_tok).append(r["mismatched"])
    print(f"\nMismatched-probe baseline (pure tokenizer/frequency artifact):")
    print(f"  same-tokenizer pairs:      {np.mean(mism_same_tok):.4f}")
    print(f"  different-tokenizer pairs: {np.mean(mism_diff_tok):.4f}")

    # ── Permutation test on EXCESS similarity ─────────────────────
    fams = [FAMILY_MAP.get(m, m) for m in models]
    def intra_inter_diff(labels):
        intra, inter = [], []
        for idx_a in range(len(models)):
            for idx_b in range(idx_a + 1, len(models)):
                v = results[(models[idx_a], models[idx_b])]["excess"]
                (intra if labels[idx_a] == labels[idx_b] else inter).append(v)
        return np.mean(intra) - np.mean(inter) if intra and inter else 0.0

    observed = intra_inter_diff(fams)
    null = [intra_inter_diff(list(rng.permutation(fams))) for _ in range(1000)]
    p = np.mean([d >= observed for d in null])
    eff = observed / (np.std(null) + 1e-10)
    print(f"\nPermutation test on EXCESS similarity: p={p:.4f}, effect={eff:.2f}σ")

    # ── Control 2: partial correlation with vocab overlap ─────────
    # Vocab overlap proxy: Jaccard of the union-of-top100-token-strings each model ever produced
    all_tokens = {}
    for m in models:
        toks = set()
        lk = get_layer_key(fps[m])
        for prompt, data in fps[m]["probes"].items():
            if isinstance(data, dict) and "error" in data: continue
            if lk not in data: continue
            toks.update(t for t, s in data[lk])
        all_tokens[m] = toks

    y, fam_match, vocab_ov = [], [], []
    for (ma, mb), r in results.items():
        y.append(r["matched"])
        fam_match.append(1.0 if FAMILY_MAP.get(ma, ma) == FAMILY_MAP.get(mb, mb) else 0.0)
        vocab_ov.append(jac(all_tokens[ma], all_tokens[mb]))
    y, fam_match, vocab_ov = map(np.array, (y, fam_match, vocab_ov))

    def partial_corr(x, ycol, z):
        rx = x - np.polyval(np.polyfit(z, x, 1), z)
        ry = ycol - np.polyval(np.polyfit(z, ycol, 1), z)
        return np.corrcoef(rx, ry)[0, 1]

    r_raw = np.corrcoef(fam_match, y)[0, 1]
    r_partial = partial_corr(fam_match, y, vocab_ov)
    print(f"\nCorr(family_match, similarity):                    r = {r_raw:.3f}")
    print(f"Partial corr controlling token-space overlap:      r = {r_partial:.3f}")

    out = {
        "n_models": len(models),
        "pairs": {f"{a}|{b}": v for (a, b), v in results.items()},
        "permutation_excess": {"p": float(p), "effect": float(eff), "observed": float(observed)},
        "corr_raw": float(r_raw), "corr_partial": float(r_partial),
        "mismatched_baseline_same_tok": float(np.mean(mism_same_tok)),
        "mismatched_baseline_diff_tok": float(np.mean(mism_diff_tok)),
    }
    with open(os.path.join(RESULTS_DIR, "tokenizer_control.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nSaved → results/tokenizer_control.json")


if __name__ == "__main__":
    main()

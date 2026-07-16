#!/usr/bin/env python3
"""
FINAL frozen analysis for AAAI paper (Plan C: dual-depth reporting).

Depth 3/4  = "deep concept fingerprint"  — primary for models whose concepts
             have crystallized by 3/4 depth (all <=14B here).
Near-final = universal fingerprint       — all 32 models.
Plus: crystallization-depth finding (Qwen3-32B case study).

Outputs results/final_analysis.json with everything the paper needs.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from paths import ROOT, FP_DIR, OUT_DIR, CKPT_DIR, CTRL_DIR, WITNESS_DIR, RESULTS_DIR
sys.path.insert(0, ROOT)

import json
import numpy as np
from collections import defaultdict

from fingerprint_v2 import compute_pairwise, FAMILY_MAP, roc_auc

OUT = os.path.join(RESULTS_DIR, "final_analysis.json")
rng = np.random.default_rng(42)

# 64-layer models where 3/4 depth precedes concept crystallization
DEEP_UNCRYSTALLIZED = {"Qwen3-32B", "DeepSeek-R1-Distill-Qwen-32B", "Qwen2.5-32B-Instruct", "Qwen2.5-32B", "gemma-2-27b-it"}


def load_all():
    fps = {}
    for f in sorted(os.listdir(FP_DIR)):
        if f.endswith("_fp.json") and not f.startswith("cloud-test"):
            fps[f.replace("_fp.json", "")] = json.load(open(os.path.join(FP_DIR, f)))
    return fps


def build_matrix(fps, models, frac):
    matrix = {}
    for i in range(len(models)):
        for j in range(i, len(models)):
            sim = compute_pairwise(fps[models[i]], fps[models[j]], frac)
            matrix[f"{models[i]}|{models[j]}"] = sim
            if i != j:
                matrix[f"{models[j]}|{models[i]}"] = sim
    return matrix


def family_stats(matrix, models):
    fams = defaultdict(list)
    for m in models:
        fams[FAMILY_MAP.get(m, m)].append(m)
    stats = {}
    for fam, members in sorted(fams.items()):
        if len(members) < 2:
            continue
        intra = [matrix[f"{a}|{b}"]["jaccard_20"] for i, a in enumerate(members) for b in members[i + 1:]]
        others = [m for m in models if m not in members]
        inter = [matrix[f"{a}|{b}"]["jaccard_20"] for a in members for b in others]
        stats[fam] = {
            "n": len(members),
            "intra": float(np.mean(intra)),
            "inter": float(np.mean(inter)),
            "ratio": float(np.mean(intra) / np.mean(inter)),
        }
    return stats


def permutation_test(matrix, models, n_perm=10000):
    fams = [FAMILY_MAP.get(m, m) for m in models]
    def diff(labels):
        intra, inter = [], []
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                v = matrix[f"{models[i]}|{models[j]}"]["jaccard_20"]
                (intra if labels[i] == labels[j] else inter).append(v)
        return np.mean(intra) - np.mean(inter)
    obs = diff(fams)
    null = [diff(list(rng.permutation(fams))) for _ in range(n_perm)]
    return {
        "observed": float(obs),
        "p": float(np.mean([d >= obs for d in null])),
        "p_str": f"< {1/n_perm}" if not any(d >= obs for d in null) else None,
        "effect_sigma": float(obs / (np.std(null) + 1e-12)),
        "n_perm": n_perm,
    }


def cohens_d(matrix, models):
    intra, inter = [], []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            v = matrix[f"{models[i]}|{models[j]}"]["jaccard_20"]
            same = FAMILY_MAP.get(models[i], models[i]) == FAMILY_MAP.get(models[j], models[j])
            (intra if same else inter).append(v)
    pooled = np.sqrt((np.var(intra, ddof=1) * (len(intra) - 1) + np.var(inter, ddof=1) * (len(inter) - 1))
                     / (len(intra) + len(inter) - 2))
    return float((np.mean(intra) - np.mean(inter)) / pooled)


def key_pairs(matrix):
    P = lambda a, b: matrix[f"{a}|{b}"]["jaccard_20"]
    return {
        "DS32B_base_Qwen2.5-32B": P("DeepSeek-R1-Distill-Qwen-32B", "Qwen2.5-32B"),
        "DS7B_base_Math-7B": P("DeepSeek-R1-Distill-Qwen-7B", "Qwen2.5-Math-7B"),
        "DSLlama8B_base_Llama31-8B": P("DeepSeek-R1-Distill-Llama-8B", "Llama-3.1-8B"),
        "Pythia_6.9_12": P("pythia-6.9b", "pythia-12b"),
        "Pythia_1.4_12": P("pythia-1.4b", "pythia-12b"),
        "Qwen3_32B_14B": P("Qwen3-32B", "Qwen3-14B"),
    }


def main():
    fps = load_all()
    models = sorted(fps.keys())
    print(f"{len(models)} models frozen for final analysis")

    out = {"models": models, "families": {m: FAMILY_MAP.get(m, m) for m in models}}

    # ── Analysis 1: deep concept fingerprint (3/4), crystallized models only ──
    crystallized = [m for m in models if m not in DEEP_UNCRYSTALLIZED]
    m34 = build_matrix(fps, crystallized, "3/4")
    out["depth_34"] = {
        "models": crystallized,
        "n": len(crystallized),
        "matrix_j20": {k: v["jaccard_20"] for k, v in m34.items()},
        "family_stats": family_stats(m34, crystallized),
        "permutation": permutation_test(m34, crystallized),
        "auc": roc_auc(fps, crystallized, m34),
        "cohens_d": cohens_d(m34, crystallized),
    }
    print(f"\n[3/4 depth, {len(crystallized)} crystallized models]")
    print(f"  perm p={out['depth_34']['permutation']['p']:.5f}  "
          f"effect={out['depth_34']['permutation']['effect_sigma']:.2f}σ  "
          f"AUC={out['depth_34']['auc']['auc']:.3f}  d={out['depth_34']['cohens_d']:.2f}")

    # ── Analysis 2: near-final fingerprint, ALL models ──
    mf = build_matrix(fps, models, "final")
    out["depth_final"] = {
        "models": models,
        "n": len(models),
        "matrix_j20": {k: v["jaccard_20"] for k, v in mf.items()},
        "family_stats": family_stats(mf, models),
        "permutation": permutation_test(mf, models),
        "auc": roc_auc(fps, models, mf),
        "cohens_d": cohens_d(mf, models),
        "key_pairs": key_pairs(mf),
    }
    print(f"\n[near-final depth, all {len(models)} models]")
    print(f"  perm p={out['depth_final']['permutation']['p']:.5f}  "
          f"effect={out['depth_final']['permutation']['effect_sigma']:.2f}σ  "
          f"AUC={out['depth_final']['auc']['auc']:.3f}  d={out['depth_final']['cohens_d']:.2f}")
    print("  key pairs:", {k: round(v, 3) for k, v in out["depth_final"]["key_pairs"].items()})

    # ── Analysis 3: crystallization depth per model ──
    # First extracted depth whose top-20 overlaps near-final top-20 by >= 0.3 Jaccard (self-comparison)
    cryst = {}
    for m in models:
        fp = fps[m]
        layers = fp["target_layers"]
        lk_final = str(layers[-1])
        fracs = []
        for lk in layers[:-1]:
            ovs = []
            for p, data in fp["probes"].items():
                if isinstance(data, dict) and "error" in data: continue
                if str(lk) not in data or lk_final not in data: continue
                a = set(t for t, s in data[str(lk)][:20])
                b = set(t for t, s in data[lk_final][:20])
                if a and b:
                    ovs.append(len(a & b) / len(a | b))
            fracs.append((lk / fp["num_layers"], float(np.mean(ovs))))
        cryst[m] = {"num_layers": fp["num_layers"], "self_overlap_by_depth": fracs}
    out["crystallization"] = cryst
    print("\n[crystallization: self-overlap(depth vs final) at 3/4]")
    for m in ["Qwen3-8B", "Qwen3-14B", "Qwen3-32B", "gemma-2-9b-it", "gemma-2-27b-it"]:
        v34 = cryst[m]["self_overlap_by_depth"][-1][1]
        print(f"  {m:22s} ({cryst[m]['num_layers']}L): {v34:.3f}")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nFrozen → {OUT}")


if __name__ == "__main__":
    main()

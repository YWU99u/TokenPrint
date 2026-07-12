#!/usr/bin/env python3
"""
PhyloLM-style same-pool baseline (reviewer W1).

PhyloLM (Yax et al., ICLR 2025) measures model distance from output-token
probabilities on fixed contexts (Nei-style similarity: sum over tokens of
sqrt(P_a * P_b), averaged over contexts). We reproduce that statistic from
our stored output distributions (top-100 logits -> softmax, matched across
models by decoded string), on the identical 26-model subpool and identical
probes, and evaluate it with the identical protocol (family AUC, model-level
permutation, base identification).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from paths import ROOT, FP_DIR, OUT_DIR, CKPT_DIR, CTRL_DIR, WITNESS_DIR, RESULTS_DIR
sys.path.insert(0, ROOT)

import json
import numpy as np
from collections import defaultdict

from fingerprint_v2 import FAMILY_MAP
rng = np.random.default_rng(42)


def load_dists():
    dists = {}
    for f in sorted(os.listdir(OUT_DIR)):
        if not f.endswith("_out.json"):
            continue
        m = f.replace("_out.json", "")
        raw = json.load(open(os.path.join(OUT_DIR, f)))
        per_probe = {}
        for p, toks in raw["probes"].items():
            if isinstance(toks, dict):
                continue
            # softmax over stored top-100 logits; merge duplicate strings
            logits = np.array([s for t, s in toks], dtype=np.float64)
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            d = defaultdict(float)
            for (t, _), pr in zip(toks, probs):
                d[t] += pr
            per_probe[p] = dict(d)
        dists[m] = per_probe
    return dists


def nei_sim(da, db, common):
    vals = []
    for p in common:
        A, B = da[p], db[p]
        shared = set(A) & set(B)
        vals.append(sum(np.sqrt(A[t] * B[t]) for t in shared))
    return float(np.mean(vals))


def auc_from(matrix, models):
    labels, scores = [], []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            same = FAMILY_MAP.get(models[i], models[i]) == FAMILY_MAP.get(models[j], models[j])
            labels.append(1 if same else 0)
            scores.append(matrix[(models[i], models[j])])
    labels, scores = np.array(labels), np.array(scores)
    order = np.argsort(-scores)
    tp = fp = 0
    npos, nneg = labels.sum(), len(labels) - labels.sum()
    tprs, fprs = [0.0], [0.0]
    for idx in order:
        tp += labels[idx]; fp += 1 - labels[idx]
        tprs.append(tp / npos); fprs.append(fp / nneg)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(tprs, fprs))


def main():
    dists = load_dists()
    models = sorted(dists.keys())
    common = sorted(set.intersection(*[set(d.keys()) for d in dists.values()]))
    print(f"{len(models)} models, {len(common)} probes (PhyloLM-style Nei similarity)")

    matrix = {}
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            matrix[(models[i], models[j])] = nei_sim(dists[models[i]], dists[models[j]], common)
    matrix_s = {**matrix, **{(b, a): v for (a, b), v in matrix.items()}}

    auc = auc_from(matrix, models)
    print(f"\nPhyloLM-style family AUC (26 models): {auc:.3f}")
    print(f"[our fingerprint, same subpool: Jaccard@20 output AUC = 0.920]")

    # Base identification (2 of 3 cases live in the 26-model subpool)
    for q, true_base in [("DeepSeek-R1-Distill-Qwen-7B", "Qwen2.5-Math-7B"),
                         ("DeepSeek-R1-Distill-Llama-8B", "Llama-3.1-8B-Instruct")]:
        ranked = sorted(((matrix_s[(q, m)], m) for m in models if m != q), reverse=True)
        hit = ranked[0][1] == true_base
        print(f"  base-ID {q}: top1={ranked[0][1]} ({ranked[0][0]:.3f}) "
              f"hit={hit} margin={ranked[0][0]/ranked[1][0]:.2f}x (top2={ranked[1][1]})")

    # Model-level permutation
    fams = [FAMILY_MAP.get(m, m) for m in models]
    def diff(labels):
        intra, inter = [], []
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                v = matrix[(models[i], models[j])]
                (intra if labels[i] == labels[j] else inter).append(v)
        return np.mean(intra) - np.mean(inter)
    obs = diff(fams)
    null = [diff(list(rng.permutation(fams))) for _ in range(5000)]
    p = np.mean([d >= obs for d in null])
    print(f"  permutation p={p:.5f} ({obs/np.std(null):.1f}σ)")

    json.dump({"auc": auc, "matrix": {f"{a}|{b}": v for (a, b), v in matrix.items()},
               "perm_p": float(p)},
              open(os.path.join(RESULTS_DIR, "phylolm_baseline.json"), "w"), indent=1)
    print("Saved → results/phylolm_baseline.json")


if __name__ == "__main__":
    main()

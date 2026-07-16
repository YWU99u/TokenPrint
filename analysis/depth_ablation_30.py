#!/usr/bin/env python3
"""
Depth ablation on the full 32-model calibration pool (was 26 — now 4 big models have
output logits from the rental session).

5 readout depths: L/4, L/2, 3L/4, L-2 (from internal fps), output (from _out.json).
For each: family AUC, pair-level Cohen's d, within/between ratio.

32-model calibration pool only (no controls, no ancestry adds).

Output: results/depth_ablation_30.json
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from paths import ROOT, FP_DIR, OUT_DIR, CKPT_DIR, CTRL_DIR, WITNESS_DIR, RESULTS_DIR
sys.path.insert(0, ROOT)

import json
import numpy as np

from fingerprint_v2 import FAMILY_MAP
K = 20

CONTROLS = {"Qwen2.5-7B", "gpt2-xl", "opt-6.7b", "OLMo-2-1124-7B",
            "pythia-1.4b-deduped", "pythia-6.9b-deduped", "pythia-12b-deduped",
            "Qwen2.5-Math-1.5B", "Qwen2.5-14B"}


def jac(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def auc(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    order = np.argsort(-scores)
    tp = fp = 0
    npos, nneg = labels.sum(), len(labels) - labels.sum()
    tprs, fprs = [0.0], [0.0]
    for i in order:
        tp += labels[i]; fp += 1 - labels[i]
        tprs.append(tp / npos); fprs.append(fp / nneg)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(tprs, fprs))


def main():
    # load all 30 calibration models' internal fps
    fps = {}
    for f in sorted(os.listdir(FP_DIR)):
        if f.endswith("_fp.json") and not f.startswith("cloud-test"):
            m = f.replace("_fp.json", "")
            if m in CONTROLS:
                continue
            # skip witness/quant models
            if any(x in m for x in ["Cerebras", "neox", "rwkv", "int4", "int8", "2.8b"]):
                continue
            fps[m] = json.load(open(os.path.join(FP_DIR, f)))
    models = sorted(fps.keys())
    print(f"calibration pool: {len(models)} models")
    assert len(models) == 32, f"expected 32, got {len(models)}: {models}"

    # load output baselines
    outs = {}
    for m in models:
        path = os.path.join(OUT_DIR, f"{m}_out.json")
        if os.path.exists(path):
            outs[m] = json.load(open(path))
    print(f"output baselines: {len(outs)} models")

    # depth fractions: 1/4, 1/2, 3/4, near-final, output
    depth_labels = ["L/4", "L/2", "3L/4", "L-2", "output"]
    results = {"n_models": len(models), "n_output": len(outs), "depths": {}}

    for di, dlabel in enumerate(depth_labels):
        if dlabel == "output":
            # use output baselines
            pool = [m for m in models if m in outs]
            def get_sets(m):
                return {p: frozenset(t for t, s in d[:K])
                        for p, d in outs[m]["probes"].items()
                        if isinstance(d, list)}
        else:
            pool = models
            frac_idx = di  # 0,1,2,3 -> target_layers[0..3]
            def get_sets(m, _fi=frac_idx):
                tl = fps[m]["target_layers"]
                lk = str(tl[_fi]) if _fi < len(tl) else str(tl[-1])
                return {p: frozenset(t for t, s in d[lk][:K])
                        for p, d in fps[m]["probes"].items()
                        if not (isinstance(d, dict) and "error" in d) and lk in d}

        sets = {m: get_sets(m) for m in pool}
        common = sorted(set.intersection(*[set(s.keys()) for s in sets.values()]))

        pair_list = [(pool[i], pool[j]) for i in range(len(pool)) for j in range(i+1, len(pool))]
        labels_arr, scores_arr = [], []
        for a, b in pair_list:
            s = float(np.mean([jac(sets[a][p], sets[b][p]) for p in common]))
            lab = 1 if FAMILY_MAP.get(a, a) == FAMILY_MAP.get(b, b) else 0
            labels_arr.append(lab); scores_arr.append(s)

        labels_arr = np.array(labels_arr); scores_arr = np.array(scores_arr)
        a_val = auc(labels_arr, scores_arr)
        intra = scores_arr[labels_arr == 1]; inter = scores_arr[labels_arr == 0]
        pooled = np.sqrt((np.var(intra, ddof=1) * (len(intra)-1) +
                          np.var(inter, ddof=1) * (len(inter)-1)) /
                         (len(intra) + len(inter) - 2))
        d_val = float((intra.mean() - inter.mean()) / pooled) if pooled > 0 else 0
        ratio = float(intra.mean() / inter.mean()) if inter.mean() > 0 else 0

        results["depths"][dlabel] = {
            "n_models": len(pool), "auc": round(a_val, 3),
            "cohens_d": round(d_val, 2), "within_between_ratio": round(ratio, 2),
            "intra_mean": round(float(intra.mean()), 3),
            "inter_mean": round(float(inter.mean()), 3)}
        print(f"  {dlabel:6s}  n={len(pool):2d}  AUC={a_val:.3f}  d={d_val:.2f}  "
              f"ratio={ratio:.2f}  intra={intra.mean():.3f}  inter={inter.mean():.3f}")

    with open(os.path.join(RESULTS_DIR, "depth_ablation_30.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("\nSaved -> results/depth_ablation_30.json")


if __name__ == "__main__":
    main()

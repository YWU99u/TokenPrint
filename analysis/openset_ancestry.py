#!/usr/bin/env python3
"""
B2: Open-set ancestry identification (pure re-analysis, near-final depth, k=20).

For each of the 5 R1 distillations, rank the 31 ancestry candidates
(30-model calibration pool + 2 documented-base additions, minus query), then:

  closed   : full candidate set (sanity check == paper Table 3)
  open-a   : documented base removed          -> who gets misattributed, at what
             score/margin (the auditor's "base not in pool" scenario)
  open-b   : base + all sibling distills removed -> true open set: no related
             model hosted at all

Abstention question: does any observable statistic (top-1 score, top-1/top-2
margin) separate "base present" from "base absent"? Reported per case +
probe-bootstrap (500) distributions.

Output: results/openset_ancestry.json
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from paths import ROOT, FP_DIR, OUT_DIR, CKPT_DIR, CTRL_DIR, WITNESS_DIR, RESULTS_DIR
sys.path.insert(0, ROOT)

import json
import numpy as np

from fingerprint_v2 import FAMILY_MAP
rng = np.random.default_rng(42)
K = 20
N_BOOT = 500

CONTROLS = {"Qwen2.5-7B", "gpt2-xl", "opt-6.7b", "OLMo-2-1124-7B",
            "pythia-1.4b-deduped", "pythia-6.9b-deduped", "pythia-12b-deduped"}

QUERIES = {  # query -> documented base (paper Table 3 convention)
    "DeepSeek-R1-Distill-Qwen-32B":  "Qwen2.5-32B-Instruct",
    "DeepSeek-R1-Distill-Qwen-14B":  "Qwen2.5-14B",
    "DeepSeek-R1-Distill-Qwen-7B":   "Qwen2.5-Math-7B",
    "DeepSeek-R1-Distill-Qwen-1.5B": "Qwen2.5-Math-1.5B",
    "DeepSeek-R1-Distill-Llama-8B":  "Llama-3.1-8B-Instruct",
}
SIBLING_FAMS = {"DS-Qwen", "DS-Llama"}


def nf_key(fp):
    ls_, nn = fp["target_layers"], fp["num_layers"]
    return str(min(ls_, key=lambda l: abs(l - (nn - 2))))


def jac(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def main():
    fps = {}
    for f in sorted(os.listdir(FP_DIR)):
        if f.endswith("_fp.json") and not f.startswith("cloud-test"):
            m = f.replace("_fp.json", "")
            if m in CONTROLS:
                continue
            fps[m] = json.load(open(os.path.join(FP_DIR, f)))
    models = sorted(fps.keys())
    print(f"ancestry pool: {len(models)} models ({len(models)-1} candidates/query)")

    sets = {}
    for m in models:
        lk = nf_key(fps[m])
        sets[m] = {p: frozenset(t for t, s in d[lk][:K])
                   for p, d in fps[m]["probes"].items()
                   if not (isinstance(d, dict) and "error" in d) and lk in d}
    common = sorted(set.intersection(*[set(s.keys()) for s in sets.values()]))
    P = len(common)
    print(f"common probes: {P}")

    # per-probe similarity arrays, query vs every candidate
    QP = {q: {m: np.array([jac(sets[q][p], sets[m][p]) for p in common])
              for m in models if m != q}
          for q in QUERIES}

    def rank(q, exclude, idx=None):
        scores = {m: float(v.mean() if idx is None else v[idx].mean())
                  for m, v in QP[q].items() if m not in exclude}
        r = sorted(scores.items(), key=lambda x: -x[1])
        top1, top2 = r[0], r[1]
        return {"top1": top1[0], "top1_score": round(top1[1], 4),
                "top2": top2[0], "top2_score": round(top2[1], 4),
                "margin": round(top1[1] / max(top2[1], 1e-9), 4)}

    out = {"pool_size": len(models), "n_probes": P, "cases": {}}
    header = f"{'condition':8s} {'top-1':34s} {'score':>6s} {'margin':>7s}"
    for q, base in QUERIES.items():
        siblings = {m for m in models
                    if FAMILY_MAP.get(m, m) in SIBLING_FAMS and m != q}
        conds = {"closed": {q},
                 "open-a": {q, base},
                 "open-b": {q, base} | siblings}
        case = {"documented_base": base}
        print(f"\n=== {q} (base: {base}) ===\n{header}")
        for cname, excl in conds.items():
            r = rank(q, excl)
            # probe bootstrap: distribution of top-1 identity and stats
            b_top1, b_scores, b_margins = {}, [], []
            for _ in range(N_BOOT):
                idx = rng.integers(0, P, P)
                rb = rank(q, excl, idx)
                b_top1[rb["top1"]] = b_top1.get(rb["top1"], 0) + 1
                b_scores.append(rb["top1_score"])
                b_margins.append(rb["margin"])
            r["boot_top1_freq"] = {m: c / N_BOOT for m, c in
                                   sorted(b_top1.items(), key=lambda x: -x[1])}
            r["boot_score_ci"] = [round(float(x), 4) for x in
                                  np.percentile(b_scores, [2.5, 97.5])]
            r["boot_margin_ci"] = [round(float(x), 4) for x in
                                   np.percentile(b_margins, [2.5, 97.5])]
            case[cname] = r
            flag = " <-- BASE" if r["top1"] == base else ""
            print(f"{cname:8s} {r['top1']:34s} {r['top1_score']:6.3f} "
                  f"{r['margin']:7.3f}{flag}")
        # abstention deltas
        case["score_drop_open_a"] = round(
            case["closed"]["top1_score"] - case["open-a"]["top1_score"], 4)
        case["score_drop_open_b"] = round(
            case["closed"]["top1_score"] - case["open-b"]["top1_score"], 4)
        out["cases"][q] = case

    # summary: can top-1 score or margin separate base-present from base-absent?
    print("\n=== abstention-signal summary (closed vs open-b) ===")
    print(f"{'query':30s} {'closed s1':>9s} {'open-b s1':>9s} {'drop':>7s} "
          f"{'closed mg':>9s} {'open-b mg':>9s}")
    for q, c in out["cases"].items():
        print(f"{q:30s} {c['closed']['top1_score']:9.3f} "
              f"{c['open-b']['top1_score']:9.3f} {c['score_drop_open_b']:7.3f} "
              f"{c['closed']['margin']:9.3f} {c['open-b']['margin']:9.3f}")

    with open(os.path.join(RESULTS_DIR, "openset_ancestry.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\nSaved -> results/openset_ancestry.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Output-level baseline (PhyloLM-style): top-100 tokens of the TRUE final
output distribution for each probe. Completes the depth-ablation curve:
1/4, 1/2, 3/4, near-final, output.

Usage: python3 output_baseline.py <model_name>   (one model per process)
       python3 output_baseline.py ANALYZE        (compare vs 3/4 fingerprints)
"""

import json, os, sys, gc, time, re
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

MODELS_DIR = "/home/user/Documents/ai_cogsci/llm_neurosci/models_dl"
OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "output_baseline")
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = "cuda:1"
TOP_K = 100


def extract(model_name):
    import torch
    from jlens import JLens
    from probes import prompt_texts
    from fingerprint_v2 import param_count, FAMILY_MAP

    out_path = os.path.join(OUT_DIR, f"{model_name}_out.json")
    if os.path.exists(out_path):
        print(f"SKIP {model_name}")
        return

    pc = param_count(model_name)
    dtype = torch.float16 if pc >= 5 else torch.float32
    jl = JLens(os.path.join(MODELS_DIR, model_name), device=DEVICE, dtype=dtype)

    result = {"model": model_name, "family": FAMILY_MAP.get(model_name, "unknown"),
              "num_layers": jl.num_layers, "probes": {}}
    t0 = time.time()
    with torch.no_grad():
        for i, prompt in enumerate(prompt_texts()):
            try:
                ids = jl.encode(prompt)
                logits = jl.model(ids, use_cache=False).logits[0, -1, :].float()
                top = jl.top_tokens(logits, TOP_K, filter_glitch=True)
                result["probes"][prompt] = [(t, round(s, 2)) for t, s in top]
            except Exception as e:
                result["probes"][prompt] = {"error": str(e)}
    print(f"  250 probes in {time.time()-t0:.0f}s")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    print(f"  Saved → {out_path}")


def analyze():
    from fingerprint_v2 import FAMILY_MAP, compute_pairwise, roc_auc
    from collections import defaultdict

    # Load output baselines
    outs = {}
    for f in sorted(os.listdir(OUT_DIR)):
        if f.endswith("_out.json"):
            outs[f.replace("_out.json", "")] = json.load(open(os.path.join(OUT_DIR, f)))
    models = sorted(outs.keys())
    print(f"{len(models)} output baselines loaded")

    def jac(a, b):
        return len(a & b) / len(a | b) if a and b else 0.0

    # Output-level pairwise Jaccard@20
    def out_matrix(k=20):
        matrix = {}
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                ma, mb = models[i], models[j]
                vals = []
                for p in outs[ma]["probes"]:
                    da, db = outs[ma]["probes"].get(p), outs[mb]["probes"].get(p)
                    if not da or not db: continue
                    if isinstance(da, dict) or isinstance(db, dict): continue
                    vals.append(jac(set(t for t, s in da[:k]), set(t for t, s in db[:k])))
                matrix[(ma, mb)] = float(np.mean(vals))
        return matrix

    om = out_matrix()

    # Family stats at output level
    fams = defaultdict(list)
    for m in models:
        fams[FAMILY_MAP.get(m, m)].append(m)
    intra_all, inter_all = [], []
    print(f"\n{'Family':12s} {'intra':>8s} {'inter':>8s} {'ratio':>7s}  (OUTPUT level)")
    fam_stats = {}
    for fam, members in sorted(fams.items()):
        if len(members) < 2: continue
        intra = [om[tuple(sorted((a, b)))] for i, a in enumerate(members) for b in members[i+1:]]
        others = [m for m in models if m not in members]
        inter = [om[tuple(sorted((a, b)))] for a in members for b in others]
        fam_stats[fam] = (np.mean(intra), np.mean(inter))
        print(f"{fam:12s} {np.mean(intra):8.3f} {np.mean(inter):8.3f} {np.mean(intra)/np.mean(inter):6.2f}x")

    # AUC at output level
    labels, scores = [], []
    for (ma, mb), v in om.items():
        labels.append(1 if FAMILY_MAP.get(ma, ma) == FAMILY_MAP.get(mb, mb) else 0)
        scores.append(v)
    labels, scores = np.array(labels), np.array(scores)
    order = np.argsort(-scores)
    tp = fp = 0; npos = labels.sum(); nneg = len(labels) - npos
    tprs, fprs = [0.0], [0.0]
    for idx in order:
        if labels[idx]: tp += 1
        else: fp += 1
        tprs.append(tp / npos); fprs.append(fp / nneg)
    trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc_out = float(trapz(tprs, fprs))

    # Cohen's d output level
    intra_v = scores[labels == 1]; inter_v = scores[labels == 0]
    pooled = np.sqrt((np.var(intra_v, ddof=1)*(len(intra_v)-1) + np.var(inter_v, ddof=1)*(len(inter_v)-1))
                     / (len(intra_v)+len(inter_v)-2))
    d_out = float((intra_v.mean() - inter_v.mean()) / pooled)

    print(f"\nOUTPUT level:  AUC={auc_out:.3f}  d={d_out:.2f}")
    print(f"[cf. 3/4 depth: AUC=0.876  d=2.24  |  near-final: AUC=0.872  d=1.83]")

    # Base-identification at output level
    kp = {}
    for a, b, label in [
        ("DeepSeek-R1-Distill-Qwen-7B", "Qwen2.5-Math-7B", "DS7B->Math7B"),
        ("DeepSeek-R1-Distill-Llama-8B", "Llama-3.1-8B-Instruct", "DSLlama->L31"),
        ("pythia-6.9b", "pythia-12b", "Py6.9<->12"),
    ]:
        if tuple(sorted((a, b))) in om:
            kp[label] = round(om[tuple(sorted((a, b)))], 3)
    print("key pairs @ output:", kp)

    json.dump({"auc": auc_out, "cohens_d": d_out,
               "family_stats": {k: {"intra": float(v[0]), "inter": float(v[1])} for k, v in fam_stats.items()},
               "matrix": {f"{a}|{b}": v for (a, b), v in om.items()},
               "key_pairs": kp},
              open(os.path.join(os.path.dirname(OUT_DIR), "output_baseline_analysis.json"), "w"), indent=1)
    print("Saved → results/output_baseline_analysis.json")


if __name__ == "__main__":
    if sys.argv[1] == "ANALYZE":
        analyze()
    else:
        extract(sys.argv[1])

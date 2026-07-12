#!/usr/bin/env python3
"""
Reviewer must-answer analyses (all from existing fingerprints, near-final depth).

M1. Tokenizer-stratified AUC: same-tokenizer pairs vs cross-tokenizer pairs;
    plus the shared-vocabulary Qwen subpool.
M2. Leave-one-domain-out: family AUC + base identification under domain removal.
M3. Separability of the two middle rungs (same-family vs successive-generation):
    probe-bootstrap CI on the difference of category means.
M4. Nearest-neighbor margin bootstrap CIs for the base-identification cases.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from paths import ROOT, FP_DIR, OUT_DIR, CKPT_DIR, CTRL_DIR, WITNESS_DIR, RESULTS_DIR
sys.path.insert(0, ROOT)

import json
import numpy as np
from collections import defaultdict

from fingerprint_v2 import FAMILY_MAP
from probes import PROBES
rng = np.random.default_rng(42)
K = 20

# 30-model main pool only (exclude controls)
CONTROLS = {"Qwen2.5-7B", "gpt2-xl", "opt-6.7b", "OLMo-2-1124-7B",
            "pythia-1.4b-deduped", "pythia-6.9b-deduped", "pythia-12b-deduped"}


def tok_group(m):
    if "Qwen" in m and "Llama" not in m: return "qwen"
    if "Llama" in m: return "llama3"
    if "pythia" in m: return "neox"
    if "gemma" in m: return "gemma"
    if "Mistral" in m or "Ministral" in m: return "mistralT"
    if "Phi" in m or "phi" in m: return "phiT"
    if "internlm" in m: return "ilmT"
    return m


def nf_key(fp):
    ls_, nn = fp["target_layers"], fp["num_layers"]
    return str(min(ls_, key=lambda l: abs(l - (nn - 2))))


def jac(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def auc_from_pairs(pairs):
    """pairs: list of (label, score)"""
    labels = np.array([l for l, s in pairs]); scores = np.array([s for l, s in pairs])
    if labels.sum() in (0, len(labels)): return None
    order = np.argsort(-scores)
    tp = fp = 0; npos, nneg = labels.sum(), len(labels) - labels.sum()
    tprs, fprs = [0.0], [0.0]
    for idx in order:
        tp += labels[idx]; fp += 1 - labels[idx]
        tprs.append(tp / npos); fprs.append(fp / nneg)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(tprs, fprs)), int(npos), int(nneg)


def main():
    fps = {}
    for f in sorted(os.listdir(FP_DIR)):
        if f.endswith("_fp.json") and not f.startswith("cloud-test"):
            m = f.replace("_fp.json", "")
            if m in CONTROLS: continue
            fps[m] = json.load(open(os.path.join(FP_DIR, f)))
    models = sorted(fps.keys())
    print(f"main pool: {len(models)} models")

    sets = {}
    for m in models:
        lk = nf_key(fps[m])
        sets[m] = {p: frozenset(t for t, s in d[lk][:K]) for p, d in fps[m]["probes"].items()
                   if not (isinstance(d, dict) and "error" in d) and lk in d}
    common = sorted(set.intersection(*[set(s.keys()) for s in sets.values()]))
    P = len(common)
    prompt_domain = {p: d for d, s, p in PROBES}

    # per-pair per-probe jaccard cache (heavy but fine)
    pair_list = [(models[i], models[j]) for i in range(len(models)) for j in range(i+1, len(models))]
    PP = {}
    for a, b in pair_list:
        PP[(a, b)] = np.array([jac(sets[a][p], sets[b][p]) for p in common])
    mean_sim = {k: float(v.mean()) for k, v in PP.items()}

    out = {}

    # ══ M1: tokenizer-stratified AUC ═══════════════════════════════
    same_tok, cross_tok = [], []
    for (a, b) in pair_list:
        lab = 1 if FAMILY_MAP.get(a, a) == FAMILY_MAP.get(b, b) else 0
        (same_tok if tok_group(a) == tok_group(b) else cross_tok).append((lab, mean_sim[(a, b)]))
    r_same = auc_from_pairs(same_tok)
    r_cross = auc_from_pairs(cross_tok)
    print(f"\nM1 tokenizer-stratified AUC:")
    print(f"  same-tokenizer pairs:  AUC={r_same[0]:.3f} (pos={r_same[1]}, neg={r_same[2]})")
    if r_cross:
        print(f"  cross-tokenizer pairs: AUC={r_cross[0]:.3f} (pos={r_cross[1]}, neg={r_cross[2]})")
    else:
        print("  cross-tokenizer pairs: no same-family positives exist across tokenizers")
    # Qwen shared-vocab subpool (Qwen3+Qwen2.5+Q25Math+DS-Qwen share the Qwen BPE)
    qpool = [m for m in models if tok_group(m) == "qwen"]
    qpairs = [(1 if FAMILY_MAP.get(a,a)==FAMILY_MAP.get(b,b) else 0, mean_sim[(a,b)] if (a,b) in mean_sim else mean_sim[(b,a)])
              for i,a in enumerate(qpool) for b in qpool[i+1:]]
    rq = auc_from_pairs(qpairs)
    print(f"  Qwen shared-vocab subpool ({len(qpool)} models): AUC={rq[0]:.3f}")
    # cross-tokenizer positives detail: which same-family pairs cross tokenizers? none by construction
    # so instead: cross-tok relatedness signal = crossgen/DS pairs? report ranking sanity:
    out["M1"] = {"same_tok_auc": r_same[0], "cross_tok_auc": r_cross[0] if r_cross else None,
                 "qwen_subpool_auc": rq[0]}

    # ══ M2: leave-one-domain-out ═══════════════════════════════════
    print(f"\nM2 leave-one-domain-out (family AUC / base-ID hits):")
    base_cases = [("DeepSeek-R1-Distill-Qwen-32B", "Qwen2.5-32B-Instruct"),
                  ("DeepSeek-R1-Distill-Llama-8B", "Llama-3.1-8B-Instruct"),
                  ("DeepSeek-R1-Distill-Qwen-7B", "Qwen2.5-Math-7B")]
    def eval_probeset(idx):
        ms = {k: float(v[idx].mean()) for k, v in PP.items()}
        pairs = [(1 if FAMILY_MAP.get(a,a)==FAMILY_MAP.get(b,b) else 0, ms[(a,b)]) for (a,b) in pair_list]
        auc = auc_from_pairs(pairs)[0]
        sym = {**ms, **{(b,a): v for (a,b),v in ms.items()}}
        hits = 0
        for q, base in base_cases:
            ranked = sorted(((sym[(q,m)], m) for m in models if m != q), reverse=True)
            hits += ranked[0][1] == base
        return auc, hits
    domains = sorted(set(prompt_domain[p] for p in common))
    all_idx = np.arange(P)
    auc_full, hits_full = eval_probeset(all_idx)
    print(f"  full 250 probes:            AUC={auc_full:.3f}  base-ID {hits_full}/3")
    out["M2"] = {"full": {"auc": auc_full, "base_id": hits_full}}
    for d in domains:
        idx = np.array([i for i, p in enumerate(common) if prompt_domain[p] != d])
        auc, hits = eval_probeset(idx)
        n_rm = P - len(idx)
        print(f"  - {d:12s} (drop {n_rm:2d}):    AUC={auc:.3f}  base-ID {hits}/3")
        out["M2"][d] = {"auc": auc, "base_id": hits}

    # ══ M3: middle-rung separability ═══════════════════════════════
    pythia = {"pythia-1.4b", "pythia-6.9b", "pythia-12b"}
    base_set = {tuple(sorted(p)) for p in base_cases}
    fam_pairs, cg_pairs = [], []
    for (a, b) in pair_list:
        key = tuple(sorted((a, b)))
        fa, fb = FAMILY_MAP.get(a, a), FAMILY_MAP.get(b, b)
        if a in pythia and b in pythia: continue
        if key in base_set: continue
        if fa == fb and fa != "DS-Qwen": fam_pairs.append((a, b))
        elif {fa, fb} == {"Qwen2.5", "Qwen3"}: cg_pairs.append((a, b))
    fam_mat = np.stack([PP[p] for p in fam_pairs])   # (n_pairs, P)
    cg_mat = np.stack([PP[p] for p in cg_pairs])
    obs = fam_mat.mean() - cg_mat.mean()
    boots = []
    for _ in range(5000):
        idx = rng.integers(0, P, P)
        boots.append(fam_mat[:, idx].mean() - cg_mat[:, idx].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_gt0 = float(np.mean([b <= 0 for b in boots]))
    print(f"\nM3 middle rungs: family({fam_mat.mean():.3f}) - crossgen({cg_mat.mean():.3f}) = {obs:+.4f}")
    print(f"  probe-bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}], P(diff<=0)={p_gt0:.3f}")
    out["M3"] = {"diff": float(obs), "ci": [float(lo), float(hi)], "p_leq0": p_gt0,
                 "separable": bool(lo > 0)}

    # ══ M4: NN-margin bootstrap CIs ════════════════════════════════
    print(f"\nM4 nearest-neighbor margin bootstrap CIs (500 resamples):")
    out["M4"] = {}
    for q, base in base_cases:
        others = [m for m in models if m != q]
        margins, hits = [], 0
        for _ in range(500):
            idx = rng.integers(0, P, P)
            ms = {m: float(PP[(q, m)][idx].mean() if (q, m) in PP else PP[(m, q)][idx].mean())
                  for m in others}
            ranked = sorted(ms.items(), key=lambda x: -x[1])
            hits += ranked[0][0] == base
            margins.append(ranked[0][1] / max(ranked[1][1], 1e-9))
        lo, hi = np.percentile(margins, [2.5, 97.5])
        print(f"  {q}:")
        print(f"    P(top-1 = true base) = {hits/500:.3f}   margin 95% CI [{lo:.2f}, {hi:.2f}]")
        out["M4"][q] = {"p_top1_is_base": hits/500, "margin_ci": [float(lo), float(hi)]}

    json.dump(out, open(os.path.join(RESULTS_DIR, "reviewer_musts.json"), "w"), indent=1)
    print("\nSaved → results/reviewer_musts.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Robustness suite addressing reviewer concerns. All computed from existing
fingerprints (near-final depth = last-2, the universal readout for all 30).

R1. Excess-corrected headline results (mismatched-probe null per pair):
    ladder category means, Pythia rung, base identification, family AUC.
R2. Leave-family-out / drop-Qwen-block / drop-Pythia robustness.
R3. Model-level jackknife (leave-one-model-out AUC range).
R4. k-sensitivity (Jaccard@10/20/50/100) at near-final.
R5. Pythia crystallization (same data & recipe across sizes -> controls the
    data-composition confound of the crystallization observation).
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
N_NULL = 200


def load():
    fps = {}
    for f in sorted(os.listdir(FP_DIR)):
        if f.endswith("_fp.json") and not f.startswith("cloud-test"):
            fps[f.replace("_fp.json", "")] = json.load(open(os.path.join(FP_DIR, f)))
    return fps


def nearfinal_key(fp):
    layers, n = fp["target_layers"], fp["num_layers"]
    return str(min(layers, key=lambda l: abs(l - (n - 2))))


def topk_sets(fp, k):
    lk = nearfinal_key(fp)
    return {p: frozenset(t for t, s in d[lk][:k]) for p, d in fp["probes"].items()
            if not (isinstance(d, dict) and "error" in d) and lk in d}


def jac(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def auc_from(matrix, models):
    labels, scores = [], []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            same = FAMILY_MAP.get(models[i], models[i]) == FAMILY_MAP.get(models[j], models[j])
            labels.append(1 if same else 0)
            scores.append(matrix[(models[i], models[j])])
    labels, scores = np.array(labels), np.array(scores)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return None
    order = np.argsort(-scores)
    tp = fp = 0
    npos, nneg = labels.sum(), len(labels) - labels.sum()
    tprs, fprs = [0.0], [0.0]
    for idx in order:
        tp += labels[idx]; fp += 1 - labels[idx]
        tprs.append(tp / npos); fprs.append(fp / nneg)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(tprs, fprs))


def perm_test(matrix, models, n_perm=10000):
    fams = [FAMILY_MAP.get(m, m) for m in models]
    def diff(labels):
        intra, inter = [], []
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                v = matrix[(models[i], models[j])]
                (intra if labels[i] == labels[j] else inter).append(v)
        if not intra or not inter:
            return 0.0
        return np.mean(intra) - np.mean(inter)
    obs = diff(fams)
    null = [diff(list(rng.permutation(fams))) for _ in range(n_perm)]
    p = float(np.mean([d >= obs for d in null]))
    return {"observed": float(obs), "p": p, "effect_sigma": float(obs / (np.std(null) + 1e-12))}


def main():
    fps = load()
    models = sorted(fps.keys())
    print(f"{len(models)} models")
    out = {}

    # Token sets at near-final, k=20 (primary) + other ks
    S = {k: {m: topk_sets(fps[m], k) for m in models} for k in [10, 20, 50, 100]}
    common = sorted(set.intersection(*[set(S[20][m].keys()) for m in models]))
    P = len(common)

    # Matched + mismatched-null matrices at k=20
    matched, null_m, excess = {}, {}, {}
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            ma, mb = models[i], models[j]
            sa, sb = S[20][ma], S[20][mb]
            mv = np.mean([jac(sa[p], sb[p]) for p in common])
            nv = np.mean([jac(sa[common[a]], sb[common[b]])
                          for a, b in rng.choice(P, (N_NULL, 2))])
            matched[(ma, mb)] = mv; null_m[(ma, mb)] = nv; excess[(ma, mb)] = mv - nv
    sym = lambda M: {**M, **{(b, a): v for (a, b), v in M.items()}}
    matched_s, excess_s = sym(matched), sym(excess)

    # ── R1: excess-corrected headlines ─────────────────────────────
    pythia = ["pythia-1.4b", "pythia-6.9b", "pythia-12b"]
    base_pairs = [("DeepSeek-R1-Distill-Qwen-32B", "Qwen2.5-32B-Instruct"),
                  ("DeepSeek-R1-Distill-Llama-8B", "Llama-3.1-8B-Instruct"),
                  ("DeepSeek-R1-Distill-Qwen-7B", "Qwen2.5-Math-7B")]
    base_set = {tuple(sorted(p)) for p in base_pairs}

    def ladder(M):
        cats = defaultdict(list)
        for i, a in enumerate(models):
            for b in models[i + 1:]:
                key = tuple(sorted((a, b)))
                fa, fb = FAMILY_MAP.get(a, a), FAMILY_MAP.get(b, b)
                v = M[(a, b)]
                if a in pythia and b in pythia: cats["identical"].append(v)
                elif key in base_set: cats["base"].append(v)
                elif fa == fb and fa != "DS-Qwen": cats["family"].append(v)
                elif {fa, fb} == {"Qwen2.5", "Qwen3"}: cats["crossgen"].append(v)
                elif fa != fb: cats["unrelated"].append(v)
        return {c: float(np.mean(v)) for c, v in cats.items()}

    out["ladder_raw"] = ladder(matched)
    out["ladder_excess"] = ladder(excess)
    print("\nLadder raw:   ", {k: round(v, 3) for k, v in out["ladder_raw"].items()})
    print("Ladder excess:", {k: round(v, 3) for k, v in out["ladder_excess"].items()})

    # Base-ID under excess
    base_id = {}
    for q, true_base in base_pairs:
        ranked = sorted(((excess_s[(q, m)], m) for m in models if m != q), reverse=True)
        base_id[q] = {"top1": ranked[0][1], "top1_v": round(ranked[0][0], 3),
                      "top2": ranked[1][1], "top2_v": round(ranked[1][0], 3),
                      "hit": ranked[0][1] == true_base,
                      "margin": round(ranked[0][0] / max(ranked[1][0], 1e-9), 2)}
    out["base_id_excess"] = base_id
    print("\nBase-ID under excess correction:")
    for q, r in base_id.items():
        print(f"  {q}: top1={r['top1']} ({r['top1_v']}) hit={r['hit']} margin={r['margin']}x")

    # AUC raw vs excess; permutation on both
    out["auc_raw"] = auc_from(matched, models)
    out["auc_excess"] = auc_from(excess, models)
    out["perm_raw"] = perm_test(matched, models)
    out["perm_excess"] = perm_test(excess, models)
    print(f"\nAUC raw={out['auc_raw']:.3f} excess={out['auc_excess']:.3f}")
    print(f"Perm raw p={out['perm_raw']['p']:.5f} ({out['perm_raw']['effect_sigma']:.1f}σ) | "
          f"excess p={out['perm_excess']['p']:.5f} ({out['perm_excess']['effect_sigma']:.1f}σ)")

    # ── R2: leave-family-out & block drops ─────────────────────────
    fams_present = sorted(set(FAMILY_MAP.get(m, m) for m in models))
    lfo = {}
    for fam in fams_present:
        keep = [m for m in models if FAMILY_MAP.get(m, m) != fam]
        sub = {(a, b): matched[(a, b)] if (a, b) in matched else matched[(b, a)]
               for i, a in enumerate(keep) for b in keep[i + 1:]}
        a = auc_from(sub, keep)
        if a: lfo[fam] = round(a, 3)
    out["leave_family_out_auc"] = lfo
    print("\nLeave-family-out AUC:", lfo)

    qwen_block = {"Qwen3", "Qwen2.5", "Q25-Math", "DS-Qwen"}
    keep = [m for m in models if FAMILY_MAP.get(m, m) not in qwen_block]
    sub = {(a, b): matched.get((a, b), matched.get((b, a)))
           for i, a in enumerate(keep) for b in keep[i + 1:]}
    out["drop_qwen_block"] = {"n_models": len(keep), "auc": auc_from(sub, keep),
                              "perm": perm_test(sub, keep, 5000)}
    print(f"Drop entire Qwen block ({len(keep)} left): AUC={out['drop_qwen_block']['auc']:.3f} "
          f"p={out['drop_qwen_block']['perm']['p']:.4f}")

    keep = [m for m in models if FAMILY_MAP.get(m, m) != "Pythia"]
    sub = {(a, b): matched.get((a, b), matched.get((b, a)))
           for i, a in enumerate(keep) for b in keep[i + 1:]}
    out["drop_pythia"] = {"auc": auc_from(sub, keep), "perm": perm_test(sub, keep, 5000)}
    print(f"Drop Pythia: AUC={out['drop_pythia']['auc']:.3f} p={out['drop_pythia']['perm']['p']:.4f}")

    # ── R3: model-level jackknife ──────────────────────────────────
    jk = []
    for held in models:
        keep = [m for m in models if m != held]
        sub = {(a, b): matched.get((a, b), matched.get((b, a)))
               for i, a in enumerate(keep) for b in keep[i + 1:]}
        a = auc_from(sub, keep)
        if a: jk.append(a)
    out["jackknife_auc"] = {"min": round(min(jk), 3), "max": round(max(jk), 3),
                            "mean": round(float(np.mean(jk)), 3)}
    print(f"\nJackknife AUC (leave-one-model-out): {out['jackknife_auc']}")

    # ── R4: k-sensitivity ──────────────────────────────────────────
    ksens = {}
    for k in [10, 20, 50, 100]:
        M = {}
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                ma, mb = models[i], models[j]
                M[(ma, mb)] = np.mean([jac(S[k][ma][p], S[k][mb][p]) for p in common])
        ksens[k] = round(auc_from(M, models), 3)
    out["k_sensitivity_auc"] = ksens
    print(f"\nk-sensitivity AUC: {ksens}")

    # ── R5: Pythia crystallization (same data & recipe) ────────────
    FA = json.load(open(os.path.join(RESULTS_DIR, "final_analysis.json")))
    C = FA["crystallization"]
    pyth = {m: {"num_layers": C[m]["num_layers"],
                "overlap_34": round(C[m]["self_overlap_by_depth"][-1][1], 3)}
            for m in pythia}
    out["pythia_crystallization"] = pyth
    print(f"\nPythia crystallization (same data/order/recipe):")
    for m, v in pyth.items():
        print(f"  {m} ({v['num_layers']}L): self-overlap@3/4 = {v['overlap_34']}")

    # Same-tokenizer null magnitudes at near-final (for W1/Pythia claim)
    def tok_group(m):
        if "Qwen" in m and "Llama" not in m: return "qwen"
        if "Llama" in m: return "llama3"
        if "pythia" in m: return "neox"
        if "gemma" in m: return "gemma"
        if "Mistral" in m or "Ministral" in m: return "mistral"
        if "Phi" in m or "phi" in m: return "phi"
        return m
    same_t = [v for (a, b), v in null_m.items() if tok_group(a) == tok_group(b)]
    diff_t = [v for (a, b), v in null_m.items() if tok_group(a) != tok_group(b)]
    out["null_same_tok"] = float(np.mean(same_t))
    out["null_diff_tok"] = float(np.mean(diff_t))
    print(f"\nNear-final mismatched null: same-tok {np.mean(same_t):.3f}, diff-tok {np.mean(diff_t):.3f}")

    # Pythia excess rung vs base rung explicitly (reviewer Q)
    py_excess = [excess[tuple(sorted((a, b)))] for i, a in enumerate(pythia) for b in pythia[i+1:]]
    base_excess = [excess[p if p in excess else (p[1], p[0])] for p in
                   [tuple(sorted(bp)) for bp in base_pairs]]
    out["pythia_rung_excess"] = float(np.mean(py_excess))
    out["base_rung_excess"] = float(np.mean(base_excess))
    print(f"Pythia rung excess: {np.mean(py_excess):.3f} | base rung excess: {np.mean(base_excess):.3f}")

    with open(os.path.join(RESULTS_DIR, "robustness_suite.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("\nSaved → results/robustness_suite.json")


if __name__ == "__main__":
    main()

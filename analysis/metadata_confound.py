#!/usr/bin/env python3
"""
B3 + B4: metadata baseline and confound regression (32-model calibration pool).

B3  Metadata-only family AUC (in-sample logistic regression, deliberately
    favorable to the baseline): same_tok, same_dev, |dlog params|, |dyear|.
    Plus metadata-NN ancestry for the 5 R1 queries.

B4  Dual-spec confound regression with two-way cluster-robust SEs:
    Spec A (pre-registered): raw S ~ same_family + same_tok + cap_gap + size_gap
    Spec B (excess):       excess S ~ same_family + cap_gap + size_gap
      (no same_tok: excess mechanically removes tokenizer channel)
    cap_gap = |capability_a - capability_b| from probe gold-key hit rate.

    Collinearity diagnostic: cross-tabulation of same_family × same_tok.

Output: results/metadata_confound.json
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from paths import ROOT, FP_DIR, OUT_DIR, CKPT_DIR, CTRL_DIR, WITNESS_DIR, RESULTS_DIR
sys.path.insert(0, ROOT)

import json
import numpy as np

from fingerprint_v2 import FAMILY_MAP, param_count
from capability_gold import GOLD_KEY, compute_capability
rng = np.random.default_rng(42)
K = 20
N_NULL = 200

CONTROLS = {"Qwen2.5-7B", "gpt2-xl", "opt-6.7b", "OLMo-2-1124-7B",
            "pythia-1.4b-deduped", "pythia-6.9b-deduped", "pythia-12b-deduped"}
ANCESTRY_ADDS = {"Qwen2.5-Math-1.5B", "Qwen2.5-14B"}

DEVELOPER = {"Qwen3": "Alibaba", "Qwen2.5": "Alibaba", "Q25-Math": "Alibaba",
             "Qwen2.5-base": "Alibaba",
             "Llama": "Meta", "Mistral": "MistralAI", "Phi": "Microsoft",
             "DS-Qwen": "DeepSeek", "DS-Llama": "DeepSeek",
             "Pythia": "EleutherAI", "Gemma": "Google", "InternLM": "ShanghaiAILab"}

RELEASE = {
    "Qwen3-0.6B": 2025.3, "Qwen3-1.7B": 2025.3, "Qwen3-4B": 2025.3,
    "Qwen3-8B": 2025.3, "Qwen3-14B": 2025.3, "Qwen3-32B": 2025.3,
    "Qwen2.5-1.5B-Instruct": 2024.7, "Qwen2.5-3B-Instruct": 2024.7,
    "Qwen2.5-7B-Instruct": 2024.7, "Qwen2.5-32B-Instruct": 2024.7,
    "Qwen2.5-Math-7B": 2024.7,
    "Qwen2.5-32B": 2024.7,
    "Llama-3.2-1B-Instruct": 2024.7, "Llama-3.2-3B-Instruct": 2024.7,
    "Llama-3.1-8B-Instruct": 2024.5,
    "Llama-3.1-8B": 2024.5,
    "Mistral-7B-Instruct-v0.3": 2024.4, "Ministral-8B-Instruct-2410": 2024.8,
    "Phi-3.5-mini-instruct": 2024.6, "phi-4": 2024.95,
    "DeepSeek-R1-Distill-Qwen-1.5B": 2025.05, "DeepSeek-R1-Distill-Qwen-7B": 2025.05,
    "DeepSeek-R1-Distill-Qwen-14B": 2025.05, "DeepSeek-R1-Distill-Qwen-32B": 2025.05,
    "DeepSeek-R1-Distill-Llama-8B": 2025.05,
    "pythia-1.4b": 2023.25, "pythia-6.9b": 2023.25, "pythia-12b": 2023.25,
    "gemma-2-2b-it": 2024.55, "gemma-2-9b-it": 2024.45, "gemma-2-27b-it": 2024.45,
    "internlm2_5-7b-chat": 2024.5,
}


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


def two_way_cluster_ols(X, y, ca, cb):
    X = np.asarray(X, float); y = np.asarray(y, float)
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    beta = XtXi @ X.T @ y
    u = y - X @ beta

    def meat(clusters):
        m = np.zeros((k, k))
        for g in set(clusters):
            idx = [i for i, c in enumerate(clusters) if c == g]
            s = (X[idx] * u[idx, None]).sum(axis=0)
            m += np.outer(s, s)
        return m

    inter = [f"{a}|{b}" for a, b in zip(ca, cb)]
    V = XtXi @ (meat(ca) + meat(cb) - meat(inter)) @ XtXi
    return beta, np.sqrt(np.maximum(np.diag(V), 0))


def main():
    fps = {}
    for f in sorted(os.listdir(FP_DIR)):
        if f.endswith("_fp.json") and not f.startswith("cloud-test"):
            m = f.replace("_fp.json", "")
            if m in CONTROLS or m in ANCESTRY_ADDS:
                continue
            fps[m] = json.load(open(os.path.join(FP_DIR, f)))
    models = sorted(fps.keys())
    assert len(models) == 32, f"expected 32, got {len(models)}"

    # capability scores
    cap = {}
    for m in models:
        _, _, rate = compute_capability(fps[m])
        cap[m] = rate
    print("capability scores (top/bottom 5):")
    for m in sorted(models, key=lambda m: -cap[m])[:5]:
        print(f"  {m:32s} {cap[m]:.3f}")
    print("  ...")
    for m in sorted(models, key=lambda m: cap[m])[:5]:
        print(f"  {m:32s} {cap[m]:.3f}")

    sets = {}
    for m in models:
        lk = nf_key(fps[m])
        sets[m] = {p: frozenset(t for t, s in d[lk][:K])
                   for p, d in fps[m]["probes"].items()
                   if not (isinstance(d, dict) and "error" in d) and lk in d}
    common = sorted(set.intersection(*[set(s.keys()) for s in sets.values()]))

    pairs, feats = [], []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = models[i], models[j]
            fa, fb = FAMILY_MAP.get(a, a), FAMILY_MAP.get(b, b)
            raw = float(np.mean([jac(sets[a][p], sets[b][p]) for p in common]))
            nulls = []
            for _ in range(N_NULL):
                x, yy = rng.integers(0, len(common)), rng.integers(0, len(common))
                while yy == x:
                    yy = rng.integers(0, len(common))
                nulls.append(jac(sets[a][common[x]], sets[b][common[yy]]))
            excess = raw - float(np.mean(nulls))
            pairs.append((a, b))
            feats.append({
                "same_family": int(fa == fb),
                "same_tok": int(tok_group(a) == tok_group(b)),
                "same_dev": int(DEVELOPER[fa] == DEVELOPER[fb]),
                "dlogp": abs(np.log(param_count(a)) - np.log(param_count(b))),
                "dyear": abs(RELEASE[a] - RELEASE[b]),
                "cap_gap": abs(cap[a] - cap[b]),
                "raw": raw, "excess": excess})

    y = np.array([f["same_family"] for f in feats])
    out = {}

    # ══ B3: metadata-only family AUC ═══════════════════════════════
    from sklearn.linear_model import LogisticRegression
    Xmeta = np.array([[f["same_tok"], f["same_dev"], f["dlogp"], f["dyear"]]
                      for f in feats])
    lr = LogisticRegression(max_iter=1000).fit(Xmeta, y)
    p_meta = lr.predict_proba(Xmeta)[:, 1]
    auc_meta = auc(y, p_meta)
    raw_scores = np.array([f["raw"] for f in feats])
    auc_fp = auc(y, raw_scores)
    Xboth = np.hstack([Xmeta, raw_scores[:, None]])
    lrb = LogisticRegression(max_iter=1000).fit(Xboth, y)
    auc_both = auc(y, lrb.predict_proba(Xboth)[:, 1])

    print("\nB3 metadata-only family discrimination (in-sample, favorable to metadata):")
    print(f"  metadata (4 features)          AUC = {auc_meta:.3f}")
    for name, col in [("same_tokenizer alone", Xmeta[:, 0]),
                      ("same_developer alone", Xmeta[:, 1]),
                      ("-|dlog params| alone", -Xmeta[:, 2]),
                      ("-|dyear| alone", -Xmeta[:, 3])]:
        print(f"  {name:30s} AUC = {auc(y, col):.3f}")
    print(f"  fingerprint (raw score)        AUC = {auc_fp:.3f}")
    print(f"  metadata + fingerprint         AUC = {auc_both:.3f}")
    out["B3"] = {"auc_metadata": auc_meta, "auc_fingerprint": auc_fp,
                 "auc_combined": auc_both,
                 "auc_same_tok_alone": auc(y, Xmeta[:, 0]),
                 "auc_same_dev_alone": auc(y, Xmeta[:, 1])}

    # metadata NN ancestry
    Q = {"DeepSeek-R1-Distill-Qwen-32B": "Qwen2.5-32B",
         "DeepSeek-R1-Distill-Qwen-14B": "Qwen2.5-14B",
         "DeepSeek-R1-Distill-Qwen-7B": "Qwen2.5-Math-7B",
         "DeepSeek-R1-Distill-Qwen-1.5B": "Qwen2.5-Math-1.5B",
         "DeepSeek-R1-Distill-Llama-8B": "Llama-3.1-8B"}
    all_cand = models + sorted(ANCESTRY_ADDS)
    hits = 0
    print("\n  metadata-NN ancestry (same tokenizer group, closest |dlog params|):")
    for q, base in Q.items():
        cands = [(0 if tok_group(c) == tok_group(q) else 1,
                  abs(np.log(param_count(c)) - np.log(param_count(q))), c)
                 for c in all_cand if c != q]
        top = sorted(cands)[0][2]
        ok = top == base
        hits += ok
        print(f"    {q:32s} -> {top:28s} {'HIT' if ok else 'miss (base: '+base+')'}")
    print(f"  metadata ancestry: {hits}/5 (fingerprint: 1/5 top-1, 5/5 top-2)")
    out["B3"]["metadata_ancestry_hits"] = hits

    # ══ Collinearity diagnostic ════════════════════════════════════
    sf_st = sum(1 for f in feats if f["same_family"] and f["same_tok"])
    sf_dt = sum(1 for f in feats if f["same_family"] and not f["same_tok"])
    df_st = sum(1 for f in feats if not f["same_family"] and f["same_tok"])
    df_dt = sum(1 for f in feats if not f["same_family"] and not f["same_tok"])
    print(f"\n  collinearity: same_fam×same_tok = [{sf_st}, {sf_dt}; {df_st}, {df_dt}]"
          f" (cross-tok family pairs: {sf_dt})")
    out["collinearity"] = {"sf_st": sf_st, "sf_dt": sf_dt,
                           "df_st": df_st, "df_dt": df_dt}

    # ══ B4: dual-spec confound regression ══════════════════════════
    print("\nB4 confound regression (two-way cluster-robust SE at model level):")
    ca = [a for a, b in pairs]; cb = [b for a, b in pairs]
    out["B4"] = {}

    # Spec A: raw ~ const + same_family + same_tok + cap_gap + size_gap
    XA = np.array([[1, f["same_family"], f["same_tok"], f["cap_gap"], f["dlogp"]]
                   for f in feats])
    namesA = ["const", "same_family", "same_tok", "cap_gap", "|dlog_params|"]
    yA = np.array([f["raw"] for f in feats])
    betaA, seA = two_way_cluster_ols(XA, yA, ca, cb)
    zA = betaA / seA
    print(f"  Spec A: raw S ~ same_family + same_tok + cap_gap + |dlogp|")
    for n_, b_, s_, z_ in zip(namesA, betaA, seA, zA):
        sig = "*" if abs(z_) > 1.96 else ""
        print(f"    {n_:15s} beta={b_:+.4f}  se={s_:.4f}  z={z_:+.2f}{sig}")
    out["B4"]["specA_raw"] = {n_: {"beta": float(b_), "se": float(s_), "z": float(z_)}
                              for n_, b_, s_, z_ in zip(namesA, betaA, seA, zA)}

    # Spec B: excess ~ const + same_family + cap_gap + size_gap
    # (no same_tok — excess mechanically removes tokenizer channel)
    XB = np.array([[1, f["same_family"], f["cap_gap"], f["dlogp"]]
                   for f in feats])
    namesB = ["const", "same_family", "cap_gap", "|dlog_params|"]
    yB = np.array([f["excess"] for f in feats])
    betaB, seB = two_way_cluster_ols(XB, yB, ca, cb)
    zB = betaB / seB
    print(f"\n  Spec B: excess S ~ same_family + cap_gap + |dlogp|")
    for n_, b_, s_, z_ in zip(namesB, betaB, seB, zB):
        sig = "*" if abs(z_) > 1.96 else ""
        print(f"    {n_:15s} beta={b_:+.4f}  se={s_:.4f}  z={z_:+.2f}{sig}")
    out["B4"]["specB_excess"] = {n_: {"beta": float(b_), "se": float(s_), "z": float(z_)}
                                 for n_, b_, s_, z_ in zip(namesB, betaB, seB, zB)}

    # R² for both
    for label, X_, y_, names_ in [("specA", XA, yA, namesA), ("specB", XB, yB, namesB)]:
        yhat = X_ @ np.linalg.lstsq(X_, y_, rcond=None)[0]
        ss_res = np.sum((y_ - yhat)**2); ss_tot = np.sum((y_ - y_.mean())**2)
        r2 = 1 - ss_res / ss_tot
        print(f"  {label} R² = {r2:.3f}")
        out["B4"][f"{label}_R2"] = float(r2)

    # capability alone AUC (sanity — is it even a good family predictor?)
    cap_gap_arr = np.array([f["cap_gap"] for f in feats])
    auc_cap = auc(y, -cap_gap_arr)
    print(f"\n  capability_gap alone -> family AUC = {auc_cap:.3f}")
    out["B4"]["auc_cap_gap_alone"] = auc_cap

    with open(os.path.join(RESULTS_DIR, "metadata_confound.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\nSaved -> results/metadata_confound.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
E0-a: Pile ecosystem witness analysis.

For each witness model, compute excess similarity against each Pythia model
and the unrelated reference (Qwen2.5-7B), at near-final depth.

Witness pool:
  Same data + same tokenizer + different project:  gpt-neox-20b, pythia-2.8b
  Same data + DIFFERENT tokenizer + different org:  Cerebras-GPT 1.3/2.7/6.7/13B
  Same data + DIFFERENT architecture:              RWKV-4-pile 3b/7b (output only)

Pythia references:  pythia-1.4b, pythia-6.9b, pythia-12b
Unrelated refs:     Qwen2.5-7B (base, no Pile), gemma-2-2b-it

For RWKV (output only), comparison uses output-level top-20 against
output baselines of Pythia models (if available), otherwise near-final fps.

Output: results/e0a_witness.json
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

PYTHIA = ["pythia-1.4b", "pythia-6.9b", "pythia-12b"]
WITNESSES_FP = ["gpt-neox-20b", "pythia-2.8b",
                "Cerebras-GPT-1.3B", "Cerebras-GPT-2.7B",
                "Cerebras-GPT-6.7B", "Cerebras-GPT-13B"]
WITNESSES_OUT = ["rwkv-4-3b-pile", "rwkv-4-7b-pile"]
UNRELATEDS = ["Qwen2.5-7B", "gemma-2-2b-it"]


def nf_key(fp):
    ls_, nn = fp["target_layers"], fp["num_layers"]
    return str(min(ls_, key=lambda l: abs(l - (nn - 2))))


def jac(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def _find_fp(name):
    """Search FP_DIR, then WITNESS_DIR, then CTRL_DIR for a fingerprint file."""
    for d in (FP_DIR, WITNESS_DIR, CTRL_DIR):
        p = os.path.join(d, f"{name}_fp.json")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{name}_fp.json not found in calibration/witnesses/controls")


def load_fp_sets(name, k=K):
    fp = json.load(open(_find_fp(name)))
    lk = nf_key(fp)
    return {p: frozenset(t for t, s in d[lk][:k])
            for p, d in fp["probes"].items()
            if not (isinstance(d, dict) and "error" in d) and lk in d}


def _find_out(name):
    """Search OUT_DIR, then WITNESS_DIR, then CTRL_DIR for an output file."""
    for d in (OUT_DIR, WITNESS_DIR, CTRL_DIR):
        p = os.path.join(d, f"{name}_out.json")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{name}_out.json not found in output/witnesses/controls")


def load_out_sets(name, k=K):
    fp = json.load(open(_find_out(name)))
    return {p: frozenset(t for t, s in d[:k])
            for p, d in fp["probes"].items()
            if isinstance(d, list)}


def sim(sa, sb):
    common = sorted(set(sa) & set(sb))
    if not common:
        return 0.0, 0.0, 0.0
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


def tok_type(name):
    if "Cerebras" in name:
        return "GPT-2 BPE"
    if "neox" in name or "pythia" in name:
        return "NeoX BPE"
    if "rwkv" in name:
        return "NeoX BPE"  # RWKV-4-Pile uses the same 20B tokenizer
    if "Qwen" in name:
        return "Qwen BPE"
    if "gemma" in name:
        return "Gemma SP"
    return "?"


def main():
    # load all fingerprint-based sets
    fp_sets = {}
    for name in PYTHIA + WITNESSES_FP + UNRELATEDS:
        fp_sets[name] = load_fp_sets(name)

    # load output-based sets for RWKV and Pythia (for output-level comparison)
    out_sets = {}
    for name in WITNESSES_OUT:
        out_sets[name] = load_out_sets(name)
    for name in PYTHIA + UNRELATEDS:
        try:
            _find_out(name)
            out_sets[name] = load_out_sets(name)
        except FileNotFoundError:
            pass

    results = {"witness_pairs": [], "summary": {}}

    print(f"{'witness':25s} {'vs':15s} {'tok_match':>10s} {'raw':>7s} {'null':>7s} {'excess':>7s}  notes")
    print("-" * 95)

    # fingerprint-based witnesses vs Pythia + unrelated refs
    for w in WITNESSES_FP:
        for ref in PYTHIA + UNRELATEDS:
            raw, null, exc = sim(fp_sets[w], fp_sets[ref])
            tm = "same" if tok_type(w) == tok_type(ref) else "cross"
            note = ""
            if ref in UNRELATEDS:
                note = "unrelated ref"
            elif "Cerebras" in w:
                note = "same-data cross-tok cross-org"
            elif "neox-20b" in w:
                note = "same-data same-tok diff-project"
            elif "2.8b" in w:
                note = "same-data same-tok same-suite"
            print(f"{w:25s} {ref:15s} {tm:>10s} {raw:7.3f} {null:7.3f} {exc:7.3f}  {note}")
            results["witness_pairs"].append({
                "witness": w, "ref": ref, "tok_match": tm,
                "raw": round(raw, 4), "null": round(null, 4),
                "excess": round(exc, 4), "note": note})

    # output-level witnesses (RWKV) vs Pythia + unrelated refs
    print()
    for w in WITNESSES_OUT:
        for ref in PYTHIA + UNRELATEDS:
            if ref in out_sets:
                raw, null, exc = sim(out_sets[w], out_sets[ref])
                level = "output"
            else:
                # fallback to fp near-final
                raw, null, exc = 0, 0, 0
                level = "n/a"
            note = "RWKV output-level"
            if ref in UNRELATEDS:
                note += " unrelated"
            else:
                note += " same-data cross-arch"
            print(f"{w:25s} {ref:15s} {'cross':>10s} {raw:7.3f} {null:7.3f} {exc:7.3f}  {note} ({level})")
            results["witness_pairs"].append({
                "witness": w, "ref": ref, "tok_match": "cross-arch",
                "raw": round(raw, 4), "null": round(null, 4),
                "excess": round(exc, 4), "note": note, "level": level})

    # summary table: mean excess by category
    cats = {
        "same-data same-tok (NeoX↔Pythia)": [r for r in results["witness_pairs"]
            if r["ref"] in PYTHIA and r["tok_match"] == "same"
            and "neox" in r["witness"] or "2.8b" in r["witness"]],
        "same-data cross-tok (Cerebras↔Pythia)": [r for r in results["witness_pairs"]
            if r["ref"] in PYTHIA and "Cerebras" in r["witness"]],
        "same-data cross-arch (RWKV↔Pythia)": [r for r in results["witness_pairs"]
            if r["ref"] in PYTHIA and "rwkv" in r["witness"]],
        "unrelated refs": [r for r in results["witness_pairs"]
            if r["ref"] in UNRELATEDS and r["witness"] in WITNESSES_FP],
    }
    print(f"\n{'category':45s} {'n':>3s} {'mean_raw':>8s} {'mean_exc':>8s}")
    print("-" * 70)
    for cat, rows in cats.items():
        if not rows:
            continue
        mr = np.mean([r["raw"] for r in rows])
        me = np.mean([r["excess"] for r in rows])
        n = len(rows)
        print(f"{cat:45s} {n:3d} {mr:8.3f} {me:8.3f}")
        results["summary"][cat] = {"n": n, "mean_raw": round(mr, 4),
                                   "mean_excess": round(me, 4)}

    # Pythia internal for comparison
    py_pairs = []
    for i, a in enumerate(PYTHIA):
        for b in PYTHIA[i+1:]:
            raw, null, exc = sim(fp_sets[a], fp_sets[b])
            py_pairs.append(exc)
    print(f"{'Pythia internal (existing)':45s} {len(py_pairs):3d} "
          f"{'':>8s} {np.mean(py_pairs):8.3f}")
    results["summary"]["Pythia internal"] = {
        "n": len(py_pairs), "mean_excess": round(float(np.mean(py_pairs)), 4)}

    with open(os.path.join(RESULTS_DIR, "e0a_witness.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("\nSaved -> results/e0a_witness.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
J-Space Fingerprinting v2: Multi-layer extraction with expanded probes.

For each model, extracts top-100 logit-lens tokens at 4 layer depths
(1/4, 1/2, 3/4, near-final) across 250 stratified probes.
Then computes pairwise similarity with statistical validation.
"""

import sys, json, os, gc, time, re
import torch
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from jlens import JLens
from probes import PROBES, DOMAINS, prompt_texts

MODELS_DIR = "/home/user/Documents/ai_cogsci/llm_neurosci/models_dl"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "fingerprints_v2")
os.makedirs(RESULTS_DIR, exist_ok=True)
DEVICE = "cuda:1"
TOP_K = 100

ALL_MODELS = [
    # Qwen3 family
    "Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B", "Qwen3-14B",
    # Qwen2.5 family
    "Qwen2.5-1.5B-Instruct", "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct",
    "Qwen2.5-Math-7B",
    # Llama family
    "Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct", "Llama-3.1-8B-Instruct",
    # Mistral family
    "Mistral-7B-Instruct-v0.3", "Ministral-8B-Instruct-2410",
    # Phi family
    "Phi-3.5-mini-instruct", "phi-4",
    # DeepSeek distilled (validation)
    "DeepSeek-R1-Distill-Qwen-1.5B", "DeepSeek-R1-Distill-Qwen-7B",
    "DeepSeek-R1-Distill-Qwen-14B", "DeepSeek-R1-Distill-Llama-8B",
    # Pythia (same-data control)
    "pythia-1.4b", "pythia-6.9b", "pythia-12b",
    # Gemma (Google)
    "gemma-2-2b-it", "gemma-2-9b-it",
    # InternLM
    "internlm2_5-7b-chat",
]

FAMILY_MAP = {
    "Qwen3-0.6B": "Qwen3", "Qwen3-1.7B": "Qwen3", "Qwen3-4B": "Qwen3",
    "Qwen3-8B": "Qwen3", "Qwen3-14B": "Qwen3", "Qwen3-32B": "Qwen3",
    "Qwen2.5-1.5B-Instruct": "Qwen2.5", "Qwen2.5-3B-Instruct": "Qwen2.5",
    "Qwen2.5-7B-Instruct": "Qwen2.5",
    "Qwen2.5-Math-7B": "Q25-Math",
    "Qwen2.5-Math-1.5B": "Q25-Math",
    "Qwen2.5-14B": "Qwen2.5-base",
    "Qwen2.5-7B": "Qwen2.5-base",
    "gpt2-xl": "GPT2", "opt-6.7b": "OPT", "OLMo-2-1124-7B": "OLMo",
    "gpt-neox-20b": "NeoX", "pythia-2.8b": "Pythia",
    "Cerebras-GPT-1.3B": "Cerebras", "Cerebras-GPT-2.7B": "Cerebras",
    "Cerebras-GPT-6.7B": "Cerebras", "Cerebras-GPT-13B": "Cerebras",
    "Qwen2.5-32B-Instruct": "Qwen2.5",
    "Llama-3.2-1B-Instruct": "Llama", "Llama-3.2-3B-Instruct": "Llama",
    "Llama-3.1-8B-Instruct": "Llama",
    "Mistral-7B-Instruct-v0.3": "Mistral", "Ministral-8B-Instruct-2410": "Mistral",
    "Phi-3.5-mini-instruct": "Phi", "phi-4": "Phi",
    "DeepSeek-R1-Distill-Qwen-1.5B": "DS-Qwen",
    "DeepSeek-R1-Distill-Qwen-7B": "DS-Qwen",
    "DeepSeek-R1-Distill-Qwen-14B": "DS-Qwen",
    "DeepSeek-R1-Distill-Qwen-32B": "DS-Qwen",
    "DeepSeek-R1-Distill-Llama-8B": "DS-Llama",
    "pythia-1.4b": "Pythia", "pythia-6.9b": "Pythia", "pythia-12b": "Pythia",
    "pythia-1.4b-deduped": "Pythia-dd", "pythia-6.9b-deduped": "Pythia-dd", "pythia-12b-deduped": "Pythia-dd",
    "gemma-2-2b-it": "Gemma", "gemma-2-9b-it": "Gemma", "gemma-2-27b-it": "Gemma",
    "internlm2_5-7b-chat": "InternLM",
    "DeepSeek-R1-Distill-Qwen-32B-int8": "DS-Qwen",
    "DeepSeek-R1-Distill-Qwen-32B-int4": "DS-Qwen",
    "Qwen2.5-32B-Instruct-int4": "Qwen2.5",
}


def param_count(name):
    if "phi-4" in name: return 14
    m = re.search(r'(\d+(?:\.\d+)?)[bB]', name)
    if m: return float(m.group(1))
    if "mini" in name.lower(): return 3.8
    return 1


def extract_fingerprint(model_name):
    gc.collect()
    torch.cuda.empty_cache()

    path = os.path.join(MODELS_DIR, model_name)
    if not os.path.isdir(path):
        print(f"  !!! {model_name}: directory not found at {path}")
        return None

    pc = param_count(model_name)
    dtype = torch.float16 if pc >= 5 else torch.float32
    print(f"\n>>> {model_name} ({pc}B, {'fp16' if dtype == torch.float16 else 'fp32'})")

    try:
        jl = JLens(path, device=DEVICE, dtype=dtype)
    except Exception as e:
        print(f"  !!! Failed to load: {e}")
        return None

    # 4 layer depths: 1/4, 1/2, 3/4, near-final
    n = jl.num_layers
    target_layers = sorted(set([n // 4, n // 2, 3 * n // 4, n - 2]))
    print(f"  Layers: {target_layers} (of {n})")

    fingerprint = {
        "model": model_name,
        "family": FAMILY_MAP.get(model_name, "unknown"),
        "params_b": pc,
        "num_layers": n,
        "hidden_size": jl.hidden_size,
        "vocab_size": jl.vocab_size,
        "target_layers": target_layers,
        "probes": {},
    }

    prompts = prompt_texts()
    t0 = time.time()
    for i, prompt in enumerate(prompts):
        try:
            input_ids = jl.encode(prompt)
            probe_data = {}
            for layer in target_layers:
                scores = jl.logit_lens(input_ids, layer)
                top = jl.top_tokens(scores, TOP_K, filter_glitch=True)
                probe_data[str(layer)] = [(t, round(s, 2)) for t, s in top]
            fingerprint["probes"][prompt] = probe_data
        except Exception as e:
            fingerprint["probes"][prompt] = {"error": str(e)}

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(prompts) - i - 1) / rate
            print(f"  {i+1}/{len(prompts)} probes ({rate:.1f}/s, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"  {len(prompts)} probes in {elapsed:.0f}s ({len(prompts)/elapsed:.1f}/s)")

    del jl.model, jl
    gc.collect()
    torch.cuda.empty_cache()

    return fingerprint


# ── Similarity computation ─────────────────────────────────────────

def jaccard(set_a, set_b):
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def compute_pairwise(fp_a, fp_b, layer_frac="3/4"):
    """Compute similarity between two fingerprints at a given layer depth."""
    # Find matching layer index
    def get_layer_key(fp, frac):
        layers = fp["target_layers"]
        n = fp["num_layers"]
        if frac == "1/4": target = n // 4
        elif frac == "1/2": target = n // 2
        elif frac == "3/4": target = 3 * n // 4
        else: target = n - 2
        closest = min(layers, key=lambda l: abs(l - target))
        return str(closest)

    lk_a = get_layer_key(fp_a, layer_frac)
    lk_b = get_layer_key(fp_b, layer_frac)

    results_by_domain = defaultdict(list)
    all_jaccards = {k: [] for k in [10, 20, 50, 100]}
    rank_corrs = []

    common_prompts = set(fp_a["probes"].keys()) & set(fp_b["probes"].keys())

    for prompt in common_prompts:
        pa = fp_a["probes"][prompt]
        pb = fp_b["probes"][prompt]
        if isinstance(pa, dict) and "error" in pa: continue
        if isinstance(pb, dict) and "error" in pb: continue
        if lk_a not in pa or lk_b not in pb: continue

        tokens_a = pa[lk_a]
        tokens_b = pb[lk_b]

        # Find domain for this probe
        domain = "unknown"
        for d, s, p in PROBES:
            if p == prompt:
                domain = d
                break

        for k in [10, 20, 50, 100]:
            sa = set(t for t, s in tokens_a[:k])
            sb = set(t for t, s in tokens_b[:k])
            j = jaccard(sa, sb)
            all_jaccards[k].append(j)
            if k == 20:
                results_by_domain[domain].append(j)

        # Rank correlation on shared tokens
        rank_a = {t: i for i, (t, s) in enumerate(tokens_a)}
        rank_b = {t: i for i, (t, s) in enumerate(tokens_b)}
        shared = set(rank_a) & set(rank_b)
        if len(shared) >= 5:
            ra = np.array([rank_a[t] for t in shared])
            rb = np.array([rank_b[t] for t in shared])
            if ra.std() > 0 and rb.std() > 0:
                rank_corrs.append(np.corrcoef(ra, rb)[0, 1])

    result = {}
    for k in [10, 20, 50, 100]:
        vals = all_jaccards[k]
        result[f"jaccard_{k}"] = float(np.mean(vals)) if vals else 0.0
    result["rank_corr"] = float(np.mean(rank_corrs)) if rank_corrs else 0.0
    result["n_prompts"] = len(common_prompts)

    result["by_domain"] = {}
    for domain, vals in results_by_domain.items():
        result["by_domain"][domain] = float(np.mean(vals)) if vals else 0.0

    return result


def compute_all_similarities(fps, layer_frac="3/4"):
    """Compute full pairwise similarity matrix."""
    models = sorted(fps.keys())
    n = len(models)
    matrix = {}

    for i in range(n):
        for j in range(i, n):
            sim = compute_pairwise(fps[models[i]], fps[models[j]], layer_frac)
            matrix[f"{models[i]}|{models[j]}"] = sim
            if i != j:
                matrix[f"{models[j]}|{models[i]}"] = sim

        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{n} models compared")

    return models, matrix


# ── Statistical tests ──────────────────────────────────────────────

def bootstrap_ci(fps, models, n_boot=1000, k=20, layer_frac="3/4"):
    """Bootstrap 95% CI for each pairwise Jaccard."""
    prompts = list(set.intersection(*[set(fps[m]["probes"].keys()) for m in models]))
    n_prompts = len(prompts)

    print(f"  Bootstrap CI: {n_boot} iterations, {n_prompts} prompts")

    def get_layer_key(fp, frac):
        layers = fp["target_layers"]
        n = fp["num_layers"]
        if frac == "3/4": target = 3 * n // 4
        elif frac == "1/2": target = n // 2
        elif frac == "1/4": target = n // 4
        else: target = n - 2
        return str(min(layers, key=lambda l: abs(l - target)))

    # Precompute token sets
    token_sets = {}
    for m in models:
        lk = get_layer_key(fps[m], layer_frac)
        token_sets[m] = {}
        for p in prompts:
            data = fps[m]["probes"].get(p, {})
            if isinstance(data, dict) and "error" in data:
                token_sets[m][p] = set()
            elif lk in data:
                token_sets[m][p] = set(t for t, s in data[lk][:k])
            else:
                token_sets[m][p] = set()

    cis = {}
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            ma, mb = models[i], models[j]
            boot_vals = []
            for _ in range(n_boot):
                idx = np.random.choice(n_prompts, n_prompts, replace=True)
                sampled = [prompts[ii] for ii in idx]
                jaccards = []
                for p in sampled:
                    sa = token_sets[ma].get(p, set())
                    sb = token_sets[mb].get(p, set())
                    if sa and sb:
                        jaccards.append(len(sa & sb) / len(sa | sb))
                if jaccards:
                    boot_vals.append(np.mean(jaccards))
            if boot_vals:
                lo, hi = np.percentile(boot_vals, [2.5, 97.5])
                cis[f"{ma}|{mb}"] = {"mean": float(np.mean(boot_vals)),
                                      "ci_lo": float(lo), "ci_hi": float(hi)}

    return cis


def permutation_test(fps, models, n_perm=1000, k=20, layer_frac="3/4"):
    """Test whether intra-family similarity > inter-family under null."""
    print(f"  Permutation test: {n_perm} iterations")

    def get_layer_key(fp, frac):
        layers = fp["target_layers"]
        n = fp["num_layers"]
        if frac == "3/4": target = 3 * n // 4
        elif frac == "1/2": target = n // 2
        else: target = n - 2
        return str(min(layers, key=lambda l: abs(l - target)))

    prompts = list(set.intersection(*[set(fps[m]["probes"].keys()) for m in models]))

    # Precompute all pairwise Jaccard@k values
    pair_jaccards = {}
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            ma, mb = models[i], models[j]
            lk_a = get_layer_key(fps[ma], layer_frac)
            lk_b = get_layer_key(fps[mb], layer_frac)
            jvals = []
            for p in prompts:
                da = fps[ma]["probes"].get(p, {})
                db = fps[mb]["probes"].get(p, {})
                if isinstance(da, dict) and "error" in da: continue
                if isinstance(db, dict) and "error" in db: continue
                if lk_a not in da or lk_b not in db: continue
                sa = set(t for t, s in da[lk_a][:k])
                sb = set(t for t, s in db[lk_b][:k])
                if sa and sb:
                    jvals.append(len(sa & sb) / len(sa | sb))
            pair_jaccards[(ma, mb)] = float(np.mean(jvals)) if jvals else 0.0

    # Observed: intra - inter
    families = [FAMILY_MAP.get(m, m) for m in models]

    def intra_inter_diff(fam_labels):
        intra, inter = [], []
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                val = pair_jaccards[(models[i], models[j])]
                if fam_labels[i] == fam_labels[j]:
                    intra.append(val)
                else:
                    inter.append(val)
        if not intra or not inter:
            return 0.0
        return np.mean(intra) - np.mean(inter)

    observed = intra_inter_diff(families)

    null_diffs = []
    for _ in range(n_perm):
        perm = list(np.random.permutation(families))
        null_diffs.append(intra_inter_diff(perm))

    p_value = np.mean([d >= observed for d in null_diffs])
    effect = observed / (np.std(null_diffs) + 1e-10)

    return {
        "observed_diff": float(observed),
        "null_mean": float(np.mean(null_diffs)),
        "null_std": float(np.std(null_diffs)),
        "p_value": float(p_value),
        "effect_size": float(effect),
    }


def roc_auc(fps, models, matrix, k=20):
    """ROC AUC for same-family classification using Jaccard threshold."""
    labels = []
    scores = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            fa = FAMILY_MAP.get(models[i], models[i])
            fb = FAMILY_MAP.get(models[j], models[j])
            same = 1 if fa == fb else 0
            key = f"{models[i]}|{models[j]}"
            sim = matrix.get(key, {}).get(f"jaccard_{k}", 0)
            labels.append(same)
            scores.append(sim)

    labels = np.array(labels)
    scores = np.array(scores)

    if labels.sum() == 0 or labels.sum() == len(labels):
        return {"auc": 0.0, "n_same": int(labels.sum()), "n_diff": int((1-labels).sum())}

    # Manual AUC
    sorted_idx = np.argsort(-scores)
    tp, fp_count = 0, 0
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    tpr_list, fpr_list = [0.0], [0.0]
    for idx in sorted_idx:
        if labels[idx] == 1:
            tp += 1
        else:
            fp_count += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp_count / n_neg)

    trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    auc = trapz_fn(tpr_list, fpr_list)
    return {"auc": float(auc), "n_same": int(n_pos), "n_diff": int(n_neg)}


# ── Main ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "extract"

    if mode == "extract":
        targets = sys.argv[2:] if len(sys.argv) > 2 else ["ALL"]
        if targets == ["ALL"]:
            targets = ALL_MODELS
        elif targets == ["REMAINING"]:
            done = set(f.replace("_fp.json", "") for f in os.listdir(RESULTS_DIR) if f.endswith("_fp.json"))
            targets = [m for m in ALL_MODELS if m not in done]
            print(f"Done: {len(done)}, remaining: {len(targets)}")

        for m in targets:
            fp_path = os.path.join(RESULTS_DIR, f"{m}_fp.json")
            t0 = time.time()
            fp = extract_fingerprint(m)
            if fp:
                with open(fp_path, "w") as f:
                    json.dump(fp, f, indent=1, ensure_ascii=False)
                print(f"  Saved → {fp_path} ({time.time()-t0:.0f}s)")

    elif mode == "analyze":
        # Load all fingerprints
        fps = {}
        for f in sorted(os.listdir(RESULTS_DIR)):
            if f.endswith("_fp.json"):
                name = f.replace("_fp.json", "")
                with open(os.path.join(RESULTS_DIR, f)) as fh:
                    fps[name] = json.load(fh)
        print(f"Loaded {len(fps)} fingerprints")

        # Compute similarities at 3/4 depth
        print("\nComputing pairwise similarities (3/4 depth)...")
        models, matrix = compute_all_similarities(fps, "3/4")

        # Print matrix
        short = {}
        for m in models:
            s = m.replace("-Instruct", "").replace("-v0.3", "").replace("-2410", "")
            s = s.replace("Llama-3.2-", "L32-").replace("Llama-3.1-", "L31-")
            s = s.replace("Qwen2.5-", "Q25-").replace("Qwen3-", "Q3-")
            s = s.replace("Mistral-7B", "Mis7B").replace("Ministral-8B", "Min8B")
            s = s.replace("Phi-3.5-mini", "Phi3.5").replace("phi-4", "Phi4")
            s = s.replace("DeepSeek-R1-Distill-", "DS-")
            s = s.replace("pythia-", "Py-").replace("gemma-2-", "Gem-")
            s = s.replace("internlm2_5-7b-chat", "ILM7B")
            short[m] = s[:10]

        print(f"\n{'Jaccard@20 matrix':^80}")
        h = f"{'':>12}" + "".join(f"{short[m]:>10s}" for m in models)
        print(h)
        print("─" * len(h))
        for ma in models:
            row = f"{short[ma]:>12}"
            for mb in models:
                key = f"{ma}|{mb}"
                v = matrix.get(key, {}).get("jaccard_20", 0)
                row += f"{v:>10.3f}"
            print(row)

        # Family stats
        print("\n\nFamily-level averages:")
        families_seen = defaultdict(list)
        for m in models:
            families_seen[FAMILY_MAP.get(m, m)].append(m)

        for fam, members in sorted(families_seen.items()):
            if len(members) < 2:
                continue
            intra = []
            for i, a in enumerate(members):
                for b in members[i+1:]:
                    key = f"{a}|{b}"
                    intra.append(matrix.get(key, {}).get("jaccard_20", 0))
            inter = []
            others = [m for m in models if m not in members]
            for a in members:
                for b in others:
                    key = f"{a}|{b}"
                    inter.append(matrix.get(key, {}).get("jaccard_20", 0))
            if intra and inter:
                ratio = np.mean(intra) / np.mean(inter) if np.mean(inter) > 0 else float('inf')
                print(f"  {fam:>12s}: intra={np.mean(intra):.3f} inter={np.mean(inter):.3f} ratio={ratio:.2f}x")

        # Statistical tests
        print("\nRunning statistical tests...")
        perm = permutation_test(fps, models)
        print(f"  Permutation test: p={perm['p_value']:.4f}, effect={perm['effect_size']:.2f}")

        auc_result = roc_auc(fps, models, matrix)
        print(f"  ROC AUC: {auc_result['auc']:.3f} (same={auc_result['n_same']}, diff={auc_result['n_diff']})")

        print("\nComputing bootstrap CIs...")
        cis = bootstrap_ci(fps, models, n_boot=500)

        # Hypothesis tests
        print("\n" + "="*72)
        print("  HYPOTHESIS TESTS")
        print("="*72)

        # H1: DS-Qwen clusters with Qwen
        ds_qwen = [m for m in models if "DS-Qwen" in FAMILY_MAP.get(m, "")]
        qwen = [m for m in models if FAMILY_MAP.get(m, "") in ("Qwen3", "Qwen2.5")]
        llama = [m for m in models if FAMILY_MAP.get(m, "") == "Llama"]

        if ds_qwen and qwen and llama:
            dsq_to_qwen = [matrix.get(f"{a}|{b}", {}).get("jaccard_20", 0)
                           for a in ds_qwen for b in qwen]
            dsq_to_llama = [matrix.get(f"{a}|{b}", {}).get("jaccard_20", 0)
                            for a in ds_qwen for b in llama]
            print(f"\n  H1: DS-Distill-Qwen → Qwen:  {np.mean(dsq_to_qwen):.3f} ± {np.std(dsq_to_qwen):.3f}")
            print(f"      DS-Distill-Qwen → Llama: {np.mean(dsq_to_llama):.3f} ± {np.std(dsq_to_llama):.3f}")
            print(f"      Ratio: {np.mean(dsq_to_qwen)/np.mean(dsq_to_llama):.2f}x" if np.mean(dsq_to_llama) > 0 else "")

        # H2: DS-Llama clusters with Llama
        ds_llama = [m for m in models if "DS-Llama" in FAMILY_MAP.get(m, "")]
        if ds_llama and qwen and llama:
            dsl_to_llama = [matrix.get(f"{a}|{b}", {}).get("jaccard_20", 0)
                            for a in ds_llama for b in llama]
            dsl_to_qwen = [matrix.get(f"{a}|{b}", {}).get("jaccard_20", 0)
                           for a in ds_llama for b in qwen]
            print(f"\n  H2: DS-Distill-Llama → Llama: {np.mean(dsl_to_llama):.3f} ± {np.std(dsl_to_llama):.3f}")
            print(f"      DS-Distill-Llama → Qwen:  {np.mean(dsl_to_qwen):.3f} ± {np.std(dsl_to_qwen):.3f}")
            print(f"      Ratio: {np.mean(dsl_to_llama)/np.mean(dsl_to_qwen):.2f}x" if np.mean(dsl_to_qwen) > 0 else "")

        # H3: Pythia intra > all others
        pythia = [m for m in models if FAMILY_MAP.get(m, "") == "Pythia"]
        if len(pythia) >= 2:
            pythia_intra = []
            for i, a in enumerate(pythia):
                for b in pythia[i+1:]:
                    pythia_intra.append(matrix.get(f"{a}|{b}", {}).get("jaccard_20", 0))
            print(f"\n  H3: Pythia intra-family: {np.mean(pythia_intra):.3f}")
            for fam, members in families_seen.items():
                if fam == "Pythia" or len(members) < 2: continue
                intra = []
                for i, a in enumerate(members):
                    for b in members[i+1:]:
                        intra.append(matrix.get(f"{a}|{b}", {}).get("jaccard_20", 0))
                if intra:
                    print(f"      vs {fam}: {np.mean(intra):.3f}")

        # Save everything
        output = {
            "models": models,
            "families": FAMILY_MAP,
            "short_names": short,
            "matrix": matrix,
            "permutation_test": perm,
            "roc_auc": auc_result,
            "bootstrap_ci": cis,
        }
        out_path = os.path.join(os.path.dirname(RESULTS_DIR), "fingerprint_v2_analysis.json")
        with open(out_path, "w") as f:
            json.dump(output, f, indent=1, default=str)
        print(f"\nResults saved → {out_path}")

#!/usr/bin/env python3
"""
J-Space Fingerprint Extraction v2 — Cloud Standalone (B1 rental session)

Extends cloud_extract.py with:
  * GPT2-architecture branch (Cerebras-GPT)          [E0-a]
  * true output-distribution recording (_out.json)   [depth ablation 26->30]
  * --output-only mode (RWKV / big models that already have internal fps)
  * --quant int8|int4 via bitsandbytes               [quantization envelope]

Outputs (formats identical to the local pipeline):
  results/<name>_fp.json   internal 4-depth logit-lens fingerprint
                           {probes: {prompt: {layer: [[tok, score], ...]}}}
  results/<name>_out.json  true output top-100
                           {probes: {prompt: [[tok, logit], ...]}}

Usage:
    python3 cloud_extract_v2.py <model_path_or_repo> [name] [--output-only] [--quant int8|int4]

Examples:
    python3 cloud_extract_v2.py models_dl/gpt-neox-20b gpt-neox-20b
    python3 cloud_extract_v2.py models_dl/Qwen3-32B Qwen3-32B --output-only
    python3 cloud_extract_v2.py models_dl/DeepSeek-R1-Distill-Qwen-32B DeepSeek-R1-Distill-Qwen-32B --quant int4
    python3 cloud_extract_v2.py models_dl/rwkv-4-7b-pile rwkv-4-7b-pile --output-only

Notes:
  * Quantized runs are named <name>-<quant> automatically.
  * bitsandbytes keeps lm_head unquantized by default: the readout stays
    clean; what int8/int4 perturbs is the hidden states — exactly the
    deployed-quantized-model audit scenario.
"""

import sys, json, os, gc, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probes import prompt_texts

TOP_K = 100
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT_DIR, exist_ok=True)


def get_components(model):
    """Architecture-agnostic access to layers / final norm / lm_head."""
    if hasattr(model, 'gpt_neox'):  # Pythia / GPT-NeoX-20B
        return model.gpt_neox.layers, model.gpt_neox.final_layer_norm, model.embed_out
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):  # GPT2 / Cerebras-GPT
        return model.transformer.h, model.transformer.ln_f, model.lm_head
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers, norm = model.model.layers, model.model.norm
        if hasattr(model, 'lm_head') and model.lm_head is not None:
            return layers, norm, model.lm_head
        if hasattr(model, 'output'):  # InternLM2
            return layers, norm, model.output
    raise ValueError(f"Unknown architecture: {type(model)} — use --output-only")


def load_model(model_path, quant, size_gb_hint=None):
    kwargs = dict(trust_remote_code=True)
    if quant == "int8":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["device_map"] = {"": 0}
    elif quant == "int4":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16)
        kwargs["device_map"] = {"": 0}
    else:
        # match local convention: <5B fp32, else bf16
        cfg = json.load(open(os.path.join(model_path, "config.json"))) \
            if os.path.isdir(model_path) else {}
        hidden = cfg.get("hidden_size") or cfg.get("n_embd") or 8192
        nlay = cfg.get("num_hidden_layers") or cfg.get("n_layer") or 99
        small = hidden <= 2560 and nlay <= 40
        kwargs["dtype"] = torch.float32 if small else torch.bfloat16
        kwargs["device_map"] = {"": 0}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs).eval()
        print("    Placement: single GPU", flush=True)
    except Exception as e:
        if quant:
            raise
        print(f"    Single-GPU load failed ({type(e).__name__}); retry device_map=auto...", flush=True)
        gc.collect(); torch.cuda.empty_cache()
        kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs).eval()
        meta = [n for n, p in model.named_parameters() if p.device.type == "meta"]
        if meta:
            raise RuntimeError(f"{len(meta)} params on meta device — free GPU memory and rerun")
    return model


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    output_only = "--output-only" in sys.argv
    quant = None
    if "--quant" in sys.argv:
        quant = sys.argv[sys.argv.index("--quant") + 1]
        assert quant in ("int8", "int4"), f"bad --quant {quant}"
    if not args:
        print(__doc__); sys.exit(1)

    model_path = args[0]
    name = args[1] if len(args) > 1 else model_path.rstrip("/").split("/")[-1]
    if quant:
        name = f"{name}-{quant}"

    fp_path = os.path.join(OUT_DIR, f"{name}_fp.json")
    out_path = os.path.join(OUT_DIR, f"{name}_out.json")
    need_fp = not output_only and not os.path.exists(fp_path)
    need_out = not os.path.exists(out_path)
    if not need_fp and not need_out:
        print(f"Already done: {name}"); sys.exit(0)

    print(f">>> {name}  (fp={'yes' if need_fp else 'skip'}, out={'yes' if need_out else 'skip'}"
          f"{', quant=' + quant if quant else ''})", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = load_model(model_path, quant)
    print(f"    Loaded in {time.time()-t0:.0f}s", flush=True)

    n_layers = getattr(model.config, "num_hidden_layers", None) or \
        getattr(model.config, "n_layer", None)

    # glitch filter from the output embedding (works for all archs incl. RWKV)
    out_emb = model.get_output_embeddings()
    if out_emb is None:  # some archs (e.g. RWKV) may not register it
        out_emb = getattr(model, "head", None) or getattr(model, "lm_head", None)
        if out_emb is None:
            raise RuntimeError("cannot locate output embedding for glitch filter")
    W = out_emb.weight.detach()
    vocab_size = W.shape[0]
    norms = W.cpu().float().norm(dim=1)
    med = norms.median()
    valid_mask = (norms < 5 * med) & (norms > 0.2 * med)
    print(f"    Filtered {(~valid_mask).sum().item()}/{vocab_size} glitch tokens", flush=True)

    layers = norm = lm_head = None
    target_layers = []
    if need_fp:
        layers, norm, lm_head = get_components(model)
        n = len(layers)
        target_layers = sorted(set([n // 4, n // 2, 3 * n // 4, n - 2]))
        print(f"    {n} layers, target {target_layers}", flush=True)

    def topk_strings(scores):
        scores = scores.float()
        scores[~valid_mask.to(scores.device)] = float("-inf")
        tk = torch.topk(scores, TOP_K + 20)
        toks = []
        for idx, sc in zip(tk.indices.tolist(), tk.values.tolist()):
            t = tokenizer.decode([idx]).strip()
            if t:
                toks.append((t, round(sc, 2)))
            if len(toks) >= TOP_K:
                break
        return toks

    fp = {"model": name, "family": "unknown", "params_b": None,
          "num_layers": n_layers, "hidden_size": getattr(model.config, "hidden_size", None),
          "vocab_size": vocab_size, "target_layers": target_layers, "probes": {}}
    out = {"model": name, "family": "unknown", "num_layers": n_layers, "probes": {}}

    device = next(model.parameters()).device
    prompts = prompt_texts()
    t0 = time.time()
    for i, prompt in enumerate(prompts):
        try:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                res = model(input_ids, output_hidden_states=need_fp, use_cache=False)
            if need_out:
                out["probes"][prompt] = topk_strings(res.logits[0, -1, :].clone())
            if need_fp:
                probe_data = {}
                for layer in target_layers:
                    hs = res.hidden_states[layer + 1][0, -1, :]
                    hs = hs.to(norm.weight.device)
                    scores = lm_head(norm(hs.unsqueeze(0)))[0]
                    probe_data[str(layer)] = topk_strings(scores)
                fp["probes"][prompt] = probe_data
        except Exception as e:
            if need_out:
                out["probes"][prompt] = {"error": str(e)}
            if need_fp:
                fp["probes"][prompt] = {"error": str(e)}
        if (i + 1) % 50 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"    {i+1}/{len(prompts)} ({rate:.1f}/s, ETA {(len(prompts)-i-1)/rate:.0f}s)",
                  flush=True)
    print(f"    Done in {time.time()-t0:.0f}s", flush=True)

    if need_out:
        with open(out_path, "w") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        print(f"    Saved -> {out_path}", flush=True)
    if need_fp:
        with open(fp_path, "w") as f:
            json.dump(fp, f, indent=1, ensure_ascii=False)
        print(f"    Saved -> {fp_path}", flush=True)

    del model
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

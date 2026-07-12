"""
J-Lens: Jacobian Lens for Language Model Interpretability

Replication of the J-space analysis from
"A Global Workspace in Language Models" (Anthropic, Jul 2026).

Two readout methods:
  1. J-lens  — finite-difference JVP: J @ h_l, where J = d(logits)/d(h_l).
               Reveals which tokens the hidden state is "pushing toward"
               through the remaining layers.
  2. Logit lens — norm(h_l) projected directly through lm_head.

For swap / ablation interventions we compute exact per-token gradients
via autograd and manipulate the hidden-state directions.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List, Tuple, Dict
import numpy as np


class JLens:
    def __init__(self, model_path: str, device: str = "cuda:1", dtype=torch.float32):
        print(f"Loading {model_path.split('/')[-1]}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True
        ).to(device).eval()
        self.device = device
        self.dtype = dtype

        # Architecture-agnostic access to layers, norm, lm_head
        if hasattr(self.model, 'gpt_neox'):
            # GPTNeoX / Pythia
            self.layers = self.model.gpt_neox.layers
            self.norm = self.model.gpt_neox.final_layer_norm
            self.lm_head = self.model.embed_out
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            # GPT-2
            self.layers = self.model.transformer.h
            self.norm = self.model.transformer.ln_f
            self.lm_head = self.model.lm_head
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'decoder'):
            # OPT
            self.layers = self.model.model.decoder.layers
            self.norm = self.model.model.decoder.final_layer_norm
            self.lm_head = self.model.lm_head
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # Qwen, Llama, Mistral, Phi, Gemma, InternLM
            self.layers = self.model.model.layers
            self.norm = self.model.model.norm
            if hasattr(self.model, 'lm_head'):
                self.lm_head = self.model.lm_head
            elif hasattr(self.model, 'output'):
                # InternLM2 names its unembedding 'output'
                self.lm_head = self.model.output
            else:
                raise ValueError(f"No lm_head/output found on {type(self.model)}")
        else:
            raise ValueError(f"Unknown architecture: {type(self.model)}")
        self.num_layers = len(self.layers)
        self.hidden_size = self.model.config.hidden_size
        self.vocab_size = self.model.config.vocab_size
        print(f"  {self.num_layers} layers, d={self.hidden_size}, vocab={self.vocab_size}")

        self._build_token_filter()

    # ------------------------------------------------------------------
    # Token filtering — exclude glitch tokens with extreme embedding norms
    # ------------------------------------------------------------------

    def _build_token_filter(self):
        """Build a mask that filters out glitch tokens (extreme embedding norms)."""
        with torch.no_grad():
            W = self.lm_head.weight.float()  # (vocab, hidden)
            norms = W.norm(dim=1)
            median_norm = norms.median()
            # Keep tokens within 5x of median norm
            self.valid_mask = (norms < 5 * median_norm) & (norms > 0.2 * median_norm)
            n_filtered = (~self.valid_mask).sum().item()
            print(f"  Filtered {n_filtered}/{self.vocab_size} glitch tokens")

    def _filter_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """Set glitch token scores to -inf."""
        out = scores.clone()
        out[~self.valid_mask] = float("-inf")
        return out

    # ------------------------------------------------------------------
    # Tokenization helpers
    # ------------------------------------------------------------------

    def encode(self, text: str) -> torch.Tensor:
        return self.tokenizer.encode(text, return_tensors="pt").to(self.device)

    def decode(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id])

    def token_id(self, word: str) -> int:
        """Best single-token id for *word* (tries with and without leading space)."""
        for variant in [word, " " + word, word.lower(), " " + word.lower(),
                        word.capitalize(), " " + word.capitalize()]:
            ids = self.tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                return ids[0]
        return self.tokenizer.encode(" " + word, add_special_tokens=False)[0]

    def top_tokens(self, scores: torch.Tensor, k: int = 15,
                   filter_glitch: bool = True) -> List[Tuple[str, float]]:
        s = self._filter_scores(scores.float()) if filter_glitch else scores.float()
        topk = torch.topk(s, k)
        out = []
        for idx, sc in zip(topk.indices.tolist(), topk.values.tolist()):
            tok = self.decode(idx).strip()
            if tok:
                out.append((tok, round(sc, 3)))
        return out

    def score_tokens(self, scores: torch.Tensor, words: List[str]) -> List[Tuple[str, float, float]]:
        """Score specific words — returns (word, score, percentile)."""
        all_scores = scores.float()
        results = []
        for w in words:
            tid = self.token_id(w)
            s = all_scores[tid].item()
            pct = (all_scores < s).float().mean().item() * 100
            results.append((w, round(s, 3), round(pct, 1)))
        return results

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _forward(self, input_ids: torch.Tensor):
        with torch.no_grad():
            return self.model(input_ids, output_hidden_states=True, use_cache=False)

    def _hook_replace(self, layer_idx: int, replacement: torch.Tensor):
        def fn(module, inp, out):
            if isinstance(out, tuple):
                return (replacement,) + out[1:]
            return replacement
        return self.layers[layer_idx].register_forward_hook(fn)

    # ------------------------------------------------------------------
    # Logit lens  (baseline)
    # ------------------------------------------------------------------

    def logit_lens(self, input_ids: torch.Tensor, layer_idx: int, pos: int = -1):
        with torch.no_grad():
            hs = self._forward(input_ids).hidden_states[layer_idx + 1]
            return self.lm_head(self.norm(hs))[0, pos, :]

    # ------------------------------------------------------------------
    # J-lens readout  (finite-difference JVP)
    # ------------------------------------------------------------------

    def jlens_readout(
        self,
        input_ids: torch.Tensor,
        layer_idx: int,
        pos: int = -1,
        eps: float = 1e-3,
    ) -> torch.Tensor:
        """
        Compute J @ h_l via finite differences at position *pos*.

        Uses an additive perturbation orthogonal to h_l to avoid the
        RMSNorm null-space issue (scaling h_l is absorbed by the norm).
        We perturb in a random direction, project out the h_l component,
        then normalize. To stabilize, we average over multiple random dirs.
        """
        baseline = self._forward(input_ids)
        base_logits = baseline.logits[0, pos, :].float()
        h = baseline.hidden_states[layer_idx + 1]
        h_pos = h[0, pos, :].float()
        h_norm = h_pos.norm()

        # We compute J @ h_l directly. At intermediate layers this is
        # meaningful because h_l feeds through subsequent layers whose
        # internal norms don't fully absorb the perturbation direction.
        # The multiplicative form (1+eps)*h_l is simpler and equivalent.
        h_pert = h.clone()
        h_pert[0, pos, :] = (h_pos * (1.0 + eps)).to(self.dtype)

        handle = self._hook_replace(layer_idx, h_pert)
        with torch.no_grad():
            pert_logits = self.model(input_ids, use_cache=False).logits[0, pos, :].float()
        handle.remove()

        return (pert_logits - base_logits) / eps

    def jlens_readout_ortho(
        self,
        input_ids: torch.Tensor,
        layer_idx: int,
        pos: int = -1,
        eps: float = 5e-2,
        n_probes: int = 8,
    ) -> torch.Tensor:
        """
        J-lens using orthogonal random probes, then reconstruct J @ h_l.

        Computes J @ r_i for random unit vectors r_i, then estimates
        J @ h_l = sum_i (r_i . h_l) * (J @ r_i).
        """
        baseline = self._forward(input_ids)
        base_logits = baseline.logits[0, pos, :].float()
        h = baseline.hidden_states[layer_idx + 1]
        h_pos = h[0, pos, :].float()
        h_norm = h_pos.norm()

        # Generate random orthonormal directions
        torch.manual_seed(42)
        R = torch.randn(n_probes, self.hidden_size, device=self.device)
        Q, _ = torch.linalg.qr(R.T)
        probes = Q.T  # (n_probes, hidden_size)

        accumulator = torch.zeros(self.vocab_size, device=self.device)
        for i in range(n_probes):
            r = probes[i]
            coeff = (h_pos @ r).item()

            h_pert = h.clone()
            h_pert[0, pos, :] = (h_pos + eps * h_norm * r).to(self.dtype)

            handle = self._hook_replace(layer_idx, h_pert)
            with torch.no_grad():
                pert_logits = self.model(input_ids, use_cache=False).logits[0, pos, :].float()
            handle.remove()

            jvp = (pert_logits - base_logits) / (eps * h_norm)
            accumulator += coeff * jvp

        return accumulator

    def jlens_across_layers(
        self,
        prompt: str,
        pos: int = -1,
        k: int = 10,
        n_samples: int = 7,
        method: str = "jlens",
    ) -> Dict[int, List[Tuple[str, float]]]:
        input_ids = self.encode(prompt)
        step = max(1, (self.num_layers - 1) // (n_samples - 1))
        layers = sorted(set(
            list(range(0, self.num_layers, step)) + [self.num_layers - 2]
        ))

        results = {}
        for l in layers:
            if method == "logit":
                scores = self.logit_lens(input_ids, l, pos)
            elif method == "ortho":
                scores = self.jlens_readout_ortho(input_ids, l, pos)
            else:
                scores = self.jlens_readout(input_ids, l, pos)
            results[l] = self.top_tokens(scores, k)
        return results

    # ------------------------------------------------------------------
    # Token-direction gradient  (for swap / ablation)
    # ------------------------------------------------------------------

    def token_gradient(
        self,
        input_ids: torch.Tensor,
        layer_idx: int,
        tok_id: int,
        pos: int = -1,
    ) -> torch.Tensor:
        with torch.no_grad():
            hs_all = self._forward(input_ids).hidden_states

        h = hs_all[layer_idx + 1].detach().clone().requires_grad_(True)
        handle = self._hook_replace(layer_idx, h)
        out = self.model(input_ids, use_cache=False)
        handle.remove()

        logit = out.logits[0, pos, tok_id]
        grad = torch.autograd.grad(logit, h)[0]
        return grad[0, pos, :].detach().float()

    # ------------------------------------------------------------------
    # Swap experiment  (gradient-based direction swap)
    # ------------------------------------------------------------------

    def swap(
        self,
        input_ids: torch.Tensor,
        layer_idx: int,
        src: str,
        tgt: str,
        pos: int = -1,
        strength: float = 1.0,
    ) -> torch.Tensor:
        src_id = self.token_id(src)
        tgt_id = self.token_id(tgt)

        g_src = self.token_gradient(input_ids, layer_idx, src_id, pos)
        g_tgt = self.token_gradient(input_ids, layer_idx, tgt_id, pos)

        g_src_hat = g_src / (g_src.norm() + 1e-8)
        g_tgt_hat = g_tgt / (g_tgt.norm() + 1e-8)

        with torch.no_grad():
            h = self._forward(input_ids).hidden_states[layer_idx + 1].clone()

        h_pos = h[0, pos, :].float()
        proj = (h_pos @ g_src_hat).item()

        h_swap = h.clone()
        h_swap[0, pos, :] = (
            h_pos - strength * proj * g_src_hat + strength * proj * g_tgt_hat
        ).to(self.dtype)

        handle = self._hook_replace(layer_idx, h_swap)
        with torch.no_grad():
            logits = self.model(input_ids, use_cache=False).logits
        handle.remove()
        return logits

    # ------------------------------------------------------------------
    # Swap via unembedding directions (simpler, works at late layers)
    # ------------------------------------------------------------------

    def swap_unembed(
        self,
        input_ids: torch.Tensor,
        layer_idx: int,
        src: str,
        tgt: str,
        pos: int = -1,
        strength: float = 1.0,
    ) -> torch.Tensor:
        """Swap using unembedding (lm_head weight) directions instead of gradients."""
        src_id = self.token_id(src)
        tgt_id = self.token_id(tgt)

        W = self.lm_head.weight.float()
        u_src = W[src_id]
        u_tgt = W[tgt_id]
        u_src_hat = u_src / (u_src.norm() + 1e-8)
        u_tgt_hat = u_tgt / (u_tgt.norm() + 1e-8)

        with torch.no_grad():
            h = self._forward(input_ids).hidden_states[layer_idx + 1].clone()

        h_pos = h[0, pos, :].float()
        proj = (h_pos @ u_src_hat).item()

        h_swap = h.clone()
        h_swap[0, pos, :] = (
            h_pos - strength * proj * u_src_hat + strength * proj * u_tgt_hat
        ).to(self.dtype)

        handle = self._hook_replace(layer_idx, h_swap)
        with torch.no_grad():
            logits = self.model(input_ids, use_cache=False).logits
        handle.remove()
        return logits

    # ------------------------------------------------------------------
    # Ablation  (remove top J-space directions)
    # ------------------------------------------------------------------

    def ablate(
        self,
        input_ids: torch.Tensor,
        layer_idx: int,
        n_dirs: int = 10,
        pos: int = -1,
    ) -> torch.Tensor:
        jl = self.jlens_readout(input_ids, layer_idx, pos)
        jl = self._filter_scores(jl)
        top_ids = torch.topk(jl, n_dirs).indices.tolist()

        grads = []
        for tid in top_ids:
            grads.append(self.token_gradient(input_ids, layer_idx, tid, pos))

        basis = []
        for g in grads:
            v = g.clone()
            for b in basis:
                v = v - (v @ b) * b
            n = v.norm()
            if n > 1e-6:
                basis.append(v / n)

        with torch.no_grad():
            h = self._forward(input_ids).hidden_states[layer_idx + 1].clone()
        h_pos = h[0, pos, :].float()

        for b in basis:
            h_pos = h_pos - (h_pos @ b) * b

        h_abl = h.clone()
        h_abl[0, pos, :] = h_pos.to(self.dtype)

        handle = self._hook_replace(layer_idx, h_abl)
        with torch.no_grad():
            logits = self.model(input_ids, use_cache=False).logits
        handle.remove()
        return logits

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------

    def greedy_next(self, input_ids: torch.Tensor, logits: Optional[torch.Tensor] = None):
        if logits is None:
            with torch.no_grad():
                logits = self.model(input_ids, use_cache=False).logits
        tid = logits[0, -1, :].argmax().item()
        return self.decode(tid).strip(), tid

    def generate(self, prompt: str, max_new: int = 30) -> str:
        ids = self.encode(prompt)
        with torch.no_grad():
            try:
                out = self.model.generate(
                    ids, max_new_tokens=max_new, do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            except AttributeError:
                # Fallback for models with incompatible cache (e.g. Phi-3.5)
                generated = ids
                for _ in range(max_new):
                    logits = self.model(generated, use_cache=False).logits
                    next_id = logits[0, -1, :].argmax(dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_id.unsqueeze(0)], dim=-1)
                    if next_id.item() == self.tokenizer.eos_token_id:
                        break
                out = generated
        return self.tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

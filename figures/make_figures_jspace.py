#!/usr/bin/env python3
"""Publication figures for the J-Space fingerprinting AAAI paper.

Outputs paper/figures/fig_jspace_*.{pdf,png}
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, FancyBboxPatch

ROOT = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIGDIR, exist_ok=True)

BLUE = "#3e6ba6"
ORANGE = "#c0392b"
INK, INK2, MUTED = "#2b2f36", "#525a64", "#8a9098"
GRID, BASE = "#e4e7ec", "#c6ccd4"
SEQ = ["#eef3fa", "#dfe7f2", "#c9d7ea", "#a8c0de", "#82a5cf", "#5d88bd", "#3e6ba6", "#2b5188"]
CMAP = LinearSegmentedColormap.from_list("seqblue", SEQ)
ORDINAL = ["#c9d7ea", "#a8c0de", "#82a5cf", "#5d88bd", "#3e6ba6", "#2b5188"]
GREEN = "#2f9e63"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.edgecolor": BASE, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK2, "text.color": INK,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

FA = json.load(open(os.path.join(ROOT, "results", "final_analysis.json")))
FAM = FA["families"]
MODELS = FA["models"]
MJ = FA["depth_final"]["matrix_j20"]
def J(a, b): return MJ[f"{a}|{b}"]

def save(fig, name):
    fig.savefig(os.path.join(FIGDIR, f"{name}.pdf"))
    fig.savefig(os.path.join(FIGDIR, f"{name}.png"), dpi=300)
    plt.close(fig)
    print(f"  {name} saved")

def style_ax(ax, grid_axis="y"):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.5, zorder=0)
    ax.tick_params(length=2.5)


# ═══════════════════════════════════════════════════════════════════
# FIG 1 — Calibration ladder (raw→corrected dumbbell)
# ═══════════════════════════════════════════════════════════════════
def fig_ladder():
    # Data from robustness_suite.json / final_analysis.json
    # (raw, excess) pairs per category
    cats = [
        ("Identical data\n(no shared weights)", 0.48, 0.35),
        ("Fine-tuned from base", 0.40, 0.35),
        ("Same family,\ndifferent size", 0.36, 0.26),
        ("Same developer,\nnext generation", 0.36, 0.25),
        ("Unrelated", 0.22, 0.17),
    ]
    colors = [ORDINAL[5], ORDINAL[4], ORDINAL[2], ORDINAL[1], ORDINAL[0]]

    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    style_ax(ax, grid_axis="x")
    for yi, (label, raw, exc) in enumerate(cats):
        y = len(cats) - 1 - yi
        col = colors[yi]
        # dumbbell: line from excess to raw, dots on both
        ax.plot([exc, raw], [y, y], color=col, lw=2.0, zorder=3, solid_capstyle="round")
        ax.scatter([raw], [y], s=28, color=col, zorder=4, edgecolors="white", linewidths=0.5)
        ax.scatter([exc], [y], s=28, color=col, zorder=4, marker="D", edgecolors="white", linewidths=0.5)
        # labels
        ax.text(raw + 0.008, y + 0.15, f"{raw:.2f}", ha="left", fontsize=6, color=MUTED)
        if exc != raw:
            ax.text(exc - 0.008, y + 0.15, f"{exc:.2f}", ha="right", fontsize=6, color=col, fontweight="bold")
        else:
            ax.text(raw + 0.008, y - 0.25, f"({exc:.2f} corrected)", ha="left", fontsize=5.5, color=col)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(reversed([c[0] for c in cats]), fontsize=6.2, color=INK2)
    ax.set_xlabel("Fingerprint similarity (Jaccard@20, near-final depth)")
    ax.set_xlim(0.08, 0.56)
    ax.set_ylim(-0.5, len(cats) - 0.3)
    # legend
    ax.scatter([], [], s=20, color=MUTED, label="Raw", edgecolors="white", linewidths=0.3)
    ax.scatter([], [], s=20, color=MUTED, marker="D", label="Corrected", edgecolors="white", linewidths=0.3)
    ax.legend(fontsize=5.5, loc="lower right", framealpha=0.9)
    save(fig, "fig_jspace_ladder")


# ═══════════════════════════════════════════════════════════════════
# FIG 2 — 30×30 similarity heatmap (no in-cell numbers; clean)
# ═══════════════════════════════════════════════════════════════════
def fig_heatmap():
    fam_order = ["Qwen3", "Qwen2.5", "Q25-Math", "DS-Qwen", "DS-Llama",
                 "Llama", "Mistral", "Phi", "InternLM", "Gemma", "Pythia"]
    FAMC = {"Qwen3": "#2b5ea7", "Qwen2.5": "#5b8ec9", "Q25-Math": "#5b8ec9",
            "DS-Qwen": "#e8912d", "DS-Llama": "#e8912d", "Llama": "#c23b3b",
            "Mistral": "#8a5fbf", "Phi": "#2f9e63", "InternLM": "#c65911",
            "Gemma": "#1d9e89", "Pythia": "#8b8b88"}
    import re
    def psize(m):
        mm = re.search(r"(\d+(?:\.\d+)?)[bB]", m)
        return float(mm.group(1)) if mm else (14 if "phi-4" in m else 3.8)
    models = sorted(MODELS, key=lambda m: (fam_order.index(FAM[m]), psize(m)))
    n = len(models)
    M = np.array([[J(a, b) if a != b else np.nan for b in models] for a in models])

    short = {}
    for m in models:
        s = (m.replace("-Instruct", "").replace("-v0.3", "").replace("-2410", "")
              .replace("DeepSeek-R1-Distill-Qwen-", "DS-Q-").replace("DeepSeek-R1-Distill-Llama-", "DS-L-")
              .replace("internlm2_5-7b-chat", "ILM-7B")
              .replace("gemma-2-", "Gem-").replace("-it", "").replace("pythia-", "Py-")
              .replace("Phi-3.5-mini-instruct", "Phi3.5").replace("phi-4", "Phi-4")
              .replace("Mistral-7B-Instruct", "Mis-7B").replace("Ministral-8B", "Min-8B")
              .replace("Llama-3.2-", "L3.2-").replace("Llama-3.1-", "L3.1-")
              .replace("Qwen2.5-", "Q2.5-").replace("Qwen3-", "Q3-"))
        short[m] = s

    RAMP = ["#eaf0f8", "#dfe7f2", "#cfdcee", "#bcd0e7", "#a8c0de", "#8fb0d6",
            "#74a0cc", "#5d88bd", "#4a77b0", "#3e6ba6"]
    REDS = ["#a63c50", "#b03246", "#c0392b"]
    DIAG = "#eceae5"
    LO, HI = 0.10, 0.42

    def cell_color(v):
        if np.isnan(v): return DIAG
        if v >= HI:
            t = min((v - HI) / 0.16, 1.0)
            return REDS[min(int(t * 3), 2)]
        t = min(max(v - LO, 0.0) / (HI - LO), 1.0)
        return RAMP[min(int(t * len(RAMP)), len(RAMP) - 1)]

    cell = 0.185
    fig, ax = plt.subplots(figsize=(1.35 + n * cell, 1.15 + n * cell))
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(n - 0.5, -0.5)
    ax.set_facecolor("white")
    for ri in range(n):
        for ci in range(n):
            v = M[ri, ci]
            ax.add_patch(Rectangle((ci - 0.46, ri - 0.46), 0.92, 0.92,
                                   facecolor=cell_color(v), edgecolor="none"))

    bounds, start = [], 0
    for i in range(1, n + 1):
        if i == n or FAM[models[i]] != FAM[models[start]]:
            bounds.append((start, i, FAM[models[start]])); start = i
    for s0, s1, fam in bounds:
        if s1 - s0 >= 2:
            ax.add_patch(Rectangle((s0 - 0.48, s0 - 0.48), (s1 - s0) - 0.04, (s1 - s0) - 0.04,
                                   fill=False, edgecolor=FAMC[fam], linewidth=0.9, zorder=5))

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([short[m] for m in models], rotation=45, ha="left", fontsize=4.6)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticklabels([short[m] for m in models], fontsize=4.6)
    for tick, m in zip(ax.get_xticklabels(), models):
        tick.set_color(FAMC[FAM[m]]); tick.set_fontweight("bold")
    for tick, m in zip(ax.get_yticklabels(), models):
        tick.set_color(FAMC[FAM[m]]); tick.set_fontweight("bold")
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)

    import matplotlib.transforms as mtr
    lax = fig.add_axes([0.30, 0.055, 0.45, 0.016])
    steps = RAMP + REDS
    edges = list(np.linspace(LO, HI, len(RAMP) + 1)) + [0.47, 0.52, 0.60]
    for i, c in enumerate(steps):
        lax.add_patch(Rectangle((i, 0), 0.94, 1, facecolor=c, edgecolor="none"))
    lax.set_xlim(0, len(steps)); lax.set_ylim(0, 1)
    for i in range(0, len(RAMP) + 1, 2):
        lax.text(i, -0.7, f"{edges[i]:.2f}", ha="center", va="top", fontsize=4.4, color=MUTED)
    lax.text(len(RAMP) + 1.5, -0.7, "\u2265 0.42", ha="center", va="top", fontsize=4.4, color="#b03246")
    lax.text(-0.6, 0.5, "Jaccard@20", ha="right", va="center", fontsize=5.0, color=INK2)
    lax.axis("off")
    save(fig, "fig_jspace_matrix")


# ═══════════════════════════════════════════════════════════════════
# FIG 3 — Depth ablation (30 models) with k=5 and dip annotated
# ═══════════════════════════════════════════════════════════════════
def fig_depth():
    DA30 = json.load(open(os.path.join(ROOT, "results", "depth_ablation_30.json")))
    depth_keys = ["L/4", "L/2", "3L/4", "L-2", "output"]
    xlab = ["¼", "½", "¾", "last−2", "output"]
    auc = [DA30["depths"][d]["auc"] for d in depth_keys]
    dd = [DA30["depths"][d]["cohens_d"] for d in depth_keys]

    # k=5 AUC at output from api_truncation.json
    api = json.load(open(os.path.join(ROOT, "results", "api_truncation.json")))
    auc_k5 = api["k=5"]["auc"]

    fig, axes = plt.subplots(2, 1, figsize=(3.3, 2.6), sharex=True,
                             gridspec_kw={"hspace": 0.18})
    for ax, vals, lab, ylim in [(axes[0], auc, "Family ROC AUC", (0.58, 0.95)),
                                 (axes[1], dd, "Cohen's d", (0.3, 2.5))]:
        style_ax(ax)
        ax.plot(range(5), vals, color=BLUE, lw=1.4, zorder=3)
        ax.scatter(range(4), vals[:4], s=16, color=BLUE, zorder=4)
        ax.scatter([4], [vals[4]], s=22, color=ORANGE, zorder=5)
        ax.set_ylabel(lab)
        ax.set_ylim(*ylim)
        for i, v in enumerate(vals):
            off = (ylim[1] - ylim[0]) * 0.06
            ax.text(i, v + off, f"{v:.2f}", ha="center", fontsize=6, color=INK)

    # annotate dip on AUC panel
    axes[0].annotate("dip", xy=(1, auc[1]), xytext=(1.5, auc[1] - 0.03),
                     fontsize=5.5, color=MUTED, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
    # annotate k=5 on AUC panel
    axes[0].scatter([4], [auc_k5], s=14, color=GREEN, zorder=6, marker="^")
    axes[0].text(4.15, auc_k5 - 0.01, f"k=5: {auc_k5:.2f}", fontsize=5.5, color=GREEN, va="top")

    axes[1].set_xticks(range(5))
    axes[1].set_xticklabels(xlab)
    axes[1].set_xlabel("Readout depth (all 30 models)")
    save(fig, "fig_jspace_depth")


# ═══════════════════════════════════════════════════════════════════
# FIG 4 — Crystallization: Qwen3 + Gemma + Pythia (3 panels)
# ═══════════════════════════════════════════════════════════════════
def fig_cryst():
    C = FA["crystallization"]
    qwen3 = [("Qwen3-0.6B", "0.6B"), ("Qwen3-1.7B", "1.7B"), ("Qwen3-4B", "4B"),
             ("Qwen3-8B", "8B"), ("Qwen3-14B", "14B"), ("Qwen3-32B", "32B")]
    gemma = [("gemma-2-2b-it", "2B"), ("gemma-2-9b-it", "9B"), ("gemma-2-27b-it", "27B")]
    pythia = [("pythia-1.4b", "1.4B"), ("pythia-6.9b", "6.9B"), ("pythia-12b", "12B")]

    fig, axes = plt.subplots(1, 3, figsize=(5.0, 1.8), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    panels = [
        (axes[0], qwen3, "Qwen3 (slopes)", ORDINAL),
        (axes[1], gemma, "Gemma-2 (slopes)", [ORDINAL[1], ORDINAL[3], ORDINAL[5]]),
        (axes[2], pythia, "Pythia (flat — control)", [ORDINAL[1], ORDINAL[3], ORDINAL[5]]),
    ]
    for ax, series, title, ramp in panels:
        style_ax(ax)
        show = {"0.6B": (0, (-2, 5)), "32B": (2, (5, -8)),
                "2B": (0, (-2, 5)), "27B": (2, (5, -8)),
                "1.4B": (0, (-2, 5)), "12B": (2, (5, -3))}
        for (m, lab), col in zip(series, ramp):
            pts = C[m]["self_overlap_by_depth"] + [[1.0, 1.0]]
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=col, lw=1.2, marker="o", ms=2.6, zorder=3)
            if lab in show:
                pi, off = show[lab]
                ax.annotate(lab, (xs[pi], ys[pi]), textcoords="offset points",
                            xytext=off, fontsize=5.8, color=col, fontweight="bold")
        ax.axvline(0.75, color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
        ax.set_xlim(0.18, 1.03)
        ax.set_ylim(-0.04, 1.05)
        ax.set_title(title, fontsize=6.5, color=INK2)
        ax.set_xlabel("Readout depth")
    axes[0].set_ylabel("Overlap with final top-20")
    save(fig, "fig_jspace_cryst")


# ═══════════════════════════════════════════════════════════════════
# FIG 5 — Training trajectory (log x, single y, annotated gap)
# ═══════════════════════════════════════════════════════════════════
def fig_trajectory():
    import sys
    sys.path.insert(0, ROOT)
    from capability_gold import compute_capability

    TJ = json.load(open(os.path.join(ROOT, "results", "e0b_trajectory.json")))
    steps = sorted(int(s) for s in TJ["steps"].keys())
    pair_exc = [TJ["steps"][str(s)]["pair_excess"] for s in steps]
    qwen_exc = [np.mean([TJ["steps"][str(s)].get("ref_qwen14_excess", 0),
                         TJ["steps"][str(s)].get("ref_qwen69_excess", 0)]) for s in steps]
    # capability curve
    cap = []
    for step in steps:
        rates = []
        for size in ["pythia-1.4b", "pythia-6.9b"]:
            if step == 143000:
                path = os.path.join(ROOT, "results", "fingerprints_v2", f"{size}_fp.json")
            else:
                path = os.path.join(ROOT, "results", "checkpoint_fps", f"{size}@step{step}_fp.json")
            if os.path.exists(path):
                fp = json.load(open(path))
                _, _, rate = compute_capability(fp)
                rates.append(rate)
        cap.append(np.mean(rates) if rates else 0)

    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    style_ax(ax)

    # all on same y-axis (all values are 0-0.5 range)
    ax.semilogx(steps, pair_exc, color=BLUE, lw=1.6, marker="o", ms=4, zorder=4,
                label="Same-data excess (1.4B ↔ 6.9B)")
    ax.semilogx(steps, cap, color=ORANGE, lw=1.4, marker="D", ms=3.5, zorder=5, alpha=0.85,
                label="Probe accuracy (mean of both)")
    ax.semilogx(steps, qwen_exc, color=MUTED, lw=1.0, marker="s", ms=2.5, zorder=3, ls="--",
                label="Unrelated excess (vs. Qwen2.5-7B)")

    # shade the "signal present, capability absent" window
    ax.axvspan(steps[0], 4000, alpha=0.06, color=BLUE, zorder=1)
    ax.annotate("signal present,\ncapability absent", xy=(1500, 0.33), fontsize=5.5,
                color=BLUE, alpha=0.6, ha="center")

    # annotate step 1000
    ax.annotate(f"step 1k:\nexcess 0.25\naccuracy 0.03", xy=(1000, 0.25),
                xytext=(3000, 0.08), fontsize=5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))

    ax.set_xlabel("Training step (log scale)")
    ax.set_ylabel("Score")
    ax.set_xlim(700, 200000)
    ax.set_ylim(-0.02, 0.50)
    ax.legend(fontsize=5.2, loc="center right", framealpha=0.9)
    save(fig, "fig_jspace_trajectory")


# ═══════════════════════════════════════════════════════════════════
# FIG 6 (new) — Identity gap: same-model variants vs cross-model
# ═══════════════════════════════════════════════════════════════════
def fig_identity_gap():
    items = [
        ("fp32 ↔ bf16", 0.98, BLUE),
        ("bf16 ↔ int8", 0.92, BLUE),
        ("bf16 ↔ int4", 0.85, BLUE),
        ("base ↔ instruct\n(same lineage)", 0.76, BLUE),
    ]
    cross = [
        ("Strongest cross-model\n(R1-Q-32B ↔ Q2.5-32B)", 0.58, ORANGE),
    ]
    all_items = items + cross

    fig, ax = plt.subplots(figsize=(3.3, 1.6))
    style_ax(ax, grid_axis="x")
    for i, (label, val, col) in enumerate(reversed(all_items)):
        y = i
        ax.barh(y, val, height=0.55, color=col, alpha=0.75, zorder=3)
        ax.text(val + 0.01, y, f"{val:.2f}", va="center", fontsize=6.5, color=INK, fontweight="bold")
    ax.set_yticks(range(len(all_items)))
    ax.set_yticklabels([x[0] for x in reversed(all_items)], fontsize=6)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Jaccard@20 self-similarity")
    # gap annotation
    ax.axvspan(0.58, 0.76, alpha=0.08, color=ORANGE, zorder=1)
    ax.text(0.67, 2.5, "identity\ngap", ha="center", fontsize=5.5, color=ORANGE, alpha=0.7)
    save(fig, "fig_jspace_identity_gap")


# ═══════════════════════════════════════════════════════════════════
# FIG 7 (new) — Open-set margin collapse
# ═══════════════════════════════════════════════════════════════════
def fig_openset_margin():
    OA = json.load(open(os.path.join(ROOT, "results", "openset_ancestry.json")))
    queries = list(OA["cases"].keys())
    short = {"DeepSeek-R1-Distill-Qwen-32B": "R1-Q-32B",
             "DeepSeek-R1-Distill-Qwen-14B": "R1-Q-14B",
             "DeepSeek-R1-Distill-Qwen-7B": "R1-Q-7B",
             "DeepSeek-R1-Distill-Qwen-1.5B": "R1-Q-1.5B",
             "DeepSeek-R1-Distill-Llama-8B": "R1-L-8B"}

    fig, ax = plt.subplots(figsize=(3.3, 1.8))
    style_ax(ax, grid_axis="y")

    ys = np.arange(len(queries))
    closed_m = [OA["cases"][q]["closed"]["margin"] for q in queries]
    openb_m = [OA["cases"][q]["open-b"]["margin"] for q in queries]

    ax.scatter(closed_m, ys + 0.12, s=30, color=BLUE, zorder=4, label="Base in pool (closed)")
    ax.scatter(openb_m, ys - 0.12, s=30, color=ORANGE, marker="x", zorder=4, linewidths=1.5,
               label="Base removed (open)")
    # connect pairs
    for i in range(len(queries)):
        ax.plot([closed_m[i], openb_m[i]], [ys[i] + 0.12, ys[i] - 0.12],
                color=MUTED, lw=0.6, zorder=2)

    ax.axvline(1.1, color=MUTED, lw=1.0, ls="--", zorder=1)
    ax.text(1.105, len(queries) - 0.5, "threshold\n1.1×", fontsize=5.5, color=MUTED, va="top")

    ax.set_yticks(ys)
    ax.set_yticklabels([short[q] for q in queries], fontsize=6.5)
    ax.set_xlabel("Top-1 / Top-2 margin")
    ax.set_xlim(0.95, 1.4)
    ax.legend(fontsize=5.5, loc="upper right", framealpha=0.9)
    save(fig, "fig_jspace_openset")


# ═══════════════════════════════════════════════════════════════════
# FIG 8 (new) — Witness forest plot
# ═══════════════════════════════════════════════════════════════════
def fig_witness_forest():
    WIT = json.load(open(os.path.join(ROOT, "results", "e0a_witness_output.json")))
    groups = [
        ("Pythia internal", WIT["witness_groups"]["Pythia internal"]),
        ("NeoX ↔ Pythia\n(diff project)", WIT["witness_groups"]["NeoX-Pythia"]),
        ("RWKV ↔ Pythia\n(diff architecture)", WIT["witness_groups"]["RWKV-Pythia"]),
        ("Cerebras ↔ Pythia\n(diff tokenizer + org)", WIT["witness_groups"]["Cerebras-Pythia"]),
        ("Unrelated refs", WIT["witness_groups"]["unrelated"]),
    ]

    fig, ax = plt.subplots(figsize=(3.3, 1.8))
    style_ax(ax, grid_axis="x")
    ys = np.arange(len(groups))
    for i, (label, data) in enumerate(reversed(groups)):
        y = i
        mean = data["mean_excess"]
        pairs = data.get("pairs", [mean])
        lo, hi = min(pairs), max(pairs)
        col = ORANGE if "Unrelated" in label else BLUE
        ax.plot([lo, hi], [y, y], color=col, lw=1.5, zorder=2, solid_capstyle="round")
        ax.scatter([mean], [y], s=40, color=col, zorder=4, edgecolors="white", linewidths=0.5)
        ax.text(mean + 0.01, y + 0.2, f"{mean:.2f}", fontsize=6, color=col, fontweight="bold")

    # unrelated reference line
    unrel = WIT["witness_groups"]["unrelated"]["mean_excess"]
    ax.axvline(unrel, color=ORANGE, lw=0.8, ls=":", zorder=1, alpha=0.6)

    ax.set_yticks(ys)
    ax.set_yticklabels([g[0] for g in reversed(groups)], fontsize=6)
    ax.set_xlabel("Excess similarity (output level)")
    ax.set_xlim(-0.02, 0.55)
    save(fig, "fig_jspace_witness")


if __name__ == "__main__":
    print("Generating figures...")
    fig_ladder()
    fig_heatmap()
    fig_depth()
    fig_cryst()
    fig_trajectory()
    fig_identity_gap()
    fig_openset_margin()
    fig_witness_forest()
    print("Done.")

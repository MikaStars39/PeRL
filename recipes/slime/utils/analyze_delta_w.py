#!/usr/bin/env python3
"""
Analyze delta weights (Δw) between pre-RL and post-RL model checkpoints.

Metrics computed:
  1. L2 norm per layer (absolute & relative)
  2. Cosine similarity per layer
  3. Low-rank analysis via SVD (effective rank, top singular values)
  4. Module-group aggregation (attn.q/k/v/o, mlp.gate/up/down, embed, norm, lm_head)
  5. Spectral density of the update (eigenvalue distribution of ΔwᵀΔw)

Usage:
    python analyze_delta_w.py \
        --before /path/to/before_rl_checkpoint \
        --after  /path/to/after_rl_checkpoint \
        --output ./delta_w_analysis \
        --top-k-svd 64 \
        --spectral-bins 200
"""

import argparse
import os
import json
import gc
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_state_dict(ckpt_path: str) -> dict[str, torch.Tensor]:
    """Load a HuggingFace checkpoint (safetensors or bin) into a flat state dict."""
    ckpt_path = Path(ckpt_path)
    state = {}

    # safetensors shards
    st_files = sorted(ckpt_path.glob("*.safetensors"))
    if st_files:
        from safetensors.torch import load_file
        for f in st_files:
            state.update(load_file(str(f), device="cpu"))
        return state

    # pytorch bin shards
    bin_files = sorted(ckpt_path.glob("*.bin"))
    if bin_files:
        for f in bin_files:
            state.update(torch.load(str(f), map_location="cpu", weights_only=True))
        return state

    raise FileNotFoundError(
        f"No .safetensors or .bin files found in {ckpt_path}"
    )


# ---------------------------------------------------------------------------
# Module group classification
# ---------------------------------------------------------------------------

MODULE_GROUPS = {
    "attn.q_proj": ["q_proj"],
    "attn.k_proj": ["k_proj"],
    "attn.v_proj": ["v_proj"],
    "attn.o_proj": ["o_proj"],
    "mlp.gate_proj": ["gate_proj", "gate_up_proj"],  # some models fuse gate+up
    "mlp.up_proj": ["up_proj"],
    "mlp.down_proj": ["down_proj"],
    "embed": ["embed_tokens", "wte", "wpe"],
    "lm_head": ["lm_head"],
    "norm": ["layernorm", "rmsnorm", "layer_norm", "input_layernorm",
             "post_attention_layernorm", "norm"],
}


def classify_param(name: str) -> str:
    name_lower = name.lower()
    for group, keywords in MODULE_GROUPS.items():
        for kw in keywords:
            if kw in name_lower:
                return group
    return "other"


def extract_layer_idx(name: str) -> int | None:
    """Extract numeric layer index from parameter name (e.g. model.layers.12.xxx -> 12)."""
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p in ("layers", "h", "block"):
            if i + 1 < len(parts) and parts[i + 1].isdigit():
                return int(parts[i + 1])
    return None


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze_pair(
    name: str,
    w_before: torch.Tensor,
    w_after: torch.Tensor,
    top_k_svd: int,
) -> dict:
    """Compute all per-parameter metrics."""
    delta = (w_after.float() - w_before.float())
    w_b = w_before.float()

    result = {"name": name, "shape": list(delta.shape), "numel": delta.numel()}

    # L2 norms
    l2_delta = delta.norm().item()
    l2_before = w_b.norm().item()
    result["l2_delta"] = l2_delta
    result["l2_before"] = l2_before
    result["l2_relative"] = l2_delta / max(l2_before, 1e-12)

    # Cosine similarity (flatten)
    cos = torch.nn.functional.cosine_similarity(
        w_b.reshape(1, -1), w_after.float().reshape(1, -1)
    ).item()
    result["cosine_sim"] = cos

    # SVD analysis (only for 2-D weight matrices)
    if delta.dim() == 2 and min(delta.shape) > 1:
        k = min(top_k_svd, min(delta.shape))
        try:
            S = torch.linalg.svdvals(delta)
            result["singular_values"] = S[:k].tolist()
            result["sv_total"] = S.sum().item()
            result["sv_top1_ratio"] = (S[0] / S.sum()).item() if S.sum() > 0 else 0.0

            # effective rank: exp(entropy of normalized singular values)
            S_norm = S / S.sum()
            S_norm = S_norm[S_norm > 1e-12]
            entropy = -(S_norm * S_norm.log()).sum().item()
            result["effective_rank"] = np.exp(entropy)
            result["matrix_rank"] = int((S > S[0] * 1e-5).sum().item())

            # spectral density: squared singular values ~ eigenvalues of ΔwᵀΔw
            result["spectral_values"] = (S ** 2).tolist()
        except Exception:
            pass  # very large matrices may OOM on full SVD

    del delta, w_b
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_all(records: list[dict], output_dir: Path, spectral_bins: int):
    """Generate all analysis plots and save to a single PDF + individual PNGs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "delta_w_report.pdf"

    # Filter 2-D weight matrices for most plots
    mat_records = [r for r in records if len(r["shape"]) == 2 and "singular_values" in r]

    with PdfPages(str(pdf_path)) as pdf:
        # ── 1. L2 norm per layer (bar chart, all params) ──
        fig, axes = plt.subplots(2, 1, figsize=(18, 10))
        names = [r["name"] for r in records]
        l2s = [r["l2_delta"] for r in records]
        rels = [r["l2_relative"] for r in records]
        x = range(len(names))
        axes[0].bar(x, l2s, width=0.8)
        axes[0].set_ylabel("L2(Δw)")
        axes[0].set_title("Absolute L2 Norm of Δw per Parameter")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(names, rotation=90, fontsize=4)
        axes[1].bar(x, rels, width=0.8, color="orange")
        axes[1].set_ylabel("L2(Δw) / L2(w)")
        axes[1].set_title("Relative L2 Norm of Δw per Parameter")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(names, rotation=90, fontsize=4)
        fig.tight_layout()
        pdf.savefig(fig); plt.savefig(output_dir / "01_l2_norm.png", dpi=150); plt.close(fig)

        # ── 2. L2 norm by layer index (line chart) ──
        layer_l2 = defaultdict(float)
        layer_rel = defaultdict(float)
        layer_count = defaultdict(int)
        for r in records:
            idx = extract_layer_idx(r["name"])
            if idx is not None:
                layer_l2[idx] += r["l2_delta"]
                layer_rel[idx] += r["l2_relative"]
                layer_count[idx] += 1
        if layer_l2:
            idxs = sorted(layer_l2.keys())
            fig, axes = plt.subplots(2, 1, figsize=(14, 8))
            axes[0].plot(idxs, [layer_l2[i] for i in idxs], "o-")
            axes[0].set_xlabel("Layer index")
            axes[0].set_ylabel("Sum L2(Δw)")
            axes[0].set_title("Total L2(Δw) per Transformer Layer")
            axes[0].grid(True, alpha=0.3)
            axes[1].plot(idxs, [layer_rel[i] / layer_count[i] for i in idxs], "o-", color="orange")
            axes[1].set_xlabel("Layer index")
            axes[1].set_ylabel("Mean relative L2")
            axes[1].set_title("Mean Relative L2(Δw) per Transformer Layer")
            axes[1].grid(True, alpha=0.3)
            fig.tight_layout()
            pdf.savefig(fig); plt.savefig(output_dir / "02_l2_by_layer.png", dpi=150); plt.close(fig)

        # ── 3. Cosine similarity per layer ──
        layer_cos = defaultdict(list)
        for r in records:
            idx = extract_layer_idx(r["name"])
            if idx is not None:
                layer_cos[idx].append(r["cosine_sim"])
        if layer_cos:
            idxs = sorted(layer_cos.keys())
            mean_cos = [np.mean(layer_cos[i]) for i in idxs]
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(idxs, mean_cos, "s-", color="green")
            ax.set_xlabel("Layer index")
            ax.set_ylabel("Cosine similarity (w_before, w_after)")
            ax.set_title("Mean Cosine Similarity per Layer (closer to 1 = less directional change)")
            ax.set_ylim(min(min(mean_cos) - 0.01, 0.99), 1.001)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            pdf.savefig(fig); plt.savefig(output_dir / "03_cosine_sim.png", dpi=150); plt.close(fig)

        # ── 4. Module-group aggregation ──
        group_l2 = defaultdict(float)
        group_numel = defaultdict(int)
        group_rel = defaultdict(list)
        group_cos = defaultdict(list)
        group_effrank = defaultdict(list)
        for r in records:
            g = classify_param(r["name"])
            group_l2[g] += r["l2_delta"] ** 2  # sum of squares for proper aggregation
            group_numel[g] += r["numel"]
            group_rel[g].append(r["l2_relative"])
            group_cos[g].append(r["cosine_sim"])
            if "effective_rank" in r:
                group_effrank[g].append(r["effective_rank"])

        if group_l2:
            groups = sorted(group_l2.keys())
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

            # 4a: total L2 per group
            ax = axes[0, 0]
            vals = [np.sqrt(group_l2[g]) for g in groups]
            ax.barh(groups, vals)
            ax.set_xlabel("L2(Δw)")
            ax.set_title("Total L2(Δw) by Module Group")

            # 4b: mean relative L2 per group
            ax = axes[0, 1]
            vals = [np.mean(group_rel[g]) for g in groups]
            ax.barh(groups, vals, color="orange")
            ax.set_xlabel("Mean relative L2")
            ax.set_title("Mean Relative L2 by Module Group")

            # 4c: mean cosine sim per group
            ax = axes[1, 0]
            vals = [np.mean(group_cos[g]) for g in groups]
            ax.barh(groups, vals, color="green")
            ax.set_xlabel("Cosine similarity")
            ax.set_title("Mean Cosine Similarity by Module Group")

            # 4d: mean effective rank per group
            ax = axes[1, 1]
            g_with_rank = [g for g in groups if group_effrank[g]]
            vals = [np.mean(group_effrank[g]) for g in g_with_rank]
            ax.barh(g_with_rank, vals, color="purple")
            ax.set_xlabel("Effective rank")
            ax.set_title("Mean Effective Rank(Δw) by Module Group")

            fig.tight_layout()
            pdf.savefig(fig); plt.savefig(output_dir / "04_module_groups.png", dpi=150); plt.close(fig)

        # ── 5. SVD / Low-rank analysis ──
        if mat_records:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

            # 5a: effective rank across layers
            names_m = [r["name"] for r in mat_records]
            eff_ranks = [r.get("effective_rank", 0) for r in mat_records]
            mat_ranks = [r.get("matrix_rank", 0) for r in mat_records]
            ax = axes[0, 0]
            x = range(len(names_m))
            ax.bar(x, eff_ranks, alpha=0.7, label="Effective rank")
            ax.set_ylabel("Effective rank")
            ax.set_title("Effective Rank of Δw (lower = more low-rank)")
            ax.set_xticks(x)
            ax.set_xticklabels(names_m, rotation=90, fontsize=3)
            ax.legend()

            # 5b: top-1 SV ratio (concentration)
            top1_ratios = [r.get("sv_top1_ratio", 0) for r in mat_records]
            ax = axes[0, 1]
            ax.bar(x, top1_ratios, color="red", alpha=0.7)
            ax.set_ylabel("σ₁ / Σσᵢ")
            ax.set_title("Top-1 Singular Value Concentration")
            ax.set_xticks(x)
            ax.set_xticklabels(names_m, rotation=90, fontsize=3)

            # 5c: singular value curves for a few representative layers
            ax = axes[1, 0]
            # pick layers at 0%, 25%, 50%, 75%, 100% positions
            sample_idxs = np.linspace(0, len(mat_records) - 1, min(5, len(mat_records)), dtype=int)
            for i in sample_idxs:
                sv = mat_records[i].get("singular_values", [])
                if sv:
                    ax.plot(sv, label=mat_records[i]["name"].split(".")[-2] + "." + mat_records[i]["name"].split(".")[-1], alpha=0.8)
            ax.set_xlabel("Singular value index")
            ax.set_ylabel("σᵢ")
            ax.set_title("Singular Value Decay (sampled layers)")
            ax.set_yscale("log")
            ax.legend(fontsize=6)
            ax.grid(True, alpha=0.3)

            # 5d: effective rank by layer index
            ax = axes[1, 1]
            layer_effrank = defaultdict(list)
            for r in mat_records:
                idx = extract_layer_idx(r["name"])
                if idx is not None and "effective_rank" in r:
                    layer_effrank[idx].append(r["effective_rank"])
            if layer_effrank:
                idxs = sorted(layer_effrank.keys())
                ax.plot(idxs, [np.mean(layer_effrank[i]) for i in idxs], "o-", color="purple")
                ax.set_xlabel("Layer index")
                ax.set_ylabel("Mean effective rank")
                ax.set_title("Mean Effective Rank per Layer")
                ax.grid(True, alpha=0.3)

            fig.tight_layout()
            pdf.savefig(fig); plt.savefig(output_dir / "05_svd_analysis.png", dpi=150); plt.close(fig)

        # ── 6. Spectral density (eigenvalue distribution of ΔwᵀΔw) ──
        all_spectral = []
        for r in mat_records:
            if "spectral_values" in r:
                all_spectral.extend(r["spectral_values"])
        if all_spectral:
            all_spectral = np.array(all_spectral)
            all_spectral = all_spectral[all_spectral > 1e-20]  # filter near-zero

            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

            # 6a: global spectral density (log scale)
            ax = axes[0, 0]
            log_spec = np.log10(all_spectral)
            ax.hist(log_spec, bins=spectral_bins, density=True, alpha=0.7, edgecolor="black", linewidth=0.3)
            ax.set_xlabel("log₁₀(σ²)")
            ax.set_ylabel("Density")
            ax.set_title("Global Spectral Density of Δw (log scale)")
            ax.grid(True, alpha=0.3)

            # 6b: global spectral density (linear scale, clipped to 95th percentile)
            ax = axes[0, 1]
            clip = np.percentile(all_spectral, 95)
            ax.hist(all_spectral[all_spectral <= clip], bins=spectral_bins, density=True,
                    alpha=0.7, color="orange", edgecolor="black", linewidth=0.3)
            ax.set_xlabel("σ²")
            ax.set_ylabel("Density")
            ax.set_title("Spectral Density (linear, ≤ 95th percentile)")
            ax.grid(True, alpha=0.3)

            # 6c: per-group spectral density
            ax = axes[1, 0]
            group_spectral = defaultdict(list)
            for r in mat_records:
                if "spectral_values" in r:
                    g = classify_param(r["name"])
                    group_spectral[g].extend(r["spectral_values"])
            for g in sorted(group_spectral.keys()):
                vals = np.array(group_spectral[g])
                vals = vals[vals > 1e-20]
                if len(vals) > 0:
                    ax.hist(np.log10(vals), bins=spectral_bins // 2, density=True,
                            alpha=0.5, label=g)
            ax.set_xlabel("log₁₀(σ²)")
            ax.set_ylabel("Density")
            ax.set_title("Spectral Density by Module Group")
            ax.legend(fontsize=6)
            ax.grid(True, alpha=0.3)

            # 6d: cumulative spectral energy
            ax = axes[1, 1]
            sorted_spec = np.sort(all_spectral)[::-1]
            cumulative = np.cumsum(sorted_spec) / sorted_spec.sum()
            ax.plot(np.arange(1, len(cumulative) + 1), cumulative)
            ax.set_xlabel("Number of components (sorted by σ²)")
            ax.set_ylabel("Cumulative energy fraction")
            ax.set_title("Cumulative Spectral Energy")
            ax.axhline(0.9, color="red", linestyle="--", alpha=0.5, label="90% energy")
            ax.axhline(0.95, color="orange", linestyle="--", alpha=0.5, label="95% energy")
            ax.axhline(0.99, color="green", linestyle="--", alpha=0.5, label="99% energy")
            # find and annotate the rank needed for each threshold
            for thresh, color in [(0.9, "red"), (0.95, "orange"), (0.99, "green")]:
                rank_needed = int(np.searchsorted(cumulative, thresh)) + 1
                ax.axvline(rank_needed, color=color, linestyle=":", alpha=0.3)
                ax.annotate(f"rank={rank_needed}", (rank_needed, thresh),
                            fontsize=8, color=color)
            ax.set_xscale("log")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

            fig.tight_layout()
            pdf.savefig(fig); plt.savefig(output_dir / "06_spectral_density.png", dpi=150); plt.close(fig)

        # ── 7. Heatmap: relative L2 by (layer_idx, module_group) ──
        heatmap_data = defaultdict(dict)
        all_groups_set = set()
        all_layers_set = set()
        for r in records:
            idx = extract_layer_idx(r["name"])
            if idx is not None:
                g = classify_param(r["name"])
                all_groups_set.add(g)
                all_layers_set.add(idx)
                if g not in heatmap_data[idx]:
                    heatmap_data[idx][g] = []
                heatmap_data[idx][g].append(r["l2_relative"])
        if heatmap_data:
            layers_sorted = sorted(all_layers_set)
            groups_sorted = sorted(all_groups_set)
            matrix = np.zeros((len(groups_sorted), len(layers_sorted)))
            for li, layer in enumerate(layers_sorted):
                for gi, group in enumerate(groups_sorted):
                    vals = heatmap_data[layer].get(group, [0])
                    matrix[gi, li] = np.mean(vals)

            fig, ax = plt.subplots(figsize=(max(18, len(layers_sorted) * 0.4), max(6, len(groups_sorted) * 0.8)))
            im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
            ax.set_xticks(range(len(layers_sorted)))
            ax.set_xticklabels(layers_sorted, fontsize=6)
            ax.set_yticks(range(len(groups_sorted)))
            ax.set_yticklabels(groups_sorted, fontsize=8)
            ax.set_xlabel("Layer index")
            ax.set_title("Relative L2(Δw) Heatmap: Layer × Module Group")
            plt.colorbar(im, ax=ax, label="Relative L2(Δw)")
            fig.tight_layout()
            pdf.savefig(fig); plt.savefig(output_dir / "07_heatmap.png", dpi=150); plt.close(fig)

    print(f"[Done] Report saved to {pdf_path}")
    print(f"[Done] Individual PNGs saved to {output_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze Δw between pre-RL and post-RL checkpoints")
    parser.add_argument("--before", required=True, help="Path to pre-RL HuggingFace checkpoint directory")
    parser.add_argument("--after", required=True, help="Path to post-RL HuggingFace checkpoint directory")
    parser.add_argument("--output", default="./delta_w_analysis", help="Output directory for plots and data")
    parser.add_argument("--top-k-svd", type=int, default=64, help="Number of top singular values to keep")
    parser.add_argument("--spectral-bins", type=int, default=200, help="Number of bins for spectral density histogram")
    parser.add_argument("--skip-svd", action="store_true", help="Skip SVD computation (faster, less memory)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading before checkpoint: {args.before}")
    state_before = load_state_dict(args.before)
    print(f"  -> {len(state_before)} parameters loaded")

    print(f"Loading after checkpoint: {args.after}")
    state_after = load_state_dict(args.after)
    print(f"  -> {len(state_after)} parameters loaded")

    # Only analyze shared keys
    shared_keys = sorted(set(state_before.keys()) & set(state_after.keys()))
    only_before = set(state_before.keys()) - set(state_after.keys())
    only_after = set(state_after.keys()) - set(state_before.keys())
    if only_before:
        print(f"  [WARN] {len(only_before)} keys only in before: {list(only_before)[:5]}...")
    if only_after:
        print(f"  [WARN] {len(only_after)} keys only in after: {list(only_after)[:5]}...")

    print(f"\nAnalyzing {len(shared_keys)} shared parameters...")
    records = []
    for i, key in enumerate(shared_keys):
        if i % 50 == 0:
            print(f"  [{i}/{len(shared_keys)}] {key}")

        w_before = state_before[key]
        w_after = state_after[key]

        if w_before.shape != w_after.shape:
            print(f"  [SKIP] {key}: shape mismatch {w_before.shape} vs {w_after.shape}")
            continue

        top_k = args.top_k_svd if not args.skip_svd else 0
        rec = analyze_pair(key, w_before, w_after, top_k)
        records.append(rec)

        # free memory for large tensors
        del w_before, w_after

    # Free full state dicts
    del state_before, state_after
    gc.collect()

    # Save raw data as JSON
    json_path = output_dir / "delta_w_metrics.json"
    # Convert for JSON serialization
    json_records = []
    for r in records:
        jr = {k: v for k, v in r.items() if k != "spectral_values"}
        json_records.append(jr)
    with open(json_path, "w") as f:
        json.dump(json_records, f, indent=2)
    print(f"\nMetrics saved to {json_path}")

    # Print summary table
    print("\n" + "=" * 100)
    print(f"{'Parameter':<60} {'L2(Δw)':>10} {'Rel L2':>10} {'Cos Sim':>10} {'Eff Rank':>10}")
    print("=" * 100)
    # Sort by relative L2 descending
    for r in sorted(records, key=lambda x: x["l2_relative"], reverse=True)[:30]:
        eff_r = f"{r['effective_rank']:.1f}" if "effective_rank" in r else "N/A"
        print(f"{r['name']:<60} {r['l2_delta']:>10.6f} {r['l2_relative']:>10.6f} {r['cosine_sim']:>10.6f} {eff_r:>10}")
    print("..." if len(records) > 30 else "")

    # Module group summary
    print("\n" + "=" * 80)
    print("Module Group Summary")
    print("=" * 80)
    group_stats = defaultdict(lambda: {"l2": 0, "rel": [], "cos": [], "effrank": [], "numel": 0})
    for r in records:
        g = classify_param(r["name"])
        group_stats[g]["l2"] += r["l2_delta"] ** 2
        group_stats[g]["rel"].append(r["l2_relative"])
        group_stats[g]["cos"].append(r["cosine_sim"])
        group_stats[g]["numel"] += r["numel"]
        if "effective_rank" in r:
            group_stats[g]["effrank"].append(r["effective_rank"])

    print(f"{'Group':<20} {'L2(Δw)':>10} {'Mean Rel':>10} {'Mean Cos':>10} {'Mean Rank':>10} {'#Params':>12}")
    print("-" * 80)
    for g in sorted(group_stats.keys()):
        s = group_stats[g]
        l2 = np.sqrt(s["l2"])
        mean_rel = np.mean(s["rel"])
        mean_cos = np.mean(s["cos"])
        mean_rank = np.mean(s["effrank"]) if s["effrank"] else float("nan")
        print(f"{g:<20} {l2:>10.4f} {mean_rel:>10.6f} {mean_cos:>10.6f} {mean_rank:>10.1f} {s['numel']:>12,}")

    # Generate plots
    print("\nGenerating plots...")
    plot_all(records, output_dir, args.spectral_bins)
    print("All done!")


if __name__ == "__main__":
    main()

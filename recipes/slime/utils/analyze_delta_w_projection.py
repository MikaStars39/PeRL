#!/usr/bin/env python3
"""
Project Δw onto the principal components of w_before and analyze
whether RL updates concentrate more on non-principal components.

Hypothesis: RL updates modify the "less important" directions of
the original weight space more than the principal directions.

Core math:
    w_before = U Σ V^T
    Right projection energy: e_j = ||Δw v_j||²
    Left  projection energy: f_j = ||u_j^T Δw||²
    Ratio: r_j = ê_j / σ̂_j²  (normalized energy ratio)
    If r_j increases with j → RL focuses on non-principal components

Usage:
    python analyze_delta_w_projection.py \
        --before /path/to/before_rl_checkpoint \
        --after  /path/to/after_rl_checkpoint \
        --output ./delta_w_analysis/projection \
        --modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
"""

import argparse
import gc
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_state_dict(ckpt_path: str) -> dict[str, torch.Tensor]:
    ckpt_path = Path(ckpt_path)
    state = {}
    st_files = sorted(ckpt_path.glob("*.safetensors"))
    if st_files:
        from safetensors.torch import load_file
        for f in st_files:
            state.update(load_file(str(f), device="cpu"))
        return state
    bin_files = sorted(ckpt_path.glob("*.bin"))
    if bin_files:
        for f in bin_files:
            state.update(torch.load(str(f), map_location="cpu", weights_only=True))
        return state
    raise FileNotFoundError(f"No .safetensors or .bin files in {ckpt_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MODULE_GROUPS = {
    "attn.q_proj": ["q_proj"],
    "attn.k_proj": ["k_proj"],
    "attn.v_proj": ["v_proj"],
    "attn.o_proj": ["o_proj"],
    "mlp.gate_proj": ["gate_proj", "gate_up_proj"],
    "mlp.up_proj": ["up_proj"],
    "mlp.down_proj": ["down_proj"],
    "embed": ["embed_tokens", "wte", "wpe"],
    "lm_head": ["lm_head"],
}


def classify_param(name: str) -> str:
    name_lower = name.lower()
    for group, keywords in MODULE_GROUPS.items():
        for kw in keywords:
            if kw in name_lower:
                return group
    return "other"


def extract_layer_idx(name: str) -> int | None:
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p in ("layers", "h", "block"):
            if i + 1 < len(parts) and parts[i + 1].isdigit():
                return int(parts[i + 1])
    return None


def should_analyze(name: str, modules_filter: list[str] | None) -> bool:
    if modules_filter is None:
        return True
    name_lower = name.lower()
    return any(m in name_lower for m in modules_filter)


# ---------------------------------------------------------------------------
# Core projection analysis
# ---------------------------------------------------------------------------

def analyze_projection(
    name: str,
    w_before: torch.Tensor,
    delta_w: torch.Tensor,
    n_bins: int = 10,
) -> dict:
    """
    SVD w_before, project Δw onto its singular vectors, compare energy distributions.
    """
    w = w_before.float()
    dw = delta_w.float()
    m, n = w.shape
    k = min(m, n)

    result = {
        "name": name,
        "shape": list(w.shape),
        "k": k,  # number of singular values
        "group": classify_param(name),
        "layer_idx": extract_layer_idx(name),
    }

    # --- Full economy SVD of w_before ---
    U, S, Vt = torch.linalg.svd(w, full_matrices=False)
    # U: (m, k), S: (k,), Vt: (k, n)
    V = Vt.T  # (n, k)

    S_sq = (S ** 2).numpy()
    S_sq_norm = S_sq / S_sq.sum()  # normalized importance of w_before

    # --- Right projection: e_j = ||Δw v_j||² ---
    # Δw @ V -> (m, k), column j = Δw v_j
    dw_V = dw @ V  # (m, k)
    e_right = (dw_V ** 2).sum(dim=0).numpy()  # (k,)
    e_right_norm = e_right / e_right.sum()

    # --- Left projection: f_j = ||u_j^T Δw||² ---
    # U^T @ Δw -> (k, n), row j = u_j^T Δw
    Ut_dw = U.T @ dw  # (k, n)
    f_left = (Ut_dw ** 2).sum(dim=1).numpy()  # (k,)
    f_left_norm = f_left / f_left.sum()

    # --- Energy ratio: r_j = ê_j / σ̂_j² ---
    # Avoid division by zero for very small singular values
    eps = 1e-12
    ratio_right = e_right_norm / np.maximum(S_sq_norm, eps)
    ratio_left = f_left_norm / np.maximum(S_sq_norm, eps)

    # --- Binned analysis ---
    bin_edges = np.linspace(0, k, n_bins + 1, dtype=int)
    bins_w = []      # w_before energy per bin
    bins_right = []   # Δw right-projection energy per bin
    bins_left = []    # Δw left-projection energy per bin
    bin_labels = []
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        bins_w.append(S_sq_norm[lo:hi].sum())
        bins_right.append(e_right_norm[lo:hi].sum())
        bins_left.append(f_left_norm[lo:hi].sum())
        pct_lo = int(100 * lo / k)
        pct_hi = int(100 * hi / k)
        bin_labels.append(f"{pct_lo}-{pct_hi}%")

    # --- Cross-cumulative: top-k of w_before captures x% of Δw ---
    cum_w = np.cumsum(S_sq_norm)
    cum_right = np.cumsum(e_right_norm)
    cum_left = np.cumsum(f_left_norm)

    # Find: when w_before reaches 90%/95%/99%, how much of Δw is captured?
    thresholds = {}
    for pct in [0.5, 0.8, 0.9, 0.95, 0.99]:
        idx = int(np.searchsorted(cum_w, pct))
        thresholds[f"w{int(pct*100)}"] = {
            "rank": idx + 1,
            "rank_frac": (idx + 1) / k,
            "dw_right_captured": float(cum_right[min(idx, k - 1)]),
            "dw_left_captured": float(cum_left[min(idx, k - 1)]),
        }

    # --- "Non-principal importance" metric ---
    # Split at median singular value: top half = principal, bottom half = non-principal
    mid = k // 2
    principal_energy_w = S_sq_norm[:mid].sum()
    principal_energy_right = e_right_norm[:mid].sum()
    principal_energy_left = f_left_norm[:mid].sum()

    # Also split at 90% energy of w
    idx_90 = int(np.searchsorted(cum_w, 0.9))
    top90_energy_right = e_right_norm[:idx_90 + 1].sum()
    top90_energy_left = f_left_norm[:idx_90 + 1].sum()

    result.update({
        # Raw curves (sampled to save space — keep every point for small k, subsample for large k)
        "S_sq_norm": _subsample(S_sq_norm, 500),
        "e_right_norm": _subsample(e_right_norm, 500),
        "e_left_norm": _subsample(f_left_norm, 500),
        "ratio_right": _subsample(ratio_right, 500),
        "ratio_left": _subsample(ratio_left, 500),
        "cum_w": _subsample(cum_w, 500),
        "cum_right": _subsample(cum_right, 500),
        "cum_left": _subsample(cum_left, 500),
        # Binned
        "bin_labels": bin_labels,
        "bins_w": [float(x) for x in bins_w],
        "bins_right": [float(x) for x in bins_right],
        "bins_left": [float(x) for x in bins_left],
        # Thresholds
        "thresholds": thresholds,
        # Summary metrics
        "principal_half_energy_w": float(principal_energy_w),
        "principal_half_energy_dw_right": float(principal_energy_right),
        "principal_half_energy_dw_left": float(principal_energy_left),
        "top90w_captures_dw_right": float(top90_energy_right),
        "top90w_captures_dw_left": float(top90_energy_left),
        "idx_90_rank": int(idx_90 + 1),
        "idx_90_rank_frac": float((idx_90 + 1) / k),
    })

    return result


def _subsample(arr: np.ndarray, max_len: int) -> list[float]:
    if len(arr) <= max_len:
        return arr.tolist()
    indices = np.linspace(0, len(arr) - 1, max_len, dtype=int)
    return arr[indices].tolist()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_all(records: list[dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "projection_report.pdf"

    with PdfPages(str(pdf_path)) as pdf:
        # ── 1. Per-layer binned comparison (sample a few representative layers) ──
        # Pick 6 layers spread across the model
        layers_with_idx = [r for r in records if r["layer_idx"] is not None]
        if layers_with_idx:
            unique_layers = sorted(set(r["layer_idx"] for r in layers_with_idx))
            sample_layers = unique_layers[:: max(1, len(unique_layers) // 6)][:6]
            sample_groups = ["attn.q_proj", "mlp.down_proj"]

            for group in sample_groups:
                group_records = [r for r in layers_with_idx
                                 if r["group"] == group and r["layer_idx"] in sample_layers]
                if not group_records:
                    continue

                n = len(group_records)
                fig, axes = plt.subplots(2, min(n, 3), figsize=(6 * min(n, 3), 10),
                                         squeeze=False)
                for idx, r in enumerate(group_records[:6]):
                    row = idx // 3
                    col = idx % 3
                    if row >= axes.shape[0] or col >= axes.shape[1]:
                        break
                    ax = axes[row, col]
                    x = np.arange(len(r["bin_labels"]))
                    width = 0.25
                    ax.bar(x - width, r["bins_w"], width, label="w_before energy", alpha=0.8)
                    ax.bar(x, r["bins_right"], width, label="Δw right-proj", alpha=0.8)
                    ax.bar(x + width, r["bins_left"], width, label="Δw left-proj", alpha=0.8)
                    ax.set_xticks(x)
                    ax.set_xticklabels(r["bin_labels"], rotation=45, fontsize=7)
                    ax.set_title(f"Layer {r['layer_idx']} {group}", fontsize=9)
                    ax.set_ylabel("Energy fraction")
                    ax.legend(fontsize=6)
                fig.suptitle(f"Binned Energy Distribution: {group}\n"
                             f"(bins = SV rank percentile, left=principal, right=non-principal)",
                             fontsize=11)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.savefig(output_dir / f"01_binned_{group.replace('.', '_')}.png", dpi=150)
                plt.close(fig)

        # ── 2. Cross-cumulative energy (all layers overlaid, per module group) ──
        groups_to_plot = ["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj",
                          "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
        available_groups = sorted(set(r["group"] for r in records))
        groups_to_plot = [g for g in groups_to_plot if g in available_groups]

        if groups_to_plot:
            n_groups = len(groups_to_plot)
            cols = min(n_groups, 4)
            rows = (n_groups + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows), squeeze=False)

            for gi, group in enumerate(groups_to_plot):
                ax = axes[gi // cols, gi % cols]
                group_recs = [r for r in records if r["group"] == group]
                # Plot each layer as a thin line
                for r in group_recs:
                    x_w = np.linspace(0, 1, len(r["cum_w"]))
                    ax.plot(x_w, r["cum_right"], alpha=0.3, color="C1", linewidth=0.8)
                # Plot the mean
                min_len = min(len(r["cum_w"]) for r in group_recs)
                mean_cum_w = np.mean([np.interp(np.linspace(0, 1, min_len),
                                                np.linspace(0, 1, len(r["cum_w"])),
                                                r["cum_w"]) for r in group_recs], axis=0)
                mean_cum_right = np.mean([np.interp(np.linspace(0, 1, min_len),
                                                    np.linspace(0, 1, len(r["cum_right"])),
                                                    r["cum_right"]) for r in group_recs], axis=0)
                x_norm = np.linspace(0, 1, min_len)
                ax.plot(x_norm, mean_cum_w, "b-", linewidth=2, label="w_before (mean)")
                ax.plot(x_norm, mean_cum_right, "r-", linewidth=2, label="Δw right-proj (mean)")
                ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.5)
                ax.set_xlabel("Rank fraction (0=top SV, 1=bottom SV)")
                ax.set_ylabel("Cumulative energy")
                ax.set_title(group, fontsize=10)
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)

            # Hide unused axes
            for gi in range(n_groups, rows * cols):
                axes[gi // cols, gi % cols].set_visible(False)

            fig.suptitle("Cumulative Energy: w_before vs Δw projection\n"
                         "(If red curve is below blue → Δw concentrates on non-principal components)",
                         fontsize=12)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.savefig(output_dir / "02_cumulative_comparison.png", dpi=150)
            plt.close(fig)

        # ── 3. Energy ratio r_j by layer (aggregated per module group) ──
        if groups_to_plot:
            fig, axes = plt.subplots(len(groups_to_plot), 1,
                                     figsize=(14, 3.5 * len(groups_to_plot)), squeeze=False)
            for gi, group in enumerate(groups_to_plot):
                ax = axes[gi, 0]
                group_recs = [r for r in records if r["group"] == group]
                # Average ratio curve across layers
                min_len = min(len(r["ratio_right"]) for r in group_recs)
                all_ratios = np.array([
                    np.interp(np.linspace(0, 1, min_len),
                              np.linspace(0, 1, len(r["ratio_right"])),
                              r["ratio_right"])
                    for r in group_recs
                ])
                mean_ratio = np.mean(all_ratios, axis=0)
                std_ratio = np.std(all_ratios, axis=0)
                x = np.linspace(0, 100, min_len)
                ax.plot(x, mean_ratio, "C0-", linewidth=1.5)
                ax.fill_between(x, mean_ratio - std_ratio, mean_ratio + std_ratio,
                                alpha=0.2, color="C0")
                ax.axhline(1.0, color="red", linestyle="--", alpha=0.5, linewidth=1)
                ax.set_xlabel("SV rank percentile (0%=top, 100%=bottom)")
                ax.set_ylabel("Energy ratio (ê_j / σ̂_j²)")
                ax.set_title(f"{group}: ratio > 1 means Δw over-represented vs w_before", fontsize=10)
                ax.grid(True, alpha=0.3)
                # Clip y axis for readability
                ax.set_ylim(0, min(np.percentile(mean_ratio + std_ratio, 99) * 1.5, 50))

            fig.suptitle("Energy Ratio r_j = ê_j / σ̂_j²\n"
                         "(Rising curve → RL focuses on non-principal components)", fontsize=12)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.savefig(output_dir / "03_energy_ratio.png", dpi=150)
            plt.close(fig)

        # ── 4. Summary bar: "principal vs non-principal" per module group ──
        if records:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # 4a: top-half energy comparison
            ax = axes[0]
            groups_seen = []
            w_vals, dw_r_vals, dw_l_vals = [], [], []
            for group in groups_to_plot:
                grecs = [r for r in records if r["group"] == group]
                if not grecs:
                    continue
                groups_seen.append(group)
                w_vals.append(np.mean([r["principal_half_energy_w"] for r in grecs]))
                dw_r_vals.append(np.mean([r["principal_half_energy_dw_right"] for r in grecs]))
                dw_l_vals.append(np.mean([r["principal_half_energy_dw_left"] for r in grecs]))

            x = np.arange(len(groups_seen))
            width = 0.25
            ax.bar(x - width, w_vals, width, label="w_before", alpha=0.8)
            ax.bar(x, dw_r_vals, width, label="Δw (right)", alpha=0.8)
            ax.bar(x + width, dw_l_vals, width, label="Δw (left)", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(groups_seen, rotation=30, fontsize=8)
            ax.set_ylabel("Energy in top-50% SVs")
            ax.set_title("Principal Half Energy")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3, axis="y")

            # 4b: w_before 90% rank captures how much of Δw?
            ax = axes[1]
            capture_vals = []
            for group in groups_seen:
                grecs = [r for r in records if r["group"] == group]
                capture_vals.append(np.mean([r["top90w_captures_dw_right"] for r in grecs]))
            ax.bar(groups_seen, capture_vals, color="C1", alpha=0.8)
            ax.axhline(0.9, color="red", linestyle="--", label="90% baseline")
            ax.set_xticklabels(groups_seen, rotation=30, fontsize=8)
            ax.set_ylabel("Δw energy captured")
            ax.set_title("w_before top-90% SVs capture how much Δw?")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3, axis="y")

            # 4c: non-principal energy ratio by layer
            ax = axes[2]
            for group in groups_to_plot[:4]:
                grecs = sorted([r for r in records if r["group"] == group],
                               key=lambda r: r["layer_idx"] or -1)
                grecs = [r for r in grecs if r["layer_idx"] is not None]
                if not grecs:
                    continue
                layers = [r["layer_idx"] for r in grecs]
                # non-principal fraction of Δw / non-principal fraction of w
                nonprinc_ratio = []
                for r in grecs:
                    dw_nonprinc = 1 - r["principal_half_energy_dw_right"]
                    w_nonprinc = 1 - r["principal_half_energy_w"]
                    nonprinc_ratio.append(dw_nonprinc / max(w_nonprinc, 1e-12))
                ax.plot(layers, nonprinc_ratio, "o-", label=group, markersize=3, alpha=0.8)
            ax.axhline(1.0, color="red", linestyle="--", alpha=0.5)
            ax.set_xlabel("Layer index")
            ax.set_ylabel("Non-principal enrichment ratio")
            ax.set_title("Non-principal enrichment by layer\n(>1 = Δw over-weights non-principal)")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

            fig.tight_layout()
            pdf.savefig(fig)
            plt.savefig(output_dir / "04_principal_summary.png", dpi=150)
            plt.close(fig)

        # ── 5. Heatmap: Δw capture rate by (layer, module) ──
        heatmap_data = defaultdict(dict)
        all_groups_set = set()
        all_layers_set = set()
        for r in records:
            idx = r["layer_idx"]
            if idx is not None:
                g = r["group"]
                all_groups_set.add(g)
                all_layers_set.add(idx)
                heatmap_data[idx][g] = r["top90w_captures_dw_right"]

        if heatmap_data:
            layers_sorted = sorted(all_layers_set)
            groups_sorted = sorted(all_groups_set)
            matrix = np.full((len(groups_sorted), len(layers_sorted)), np.nan)
            for li, layer in enumerate(layers_sorted):
                for gi, group in enumerate(groups_sorted):
                    if group in heatmap_data[layer]:
                        matrix[gi, li] = heatmap_data[layer][group]

            fig, ax = plt.subplots(figsize=(max(18, len(layers_sorted) * 0.4),
                                            max(4, len(groups_sorted) * 0.8)))
            im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
            ax.set_xticks(range(len(layers_sorted)))
            ax.set_xticklabels(layers_sorted, fontsize=6)
            ax.set_yticks(range(len(groups_sorted)))
            ax.set_yticklabels(groups_sorted, fontsize=8)
            ax.set_xlabel("Layer index")
            ax.set_title("Δw capture rate by w_before top-90% SVs\n"
                         "(Green=high capture=principal-dominated, Red=low capture=non-principal)")
            plt.colorbar(im, ax=ax, label="Δw energy captured by top-90% SVs of w")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.savefig(output_dir / "05_capture_heatmap.png", dpi=150)
            plt.close(fig)

        # ── 6. Detailed per-layer curves for one selected module ──
        # Pick the module with most records
        group_counts = defaultdict(int)
        for r in records:
            if r["layer_idx"] is not None:
                group_counts[r["group"]] += 1
        if group_counts:
            best_group = max(group_counts, key=group_counts.get)
            grecs = sorted([r for r in records if r["group"] == best_group],
                           key=lambda r: r["layer_idx"] or -1)
            grecs = [r for r in grecs if r["layer_idx"] is not None]

            if grecs:
                n = len(grecs)
                cols = min(6, n)
                rows = (n + cols - 1) // cols
                fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), squeeze=False)
                for idx, r in enumerate(grecs):
                    ax = axes[idx // cols, idx % cols]
                    x = np.linspace(0, 100, len(r["cum_w"]))
                    ax.plot(x, r["cum_w"], "b-", linewidth=1.5, label="w_before")
                    ax.plot(x, r["cum_right"], "r-", linewidth=1.5, label="Δw")
                    ax.fill_between(x, r["cum_w"], r["cum_right"],
                                    where=np.array(r["cum_w"]) > np.array(r["cum_right"]),
                                    alpha=0.2, color="red", label="Δw deficit")
                    ax.set_title(f"L{r['layer_idx']}", fontsize=8)
                    ax.set_xlim(0, 100)
                    ax.set_ylim(0, 1.05)
                    ax.grid(True, alpha=0.3)
                    if idx == 0:
                        ax.legend(fontsize=6)

                for idx in range(n, rows * cols):
                    axes[idx // cols, idx % cols].set_visible(False)

                fig.suptitle(f"Cumulative Energy per Layer: {best_group}\n"
                             f"(Red area = Δw energy NOT on principal components)", fontsize=12)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.savefig(output_dir / "06_per_layer_cumulative.png", dpi=150)
                plt.close(fig)

    print(f"[Done] Report saved to {pdf_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Project Δw onto w_before's principal components")
    parser.add_argument("--before", required=True,
                        help="Path to pre-RL HuggingFace checkpoint")
    parser.add_argument("--after", required=True,
                        help="Path to post-RL HuggingFace checkpoint")
    parser.add_argument("--output", default="./delta_w_analysis/projection",
                        help="Output directory")
    parser.add_argument("--modules", nargs="*", default=None,
                        help="Only analyze params containing these substrings "
                             "(e.g., q_proj k_proj v_proj). Default: all 2D weights.")
    parser.add_argument("--n-bins", type=int, default=10,
                        help="Number of bins for energy distribution")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading before: {args.before}")
    state_before = load_state_dict(args.before)
    print(f"  -> {len(state_before)} params")

    print(f"Loading after: {args.after}")
    state_after = load_state_dict(args.after)
    print(f"  -> {len(state_after)} params")

    shared_keys = sorted(set(state_before.keys()) & set(state_after.keys()))

    # Filter to 2D weight matrices that match module filter
    keys_to_analyze = []
    for key in shared_keys:
        if state_before[key].dim() != 2:
            continue
        if min(state_before[key].shape) < 2:
            continue
        if not should_analyze(key, args.modules):
            continue
        if state_before[key].shape != state_after[key].shape:
            continue
        keys_to_analyze.append(key)

    print(f"\nWill analyze {len(keys_to_analyze)} weight matrices (SVD each one)")
    print(f"Modules filter: {args.modules or 'all'}\n")

    records = []
    for i, key in enumerate(keys_to_analyze):
        shape = state_before[key].shape
        print(f"  [{i+1}/{len(keys_to_analyze)}] {key}  {list(shape)}  "
              f"(SVD of {shape[0]}x{shape[1]}, k={min(shape)})")

        w_before = state_before[key]
        w_after = state_after[key]
        delta_w = w_after.float() - w_before.float()

        rec = analyze_projection(key, w_before, delta_w, n_bins=args.n_bins)
        records.append(rec)

        del w_before, w_after, delta_w
        gc.collect()

    del state_before, state_after
    gc.collect()

    # ── Save JSON (without large arrays to keep file small) ──
    json_records = []
    for r in records:
        jr = {k: v for k, v in r.items()
              if k not in ("S_sq_norm", "e_right_norm", "e_left_norm",
                           "ratio_right", "ratio_left",
                           "cum_w", "cum_right", "cum_left")}
        json_records.append(jr)
    json_path = output_dir / "projection_metrics.json"
    with open(json_path, "w") as f:
        json.dump(json_records, f, indent=2)

    # ── Print summary ──
    print("\n" + "=" * 100)
    print("PROJECTION ANALYSIS SUMMARY")
    print("=" * 100)
    print(f"\n{'Parameter':<50} {'top50% w':>10} {'top50% Δw':>10} "
          f"{'w90% capt':>10} {'90% rank':>10}")
    print("-" * 100)
    for r in records:
        short = r["name"].replace("model.layers.", "L").replace(".self_attn.", ".attn.")
        short = short.replace(".mlp.", ".mlp.").replace(".weight", "")
        print(f"{short:<50} "
              f"{r['principal_half_energy_w']:>10.4f} "
              f"{r['principal_half_energy_dw_right']:>10.4f} "
              f"{r['top90w_captures_dw_right']:>10.4f} "
              f"{r['idx_90_rank']:>6d}/{r['k']:<4d}")

    # ── Module group summary ──
    print("\n" + "=" * 80)
    print("MODULE GROUP SUMMARY")
    print("=" * 80)
    group_stats = defaultdict(lambda: {
        "w_half": [], "dw_half": [], "capture90": [], "idx90_frac": []
    })
    for r in records:
        g = r["group"]
        group_stats[g]["w_half"].append(r["principal_half_energy_w"])
        group_stats[g]["dw_half"].append(r["principal_half_energy_dw_right"])
        group_stats[g]["capture90"].append(r["top90w_captures_dw_right"])
        group_stats[g]["idx90_frac"].append(r["idx_90_rank_frac"])

    print(f"\n{'Group':<20} {'w top50%':>10} {'Δw top50%':>10} {'Diff':>10} "
          f"{'w90% capt Δw':>12} {'Conclusion':<30}")
    print("-" * 100)
    for g in sorted(group_stats.keys()):
        s = group_stats[g]
        w_h = np.mean(s["w_half"])
        dw_h = np.mean(s["dw_half"])
        capt = np.mean(s["capture90"])
        conclusion = ""
        if dw_h < w_h - 0.02:
            conclusion = "Δw favors NON-principal ✓"
        elif dw_h > w_h + 0.02:
            conclusion = "Δw favors PRINCIPAL"
        else:
            conclusion = "~isotropic"
        print(f"{g:<20} {w_h:>10.4f} {dw_h:>10.4f} {dw_h - w_h:>+10.4f} "
              f"{capt:>12.4f} {conclusion:<30}")

    # ── Hypothesis test summary ──
    all_w_half = [r["principal_half_energy_w"] for r in records]
    all_dw_half = [r["principal_half_energy_dw_right"] for r in records]
    all_capture = [r["top90w_captures_dw_right"] for r in records]

    print(f"\n{'='*60}")
    print("HYPOTHESIS: Δw concentrates on non-principal components")
    print(f"{'='*60}")
    print(f"  w_before top-50% SVs contain:  {np.mean(all_w_half):.4f} of w energy")
    print(f"  Δw projection on same SVs:     {np.mean(all_dw_half):.4f} of Δw energy")
    print(f"  w_before top-90% SVs capture:  {np.mean(all_capture):.4f} of Δw energy")
    if np.mean(all_dw_half) < np.mean(all_w_half):
        print(f"\n  → CONFIRMED: Δw has relatively MORE energy on non-principal components")
        print(f"    (Δw puts {np.mean(all_w_half) - np.mean(all_dw_half):.4f} less energy "
              f"on principal half)")
    else:
        print(f"\n  → NOT CONFIRMED: Δw has similar or more energy on principal components")

    # ── Plots ──
    print("\nGenerating plots...")
    plot_all(records, output_dir)
    print("All done!")


if __name__ == "__main__":
    main()

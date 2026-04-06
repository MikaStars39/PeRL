#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot eval results across multiple model directories.

Usage:
    python plot_eval.py [--output plot.png]

Reads eval_results.json from all iter_*-hf/ directories under each MODEL_BASE,
plots avg_k and pass_k for each benchmark + overall.
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

# ========== 配置 ==========
MODEL_BASES = [
    "/jpfs-5p/chenyanxu.9/model/Qwen3-8B-onpolicy-profiling-20260403_091551",
]

# 从目录名提取简短标签
def short_label(path):
    name = os.path.basename(path)
    # e.g. Qwen3-8B-onpolicy-profiling-20260403_091551 -> onpolicy
    m = re.search(r"Qwen3-8B-(.+?)-(?:profiling|rl)", name)
    return m.group(1) if m else name


def load_results(model_base):
    """Load all eval_results.json under model_base, return {iter_num: {bmk: {avg_k, pass_k}}}."""
    results = {}
    base = Path(model_base)
    for hf_dir in sorted(base.glob("iter_*-hf")):
        result_file = hf_dir / "eval_results.json"
        if not result_file.exists():
            continue
        # extract iter number
        m = re.match(r"iter_(\d+)-hf", hf_dir.name)
        if not m:
            continue
        iter_num = int(m.group(1))
        bmk_data = {}
        with open(result_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                bmk_name, metrics = row[0], row[1]
                bmk_data[bmk_name] = metrics
        results[iter_num] = bmk_data
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="eval_results.png")
    args = parser.parse_args()

    # {label: {iter_num: {bmk: metrics}}}
    all_data = {}
    all_bmks = set()
    for mb in MODEL_BASES:
        label = short_label(mb)
        data = load_results(mb)
        if data:
            all_data[label] = data
            for bmk_dict in data.values():
                all_bmks.update(bmk_dict.keys())

    if not all_data:
        print("[WARN] No eval results found!")
        return

    # 排序 benchmark: 先具体 bmk，最后 overall
    bmks = sorted(b for b in all_bmks if b != "overall")
    # 每个 bmk 两张图 (avg_k, pass_k)，再加 overall 两张图
    metrics = ["avg_k", "pass_k"]

    n_bmk = len(bmks)
    n_plots = (n_bmk + 1) * 2  # +1 for overall, x2 for avg_k/pass_k
    n_cols = 4
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    axes = axes.flatten()

    plot_idx = 0
    # Per-benchmark plots
    for bmk in bmks:
        for metric in metrics:
            ax = axes[plot_idx]
            for label, data in all_data.items():
                iters = sorted(data.keys())
                vals = []
                valid_iters = []
                for it in iters:
                    if bmk in data[it] and metric in data[it][bmk]:
                        valid_iters.append(it)
                        vals.append(data[it][bmk][metric])
                if valid_iters:
                    ax.plot(valid_iters, vals, marker="o", markersize=3, label=label)
            ax.set_title(f"{bmk} - {metric}")
            ax.set_xlabel("iter")
            ax.set_ylabel(metric)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            plot_idx += 1

    # Overall plots
    for metric in metrics:
        ax = axes[plot_idx]
        for label, data in all_data.items():
            iters = sorted(data.keys())
            vals = []
            valid_iters = []
            for it in iters:
                if "overall" in data[it] and metric in data[it]["overall"]:
                    valid_iters.append(it)
                    vals.append(data[it]["overall"][metric])
            if valid_iters:
                ax.plot(valid_iters, vals, marker="o", markersize=3, linewidth=2, label=label)
        ax.set_title(f"OVERALL - {metric}", fontweight="bold")
        ax.set_xlabel("iter")
        ax.set_ylabel(metric)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # 隐藏多余的 subplot
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Eval Results Across Models", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"[OK] Plot saved to {args.output}")


if __name__ == "__main__":
    main()

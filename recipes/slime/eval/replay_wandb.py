#!/usr/bin/env python3
"""
从 SLIME 的 output.log 解析训练/rollout/perf 指标，replay 到 W&B。

用法:
    python replay_wandb.py \
        --log /jpfs-5p/.../output.log \
        --project slime-rl-optim \
        --name "Qwen3-8B-cispo-rl-20260408_142430"
"""

import argparse
import ast
import os
import re

os.environ.setdefault("WANDB_BASE_URL", "http://11.71.1.153:8080")

import wandb

# 三种日志行的正则
STEP_RE = re.compile(r"model\.py:\d+ - step (\d+): (\{.+\})")
ROLLOUT_RE = re.compile(r"data\.py:\d+ - rollout (\d+): (\{.+\})")
PERF_ROLLOUT_RE = re.compile(r"rollout\.py:\d+ - perf (\d+): (\{.+\})")
PERF_TRAIN_RE = re.compile(r"train_metric_utils\.py:\d+ - perf (\d+): (\{.+\})")


def parse_output_log(path: str):
    """解析 output.log，返回按全局顺序排列的 (type, idx, dict) 列表。"""
    events = []  # (line_no, type, idx, dict)

    with open(path, "r", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            for pattern, etype in [
                (STEP_RE, "step"),
                (ROLLOUT_RE, "rollout"),
                (PERF_ROLLOUT_RE, "perf_rollout"),
                (PERF_TRAIN_RE, "perf_train"),
            ]:
                m = pattern.search(line)
                if m:
                    idx = int(m.group(1))
                    try:
                        data = ast.literal_eval(m.group(2))
                        if isinstance(data, dict):
                            events.append((line_no, etype, idx, data))
                    except Exception:
                        pass
                    break  # 一行只匹配一种

    return events


def main():
    parser = argparse.ArgumentParser(description="Replay SLIME output.log to W&B")
    parser.add_argument("--log", required=True, help="Path to output.log")
    parser.add_argument("--project", default="slime-rl-optim")
    parser.add_argument("--entity", default="duo")
    parser.add_argument("--name", default=None)
    parser.add_argument("--host", default="http://11.71.1.153:8080")
    parser.add_argument("--api-key", default=os.environ.get("WANDB_API_KEY"))
    args = parser.parse_args()

    os.environ["WANDB_BASE_URL"] = args.host

    # 1. 解析
    print(f"Parsing {args.log} ...")
    events = parse_output_log(args.log)

    # 统计
    counts = {}
    for _, etype, _, _ in events:
        counts[etype] = counts.get(etype, 0) + 1
    print(f"Total events: {len(events)}")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    print()

    if not events:
        print("No events found, exiting")
        return

    # 去重：同类型同 idx 只保留第一条
    seen = set()
    deduped = []
    for e in events:
        key = (e[1], e[2])  # (type, idx)
        if key not in seen:
            seen.add(key)
            deduped.append(e)
    events = deduped
    print(f"After dedup: {len(events)} events\n")

    # 2. 登录
    if not args.api_key:
        print("[ERROR] 请设置 --api-key 或 export WANDB_API_KEY=...")
        return
    wandb.login(key=args.api_key, host=args.host)
    run = wandb.init(project=args.project, entity=args.entity, name=args.name)
    print(f"Created new run: {run.url}\n")

    # 3. 按日志出现顺序逐条上传
    for i, (line_no, etype, idx, data) in enumerate(events):
        run.log(data)
        if (i + 1) % 100 == 0:
            print(f"  uploaded {i + 1}/{len(events)} ({etype} #{idx})")

    run.finish()
    print(f"\nDone! {len(events)} events uploaded.")
    print(f"Run URL: {run.url}")


if __name__ == "__main__":
    main()

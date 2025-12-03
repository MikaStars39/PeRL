#!/usr/bin/env python3
"""
Generate a pre-baked eval set from the deduplicated math data, allowing
per-data_source replication (e.g., sample AIME 3x, AMC 1x).

Default input: ../check/valid.math.dedup.parquet
Default output: bench.generated.parquet + bench.generated.json in this folder.

Examples:
  python make_bench.py --repeat math/aime=3 --repeat math/aime25=2
  python make_bench.py --input ../check/valid.math.dedup.json --output my.parquet
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import numpy as np


def load_dataset(path: Path) -> List[List[Any]]:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        records = df.to_dict(orient="records")
        # Normalize prompt if it was stored as ndarray
        for r in records:
            if isinstance(r.get("prompt"), np.ndarray):
                r["prompt"] = r["prompt"].tolist()
        # Keep column order: data_source, prompt, reward_model, difficulty
        return [[r["data_source"], r["prompt"], r["reward_model"], int(r["difficulty"])] for r in records]
    # JSON expected to be list of 4-element rows
    return json.loads(path.read_text())


def parse_repeats(items: List[str]) -> Dict[str, int]:
    repeats: Dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--repeat expects name=count, got {item}")
        name, count = item.split("=", 1)
        repeats[name] = int(count)
    return repeats


def replicate(data: List[List[Any]], repeats: Dict[str, int]) -> List[List[Any]]:
    out: List[List[Any]] = []
    for row in data:
        source = row[0]
        times = repeats.get(source, 1)
        out.extend([row] * times)
    return out


def save_outputs(data: List[List[Any]], out_base: Path):
    # Save parquet and json side by side
    columns = ["data_source", "prompt", "reward_model", "difficulty"]
    df = pd.DataFrame(data, columns=columns)
    df.to_parquet(out_base.with_suffix(".parquet"), index=False)
    out_base.with_suffix(".json").write_text(json.dumps(data, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="Replicate deduped eval data per data_source.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("valid.math.dedup.parquet"),
        help="Input deduplicated dataset (parquet or json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("bench.generated.parquet"),
        help="Output base filename (parquet/json will be produced).",
    )
    parser.add_argument(
        "--repeat",
        action="append",
        default=[],
        help="Repeat rule in the form data_source=count. Can be passed multiple times.",
    )
    args = parser.parse_args()

    data = load_dataset(args.input)
    repeats = parse_repeats(args.repeat)
    replicated = replicate(data, repeats)
    save_outputs(replicated, args.output)

    print(f"Loaded {len(data)} rows from {args.input}")
    if repeats:
        print(f"Applied repeats: {repeats}")
    print(f"Final rows: {len(replicated)}")
    print(f"Written: {args.output.with_suffix('.parquet')} and {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()

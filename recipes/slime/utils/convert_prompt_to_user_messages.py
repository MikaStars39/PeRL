#!/usr/bin/env python3
"""Convert Polaris parquet: prompt (string) -> prompt as list[{"role":"user","content":...}]."""
import argparse
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "parquet",
        nargs="?",
        default="/jpfs/chenyanxu.9/data/Polaris-V2-RL-14K/train-00000-of-00001.parquet",
    )
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    path = Path(args.parquet)
    if not path.is_file():
        raise SystemExit(f"not found: {path}")

    table = pq.read_table(path)
    if "prompt" not in table.column_names:
        raise SystemExit("no column 'prompt'")

    col = table.column("prompt")
    if pa.types.is_string(col.type) or pa.types.is_large_string(col.type):
        prompts = col.to_pylist()
    else:
        raise SystemExit(f"unexpected prompt type: {col.type}")

    rows = [[{"role": "user", "content": p}] for p in prompts]
    msg_type = pa.list_(
        pa.struct([("role", pa.string()), ("content", pa.string())])
    )
    prompt_arr = pa.array(rows, type=msg_type)

    names = table.column_names
    idx = names.index("prompt")
    cols = []
    for name in names:
        if name == "prompt":
            cols.append(prompt_arr)
        else:
            cols.append(table.column(name))
    new_table = pa.table(dict(zip(names, cols)))

    if not args.no_backup:
        bak = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, bak)
        print(f"backup: {bak}")

    pq.write_table(new_table, path)
    print(f"wrote: {path}")
    print(f"rows: {new_table.num_rows}")
    print("schema prompt:", new_table.schema.field("prompt"))


if __name__ == "__main__":
    main()

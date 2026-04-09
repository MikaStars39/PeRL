#!/usr/bin/env python3
"""诊断 .wandb record 内部结构"""
import sys, os, json

path = sys.argv[1]

from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2 as pb

ds = DataStore()
ds.open_for_scan(path)

field_counts = {}
sample_records = []

for i in range(200):
    try:
        raw = ds.scan_record()
    except Exception as e:
        print(f"Record {i}: scan error: {e}")
        break
    if raw is None:
        print(f"Record {i}: None (EOF)")
        break

    # 看 tuple 结构
    if isinstance(raw, tuple):
        if i < 3:
            print(f"Record {i}: tuple len={len(raw)}, types={[type(x).__name__ for x in raw]}")
            for j, elem in enumerate(raw):
                if isinstance(elem, (bytes, bytearray)):
                    print(f"  elem[{j}]: {len(elem)} bytes, first 50: {elem[:50]}")
                elif isinstance(elem, int):
                    print(f"  elem[{j}]: int = {elem}")
                else:
                    print(f"  elem[{j}]: {type(elem).__name__} = {repr(elem)[:100]}")

        # 尝试每个 bytes 元素
        for j, elem in enumerate(raw):
            if not isinstance(elem, (bytes, bytearray)):
                continue

            # 尝试 Record
            try:
                rec = pb.Record()
                rec.ParseFromString(elem)
                fields = [f.name for f, _ in rec.ListFields()]
                if fields:
                    for f in fields:
                        field_counts[f"Record.{f}"] = field_counts.get(f"Record.{f}", 0) + 1
                    if i < 10:
                        print(f"  -> Record fields: {fields}")
                    if rec.HasField("history") and i < 5:
                        items = {kv.key: kv.value_json[:50] for kv in rec.history.item[:5]}
                        print(f"     history items: {items}")
                    continue
            except Exception:
                pass

            # 尝试 Result
            try:
                res = pb.Result()
                res.ParseFromString(elem)
                fields = [f.name for f, _ in res.ListFields()]
                if fields:
                    for f in fields:
                        field_counts[f"Result.{f}"] = field_counts.get(f"Result.{f}", 0) + 1
                    if i < 10:
                        print(f"  -> Result fields: {fields}")
                    continue
            except Exception:
                pass

            # 尝试 ServerRequest / 其他
            for msg_name in dir(pb):
                if msg_name.startswith('_'):
                    continue
                cls = getattr(pb, msg_name)
                if not hasattr(cls, 'ParseFromString'):
                    continue
                try:
                    obj = cls()
                    obj.ParseFromString(elem)
                    fields = [f.name for f, _ in obj.ListFields()]
                    if fields and len(fields) > 0:
                        if i < 5:
                            print(f"  -> {msg_name} fields: {fields}")
                        break
                except Exception:
                    continue
    else:
        if i < 3:
            print(f"Record {i}: type={type(raw).__name__}, repr={repr(raw)[:200]}")

print(f"\n=== Field frequency (first 200 records) ===")
for k, v in sorted(field_counts.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")

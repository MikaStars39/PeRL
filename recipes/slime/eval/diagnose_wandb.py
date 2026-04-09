#!/usr/bin/env python3
"""诊断 .wandb 二进制文件格式"""
import struct
import sys
import os

path = sys.argv[1]
file_size = os.path.getsize(path)
print(f"File: {path}")
print(f"Size: {file_size} bytes ({file_size/1024/1024:.1f} MB)\n")

with open(path, "rb") as f:
    # 打印前 64 字节的 hex dump
    head = f.read(256)
    print("First 256 bytes (hex):")
    for i in range(0, min(256, len(head)), 16):
        hex_str = " ".join(f"{b:02x}" for b in head[i:i+16])
        ascii_str = "".join(chr(b) if 32 <= b < 127 else "." for b in head[i:i+16])
        print(f"  {i:04x}: {hex_str:<48s} {ascii_str}")

    print()

    # 尝试不同的解析策略
    f.seek(0)

    # 策略1: 直接 u32 length + data
    print("=== Strategy 1: [u32 len][data] ===")
    f.seek(0)
    for i in range(5):
        pos = f.tell()
        hdr = f.read(4)
        if len(hdr) < 4: break
        size = struct.unpack("<I", hdr)[0]
        print(f"  record {i}: offset={pos}, declared_size={size}")
        if size > 10_000_000 or size == 0:
            print(f"    -> suspicious size, skipping")
            f.seek(pos + 1)  # try next byte
            continue
        data = f.read(size)
        print(f"    -> read {len(data)} bytes, first 20: {data[:20].hex()}")

    # 策略2: [u32 len][data][u32 crc]  (with CRC after each record)
    print("\n=== Strategy 2: [u32 len][data][u32 crc] ===")
    f.seek(0)
    for i in range(5):
        pos = f.tell()
        hdr = f.read(4)
        if len(hdr) < 4: break
        size = struct.unpack("<I", hdr)[0]
        if size > 10_000_000 or size == 0:
            print(f"  record {i}: offset={pos}, size={size} -> skip")
            f.seek(pos + 1)
            continue
        data = f.read(size)
        crc = f.read(4)
        print(f"  record {i}: offset={pos}, size={size}, data[:20]={data[:20].hex()}, crc={crc.hex() if crc else 'EOF'}")

    # 策略3: leveldb style [crc32(4)][length(2)][type(1)][data]
    print("\n=== Strategy 3: leveldb [u32 crc][u16 len][u8 type][data] ===")
    f.seek(0)
    for i in range(5):
        pos = f.tell()
        hdr = f.read(7)
        if len(hdr) < 7: break
        crc, length, rtype = struct.unpack("<IHB", hdr)
        print(f"  record {i}: offset={pos}, crc={crc:08x}, length={length}, type={rtype}")
        if length > 100000 or length == 0:
            f.seek(pos + 1)
            continue
        data = f.read(length)
        print(f"    -> data[:20]={data[:20].hex()}")

    # 策略4: 试试 wandb 自己的 reader
    print("\n=== Strategy 4: wandb built-in reader ===")
    try:
        from wandb.sdk.internal.datastore import DataStore
        ds = DataStore()
        ds.open_for_scan(path)
        count = 0
        for i in range(10):
            data = ds.scan_record()
            if data is None:
                print(f"  scan_record returned None at record {i}")
                break
            print(f"  record {i}: type={type(data).__name__}, len={len(data) if hasattr(data, '__len__') else '?'}")
            count += 1
        print(f"  Successfully read {count} records with DataStore")
    except ImportError:
        print("  DataStore not available")
    except Exception as e:
        print(f"  DataStore error: {e}")

    # 策略5: wandb RecordReader
    print("\n=== Strategy 5: wandb record module ===")
    try:
        from wandb.proto import wandb_internal_pb2 as pb
        from wandb.sdk.internal.datastore import DataStore
        ds = DataStore()
        ds.open_for_scan(path)

        history_count = 0
        other_count = 0
        for i in range(100):
            data = ds.scan_record()
            if data is None:
                break
            try:
                rec = pb.Record()
                rec.ParseFromString(data)
                if rec.HasField("history"):
                    history_count += 1
                    if history_count <= 3:
                        items = {kv.key: kv.value_json for kv in rec.history.item}
                        keys = list(items.keys())[:10]
                        print(f"  history record {history_count}: keys={keys}")
                else:
                    other_count += 1
                    # What field does it have?
                    fields = [f.name for f, v in rec.ListFields()]
                    if other_count <= 5:
                        print(f"  other record: fields={fields}")
            except Exception as e:
                if i < 5:
                    print(f"  parse error at record {i}: {e}")
        print(f"\n  Summary: {history_count} history + {other_count} other records (in first 100)")
    except Exception as e:
        print(f"  Error: {e}")

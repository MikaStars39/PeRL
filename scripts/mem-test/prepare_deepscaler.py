"""
从 ModelScope 下载 DeepScaleR-Preview-Dataset，
转换为 OpenR1 兼容格式，并保存到指定的 output-dir。

依赖：
  pip install modelscope datasets addict
"""

import os
import json
import argparse
from typing import Optional
from modelscope.msdatasets import MsDataset

def load_deepscaler_from_modelscope():
    """从 ModelScope 加载 DeepScaleR-Preview-Dataset。"""

    dataset_id = "agentica-org/DeepScaleR-Preview-Dataset"

    last_err: Optional[Exception] = None
    for split in ("train", "default", None):
        try:
            if split is None:
                ds = MsDataset.load(dataset_id)
            else:
                ds = MsDataset.load(dataset_id, split=split)

            print(
                f"[INFO] Loaded ModelScope dataset '{dataset_id}', "
                f"split={split!r}, size={len(ds)}"
            )
            return ds
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[WARN] Failed to load split={split!r}: {e}")

    raise RuntimeError(
        f"Failed to load {dataset_id} from ModelScope; last error: {last_err}"
    )


def build_openr1_example(example: dict, idx: int) -> dict:
    """
    把一条 DeepScaleR 样本转换成 OpenR1 样本。

    目标 schema（和你给的例子一致）：
      - prompt: string                -> 题目本身
      - solution: string              -> 最终答案（例如 "34"）
      - data_source: string           -> 数据来源（这里固定 "deepscaler_preview"）
      - source_prompt: list[dict]     -> [{"content": "...", "role": "user"}]
      - ability: string               -> 例如 "MATH"
      - reward_model: dict            -> {"ground_truth": "...", "style": "..."}
      - extra_info: dict              -> {"index": ... , "cot": "..."} 等
    """

    problem = (example.get("problem") or "").strip()
    final_answer = str(example.get("answer") or "").strip()
    cot_solution = (example.get("solution") or "").strip()  # deepscaler 里的推理

    # ----- 顶层字段 -----
    prompt = problem
    solution = final_answer        # 对齐你例子里的那行 "34"
    data_source = "deepscaler_preview"
    ability = "MATH"

    # ----- source_prompt -----
    # 模仿你给的那段：
    # "Solve the following math problem step by step. The last line ..."
    user_prompt = (
        "Solve the following math problem step by step. "
        "The last line of your response should be of the form "
        'Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n'
        f"{problem}\n\n"
        'Remember to put your answer on its own line after "Answer:".'
    )

    source_prompt = [
        {
            "content": user_prompt,
            "role": "user",
        }
    ]

    # ----- reward_model -----
    reward_model = {
        "ground_truth": final_answer,
        "style": "rule-lighteval/MATH_v2",
    }

    # ----- extra_info -----
    extra_info = {
        "index": str(idx),          # 你例子里是一个 uuid，这里用 index 字符串
        "cot": cot_solution,        # 保留原始 step-by-step 解答，方便以后用
        "raw_example": {
            # 如果你以后要 debug 原数据，可以多放一些字段
            "problem": problem,
            "answer": final_answer,
        },
    }

    return {
        "prompt": prompt,
        "solution": solution,
        "data_source": data_source,
        "source_prompt": source_prompt,
        "ability": ability,
        "reward_model": reward_model,
        "extra_info": extra_info,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert DeepScaleR-Preview-Dataset from ModelScope "
                    "into OpenR1-compatible JSONL."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="/root/data/openr1-deepscaler",
        help="输出数据集保存目录（默认 /root/data/openr1-deepscaler）",
    )

    parser.add_argument(
        "--max-sample",
        type=int,
        default=0,
        help="最多转换多少条样本；0 表示全部",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ds = load_deepscaler_from_modelscope()

    out_path = os.path.join(args.output_dir, "openr1_deepscaler.jsonl")
    converted = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for idx, ex in enumerate(ds):
            if args.max_sample and converted >= args.max_sample:
                break

            item = build_openr1_example(ex, idx)
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            converted += 1

            if converted % 1000 == 0:
                print(f"[INFO] Converted {converted} samples...")

    print(f"[DONE] Total converted: {converted}")
    print(f"[SAVED] Path: {out_path}")
    print(
        "[NOTE] 原始数据已缓存在 ModelScope cache 目录，"
        "这里只是做了格式转换写成 OpenR1 风格 JSONL。"
    )


if __name__ == "__main__":
    main()

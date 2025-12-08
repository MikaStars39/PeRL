import os
import json
import zipfile
import argparse
from pathlib import Path
from tqdm import tqdm


def main(result_dir: Path) -> None:
    # 输出文件名
    zip_output_name = result_dir / "all_result_files.zip"
    aggregated_output_name = result_dir / "aggregated_results.jsonl"

    # 用于存储 result.json 的汇总数据
    aggregated_data = []
    # 用于存储需要添加到压缩包的文件的相对路径
    files_to_add = []

    print(f"开始扫描目录: {result_dir}")

    for root, dirs, files in os.walk(result_dir):
        for file in files:
            full_path = os.path.join(root, file)

            # 计算相对路径，用于 zip 内部结构和记录
            rel_path = os.path.relpath(full_path, result_dir)

            # 1. 添加到 ZIP (保持相对路径)
            if file == "result.json" or file == "result.jsonl":
                files_to_add.append((full_path, rel_path))

            # 2. 如果是 result.json，提取特定字段
            if file == "result.json":
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        entry = {
                            "exp_name": Path(rel_path).parent.parent.name,
                            "dataset": f"{data.get('data_source', 'N/A')}@{data.get('rollout_n', 'N/A')}",
                            **data.get("summary", {}),
                        }
                        aggregated_data.append(entry)
                except Exception as e:
                    print(f"读取或解析 {rel_path} 时出错: {e}")

    # 保存汇总后的 jsonl 文件
    with aggregated_output_name.open("w", encoding="utf-8") as f:
        for entry in aggregated_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    files_to_add.append((aggregated_output_name, aggregated_output_name.name))

    # 任务：打印输出汇总数据
    # 要求：浮点数先乘以100然后保留两位小数输出，所有列都需要对其到相同宽度
    # 实现方案：
    # 1. 收集所有列名，优先展示 exp_name 和 dataset
    # 2. 计算每列所需的最大宽度（考虑表头和内容，特别是格式化后的浮点数）
    # 3. 逐行格式化输出
    if aggregated_data:
        # 1. 确定列名顺序
        all_keys = set().union(*(d.keys() for d in aggregated_data))
        # 优先显示的列
        priority_cols = ['exp_name', 'dataset']
        # 其他列按字母顺序排序
        other_cols = sorted([k for k in all_keys if k not in priority_cols])
        headers = [k for k in priority_cols if k in all_keys] + other_cols

        # 2. 计算每列最大宽度
        col_widths = {}
        for k in headers:
            # 初始宽度为表头长度
            max_w = len(k)
            for entry in aggregated_data:
                val = entry.get(k)
                if isinstance(val, float):
                    # 浮点数格式：*100 并保留2位小数
                    val_str = f"{val * 100:.2f}"
                else:
                    val_str = str(val)
                max_w = max(max_w, len(val_str))
            col_widths[k] = max_w + 4  # 增加一些间距

        # 3. 打印输出
        print("\n" + "=" * 80)
        print("汇总结果 (Result Summary):")
        
        # 打印表头
        header_str = "".join(k.ljust(col_widths[k]) for k in headers)
        print(header_str)
        print("-" * len(header_str))

        # 打印数据行
        for entry in aggregated_data:
            row_str = ""
            for k in headers:
                val = entry.get(k)
                if isinstance(val, float):
                    val_str = f"{val * 100:.2f}"
                else:
                    val_str = str(val)
                row_str += val_str.ljust(col_widths[k])
            print(row_str)
        print("=" * 80 + "\n")
    

    # 创建 ZIP 文件
    print("开始创建 ZIP 文件")
    with zipfile.ZipFile(zip_output_name, "w", zipfile.ZIP_DEFLATED) as zipf:
        for full_path, rel_path in tqdm(files_to_add, desc="Archiving"):
            zipf.write(full_path, arcname=rel_path)
    print(f"归档完成，已保存至: {zip_output_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dump all result.json files to a zip file and a csv file"
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        required=True,
        help="The directory to dump the result.json files",
    )
    args = parser.parse_args()

    main(Path(args.result_dir))

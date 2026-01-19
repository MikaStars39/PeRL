import os
import re
import ast
import csv
import sys

def parse_log_to_csv(folder_path):
    """
    Parses output.log in the specified folder and saves metrics to metrics.csv.
    Merges 'rollout' and 'train' data based on the step number.
    """
    log_file = os.path.join(folder_path, 'output.log')
    csv_file = os.path.join(folder_path, 'metrics.csv')

    if not os.path.exists(log_file):
        print(f"Error: {log_file} not found.")
        return

    # Dictionary to aggregate data by step: { step_number: {metric_dict} }
    aggregated_data = {}
    all_headers = set(['step'])

    print(f"Reading {log_file}...")
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Regex to find the step number (supports 'rollout 66:' or 'step 66:')
            step_match = re.search(r"(?:rollout|step)\s+(\d+):", line)
            # Regex to find the dictionary part of the log
            dict_match = re.search(r"(\{.*\})", line)

            if step_match and dict_match:
                try:
                    step = int(step_match.group(1))
                    # Safely evaluate the string dictionary to a Python dict
                    metrics = ast.literal_eval(dict_match.group(1))

                    if not isinstance(metrics, dict):
                        continue

                    # Initialize step entry if not exists
                    if step not in aggregated_data:
                        aggregated_data[step] = {'step': step}
                    
                    # Merge new metrics into the existing step record
                    aggregated_data[step].update(metrics)
                    all_headers.update(metrics.keys())

                except (ValueError, SyntaxError):
                    continue # Skip lines that look like matches but are malformed

    # Sort headers: 'step' first, then alphabetical
    sorted_headers = ['step'] + sorted([h for h in all_headers if h != 'step'])
    
    # Sort data rows by step number
    sorted_rows = sorted(aggregated_data.values(), key=lambda x: x['step'])

    # Write to CSV
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=sorted_headers)
        writer.writeheader()
        writer.writerows(sorted_rows)

    print(f"Success! Saved {len(sorted_rows)} rows to {csv_file}")

if __name__ == "__main__":
    # Use the provided argument as path, otherwise use current directory
    target_folder = sys.argv[1] if len(sys.argv) > 1 else "."
    parse_log_to_csv(target_folder)

"""
python /mnt/llm-train/users/explore-train/qingyu/PeRL/log/log_parser.py

"""
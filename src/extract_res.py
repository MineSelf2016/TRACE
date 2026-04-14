import os
import re
import ast
import csv

LOG_DIR = "logs"          # 日志目录
OUTPUT_CSV = "metrics.csv"

# 用于提取关键信息的正则
TIME_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
DATASET_PATTERN = re.compile(r"dataset=([^\s]+)")
METRICS_PATTERN = re.compile(r"metrics=(\{.*\})")

rows = []
all_metric_keys = set()

fnames = [
    'TCR-0_727510.out',
    'TCR-1_727519.out',
    'TCR-2_728099.out',
    'TCR-3_728100.out',
    'TCR-4_728101.out',
]

for fname in fnames:
    # if not fname.endswith((".log", ".txt")):
    #     continue

    filepath = os.path.join(LOG_DIR, fname)
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "metrics=" not in line:
                continue

            # 时间戳
            time_match = TIME_PATTERN.search(line)
            timestamp = time_match.group(0) if time_match else None

            # dataset
            dataset_match = DATASET_PATTERN.search(line)
            dataset = dataset_match.group(1) if dataset_match else None

            # metrics dict
            metrics_match = METRICS_PATTERN.search(line)
            if not metrics_match:
                continue

            try:
                metrics_dict = ast.literal_eval(metrics_match.group(1))
            except Exception as e:
                print(f"[WARN] Failed to parse metrics in {fname}: {e}")
                continue

            all_metric_keys.update(metrics_dict.keys())

            row = {
                "file": fname,
                # "timestamp": timestamp,
                "dataset": dataset,
                "frac": "",
            }
            row.update(metrics_dict)
            rows.append(row)

# 写 CSV
fieldnames = ["file", "dataset", "frac"] + sorted(all_metric_keys)

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print(f"Saved {len(rows)} rows to {OUTPUT_CSV}")

#!/usr/bin/env python3
"""Collect submission sweep results into a single CSV for paper writing.

It scans `multimodal_clip_binding_results/**/evaluation_*.csv` and
produces a tidy, merged table (long + wide forms).

Usage:
  conda activate spaVAE_env
  cd /project/zhiwei/cq5/PythonWorkSpace/TCR-submission
  python analysis_runs/collect_results.py --root ./multimodal_clip_binding_results --out ./analysis_runs/collected

Outputs:
- collected_long.csv  (one row per metric)
- collected_wide.csv  (one row per run; AUROC/AUPR/etc as columns)

Notes:
- `train.py` writes `run_meta.json` in each run dir; we merge it when present.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None


def _safe_read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./multimodal_clip_binding_results", help="Results root to scan")
    ap.add_argument("--out", default="./analysis_runs/collected", help="Output directory")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_files = sorted(root.glob("**/evaluation_*.csv"))
    if not eval_files:
        raise SystemExit(f"No evaluation_*.csv found under {root}")

    long_path = out_dir / "collected_long.csv"
    wide_path = out_dir / "collected_wide.csv"

    if pd is not None:
        frames = []
        for fp in eval_files:
            df = pd.read_csv(fp)
            df["run_dir"] = str(fp.parent)
            df["eval_file"] = str(fp)

            meta = _safe_read_json(fp.parent / "run_meta.json")
            if meta is not None:
                # flatten a few useful fields
                df["run_name"] = meta.get("run_name")
                df["seed"] = meta.get("seed")
                df["cmd"] = meta.get("cmd")
                df["timestamp"] = meta.get("timestamp")
            frames.append(df)

        long_df = pd.concat(frames, ignore_index=True)

        # Wide pivot: one row per run_dir (+ key params), columns are metrics.
        id_cols = [
            c
            for c in [
                "run_dir",
                "run_name",
                "experiment",
                "seed",
                "proj_dim",
                "graph_hidden",
                "graph_layers",
                "dropout",
                "learning_rate",
                "batch_size",
                "weight_decay",
                "optimizer",
                "lambda_clip",
                "lambda_bind",
                "temperature",
                "edge_dropout",
            ]
            if c in long_df.columns
        ]

        wide_df = long_df.pivot_table(
            index=id_cols,
            columns="metrics",
            values="score",
            aggfunc="first",
        ).reset_index()

        long_df.to_csv(long_path, index=False)
        wide_df.to_csv(wide_path, index=False)
        print(f"Wrote {long_path}")
        print(f"Wrote {wide_path}")
        return

    # Fallback: stdlib-only collection (works outside conda env).
    rows: list[dict[str, str]] = []
    for fp in eval_files:
        with fp.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                r = dict(r)
                r["run_dir"] = str(fp.parent)
                r["eval_file"] = str(fp)
                meta = _safe_read_json(fp.parent / "run_meta.json")
                if meta is not None:
                    r.setdefault("run_name", str(meta.get("run_name", "")))
                    r.setdefault("seed", str(meta.get("seed", "")))
                    r.setdefault("cmd", str(meta.get("cmd", "")))
                    r.setdefault("timestamp", str(meta.get("timestamp", "")))
                rows.append(r)

    if not rows:
        raise SystemExit(f"No rows found in evaluation CSVs under {root}")

    # Write long CSV
    long_cols = sorted({k for r in rows for k in r.keys()})
    with long_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=long_cols)
        w.writeheader()
        w.writerows(rows)

    # Build wide rows keyed by run_dir
    preferred_id_cols = [
        "run_dir",
        "run_name",
        "experiment",
        "seed",
        "proj_dim",
        "graph_hidden",
        "graph_layers",
        "dropout",
        "learning_rate",
        "batch_size",
        "weight_decay",
        "optimizer",
        "lambda_clip",
        "lambda_bind",
        "temperature",
        "edge_dropout",
    ]
    present_id_cols = [c for c in preferred_id_cols if any(c in r for r in rows)]

    metric_names = sorted({r.get("metrics", "") for r in rows if r.get("metrics")})
    wide_cols = present_id_cols + metric_names

    by_run: dict[str, dict[str, str]] = {}
    for r in rows:
        run_dir = r.get("run_dir", "")
        if not run_dir:
            continue
        d = by_run.setdefault(run_dir, {c: r.get(c, "") for c in present_id_cols})
        m = r.get("metrics")
        s = r.get("score")
        if m:
            # keep first score if duplicated
            d.setdefault(m, s if s is not None else "")

    with wide_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=wide_cols)
        w.writeheader()
        for _, d in sorted(by_run.items()):
            w.writerow({c: d.get(c, "") for c in wide_cols})

    print(f"Wrote {long_path} (stdlib mode; pandas not available)")
    print(f"Wrote {wide_path} (stdlib mode; pandas not available)")


if __name__ == "__main__":
    main()

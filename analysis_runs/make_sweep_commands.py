#!/usr/bin/env python3
"""Generate (but do not run) sweep commands for submission evidence-chain experiments.

Design goal:
- Produce a small set of controlled experiments that support the
  "failure (no alignment) -> fix (alignment)" story.
- Do not run anything by default; just write shell scripts.

Usage examples:
  python analysis_runs/make_sweep_commands.py --mode only-sampled-negs --dataset-index 1
  python analysis_runs/make_sweep_commands.py --mode only-sampled-negs --dataset-index 1 --write ./analysis_runs/noise_sweep.sh

Then run later (manually):
  bash ./analysis_runs/noise_sweep.sh

Notes:
- This assumes you will `conda activate spaVAE_env` before running.
- train.py already supports:
    --lambda_clip  (set to 0.0 for no-alignment)
    --edge_dropout (training-only graph noise)
    --pos_downsample / --neg_downsample (supervision sweeps)
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _cmd(*, run_name: str, align: bool, mode: str, dataset_index: int, edge_dropout: float, pos_downsample: float, extra: str) -> str:
    lambda_clip = 0.2 if align else 0.0
    name = "align" if align else "noalign"
    parts = [
        "python train.py",
        f"--mode {mode}",
        f"--dataset_index {dataset_index}",
        f"--lambda_clip {lambda_clip}",
        f"--edge_dropout {edge_dropout}",
        f"--pos_downsample {pos_downsample}",
        "--neg_downsample 1.0",
        f"--results_dir ./multimodal_clip_binding_results/{name}",
        f"--run_name {run_name}",
    ]
    if extra:
        parts.append(extra.strip())
    return " ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="only-sampled-negs")
    ap.add_argument("--dataset-index", type=int, default=1)
    ap.add_argument("--write", default="./analysis_runs/sweep_commands.sh")

    # Noise sweep: training-time edge dropout.
    ap.add_argument("--edge-dropouts", default="0.0,0.1,0.2,0.3,0.4")

    # Supervision sweep: downsample positives only (keep negatives).
    ap.add_argument("--pos-downsamples", default="1.0,0.5,0.2,0.1")

    ap.add_argument(
        "--extra",
        default="",
        help=(
            "Extra args appended verbatim to each command. "
            "Recommended for faster CPU sweep: --num_epochs 30 --patience 10 --num_workers 4"
        ),
    )

    args = ap.parse_args()

    edge_dropouts = [float(x) for x in args.edge_dropouts.split(",") if x.strip()]
    pos_downsamples = [float(x) for x in args.pos_downsamples.split(",") if x.strip()]

    out_path = Path(args.write)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("#!/usr/bin/env bash")
    lines.append("set -euo pipefail")
    lines.append("")
    lines.append("# Activate env first: conda activate spaVAE_env")
    lines.append("")

    # 1) Noise sweep: for each dropout level, compare align vs noalign.
    lines.append("# ------------------------------")
    lines.append("# 1) Noise sweep (edge dropout)")
    lines.append("# ------------------------------")
    for p in edge_dropouts:
        lines.append(f"# edge_dropout={p}")
        p_name = str(p).replace(".", "p")
        lines.append(_cmd(run_name=f"noise_ed{p_name}_noalign", align=False, mode=args.mode, dataset_index=args.dataset_index, edge_dropout=p, pos_downsample=1.0, extra=args.extra))
        lines.append(_cmd(run_name=f"noise_ed{p_name}_align",   align=True,  mode=args.mode, dataset_index=args.dataset_index, edge_dropout=p, pos_downsample=1.0, extra=args.extra))
        lines.append("")

    # 2) Supervision sweep: for each pos fraction, compare align vs noalign.
    lines.append("# ------------------------------")
    lines.append("# 2) Supervision sweep (pos downsample)")
    lines.append("# ------------------------------")
    for frac in pos_downsamples:
        lines.append(f"# pos_downsample={frac}")
        frac_name = str(frac).replace(".", "p")
        lines.append(_cmd(run_name=f"sup_pos{frac_name}_noalign", align=False, mode=args.mode, dataset_index=args.dataset_index, edge_dropout=0.0, pos_downsample=frac, extra=args.extra))
        lines.append(_cmd(run_name=f"sup_pos{frac_name}_align",   align=True,  mode=args.mode, dataset_index=args.dataset_index, edge_dropout=0.0, pos_downsample=frac, extra=args.extra))
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n")
    out_path.chmod(0o755)

    print(f"Wrote {out_path} with {len([l for l in lines if l.startswith('python ')])} commands")


if __name__ == "__main__":
    main()

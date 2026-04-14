#!/usr/bin/env python3
"""Plot t-SNE/UMAP for Alignment Geometry outputs.

Reads CSVs produced by analysis_runs/align_geometry.py and renders PDFs
with the paper palette. If UMAP CSV is unavailable, only plots t-SNE.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path to import analysis.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis import set_paper_style, PALETTE


def _scatter(df: pd.DataFrame, title: str, out_pdf: Path):
    set_paper_style()
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    colors = {0: PALETTE["neg"], 1: PALETTE["pos"]}
    for label in [0, 1]:
        sub = df[df["label"] == label]
        ax.scatter(sub["x"], sub["y"], s=8, alpha=0.85, linewidths=0, color=colors[label], label=f"{'Pos' if label==1 else 'Neg'} (n={len(sub)})")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_title(title)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser("Plot t-SNE/UMAP PDFs from align_geometry outputs")
    ap.add_argument("--indir", default="analysis_runs/align_geometry", help="Directory containing fused_tsne.csv and optional fused_umap.csv")
    ap.add_argument("--outdir", default="analysis_runs/align_geometry", help="Directory to write PDFs")
    ap.add_argument("--to_overleaf", action="store_true", help="Also write PDFs to ../Overleaf/TCR---Bioinformatics/figures")
    args = ap.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tsne_csv = indir / "fused_tsne.csv"
    umap_csv = indir / "fused_umap.csv"

    if tsne_csv.exists():
        df = pd.read_csv(tsne_csv)
        _scatter(df, "Fused embedding (t-SNE)", outdir / "fused_tsne.pdf")
        print("✓ Saved:", outdir / "fused_tsne.pdf")
    else:
        print("⚠ t-SNE CSV not found:", tsne_csv)

    if umap_csv.exists():
        df = pd.read_csv(umap_csv)
        _scatter(df, "Fused embedding (UMAP)", outdir / "fused_umap.pdf")
        print("✓ Saved:", outdir / "fused_umap.pdf")
    else:
        print("ℹ UMAP CSV not found (optional):", umap_csv)

    if args.to_overleaf:
        ov = Path("../Overleaf/TCR---Bioinformatics/figures")
        ov.mkdir(parents=True, exist_ok=True)
        for name in ["fused_tsne.pdf", "fused_umap.pdf"]:
            src = outdir / name
            if src.exists():
                dst = ov / name
                dst.write_bytes(src.read_bytes())
                print("→ Copied to Overleaf:", dst)


if __name__ == "__main__":
    main()

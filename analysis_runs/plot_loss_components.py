#!/usr/bin/env python3
"""Plot loss components over epochs from loss_components.csv.

Renders a paper-ready PDF with curves for binding loss, CLIP total,
and CLIP per-view (TCR, peptide). Use with outputs from train.py.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path to import analysis.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis import set_paper_style, PALETTE


def main() -> None:
    ap = argparse.ArgumentParser("Plot training loss components")
    ap.add_argument("--components", required=True, help="Path to loss_components.csv in a run output_dir")
    ap.add_argument("--out", default=None, help="Output PDF path (defaults next to CSV)")
    ap.add_argument("--to_overleaf", action="store_true", help="Also copy PDF to ../Overleaf/TCR---Bioinformatics/figures")
    args = ap.parse_args()

    path = Path(args.components)
    df = pd.read_csv(path)

    set_paper_style()
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4))
    ax.plot(df["epoch"], df["bind"], label="Binding CE", color=PALETTE["pos"], linewidth=1.8)
    ax.plot(df["epoch"], df["clip"], label="CLIP total", color=PALETTE["accent"], linewidth=1.8)
    ax.plot(df["epoch"], df["clip_tcr"], label="CLIP TCR", color=PALETTE["tcr"], linewidth=1.6, linestyle="--")
    ax.plot(df["epoch"], df["clip_pep"], label="CLIP Peptide", color=PALETTE["pep"], linewidth=1.6, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average loss (per-sample)")
    ax.set_title("Training loss components")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()

    out_pdf = Path(args.out) if args.out else path.with_name("loss_components.pdf")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print("✓ Saved:", out_pdf)

    if args.to_overleaf:
        ov = Path("../Overleaf/TCR---Bioinformatics/figures")
        ov.mkdir(parents=True, exist_ok=True)
        dst = ov / out_pdf.name
        dst.write_bytes(out_pdf.read_bytes())
        print("→ Copied to Overleaf:", dst)


if __name__ == "__main__":
    main()

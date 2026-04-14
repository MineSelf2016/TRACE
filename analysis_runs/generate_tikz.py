#!/usr/bin/env python3
"""Generate simple ASCII/Unicode plots and LaTeX TikZ code for figures."""

import csv
from pathlib import Path

# Load data
csv_path = Path('analysis_runs/collected_final/collected_wide.csv')
rows = list(csv.DictReader(csv_path.open()))

sweep_rows = [r for r in rows if 'demo' not in r.get('run_name', '')]
align_rows = [r for r in sweep_rows if 'align' in r.get('run_name', '') and 'noalign' not in r.get('run_name', '')]
noalign_rows = [r for r in sweep_rows if 'noalign' in r.get('run_name', '')]

out_dir = Path('../Overleaf/TCR---Bioinformatics/figures')
out_dir.mkdir(parents=True, exist_ok=True)

# Extract noise sweep data
noise_configs = [0.0, 0.1, 0.2, 0.3, 0.4]
noise_data = []
for ed in noise_configs:
    ed_str = str(ed)
    noalign_row = [r for r in noalign_rows if r.get('edge_dropout') == ed_str and 'noise' in r.get('run_name', '')]
    align_row = [r for r in align_rows if r.get('edge_dropout') == ed_str and 'noise' in r.get('run_name', '')]
    if noalign_row and align_row:
        na = float(noalign_row[0].get('AUROC', 0))
        a = float(align_row[0].get('AUROC', 0))
        noise_data.append((ed, na, a))

# Extract supervision sweep data
sup_configs = [0.1, 0.2, 0.5, 1.0]
sup_data = []
for frac in sup_configs:
    frac_name = str(frac).replace('.', 'p')
    noalign_row = [r for r in noalign_rows if f'pos{frac_name}' in r.get('run_name', '')]
    align_row = [r for r in align_rows if f'pos{frac_name}' in r.get('run_name', '')]
    if noalign_row and align_row:
        na = float(noalign_row[0].get('AUROC', 0))
        a = float(align_row[0].get('AUROC', 0))
        sup_data.append((frac, na, a))

# ============================================================
# Generate LaTeX TikZ figures
# ============================================================

tikz_noise = r"""\begin{tikzpicture}
\begin{axis}[
    width=0.48\textwidth,
    height=6cm,
    xlabel={Edge Dropout Rate},
    ylabel={Test AUROC},
    title={Effect of Training-Time Graph Noise},
    legend pos=south west,
    grid=major,
    ymin=0.48, ymax=0.57,
    xmin=-0.05, xmax=0.45,
    xtick={0, 0.1, 0.2, 0.3, 0.4},
    ymajorgrids=true,
    mark size=3pt
]

% No Alignment
\addplot[color=red, mark=o, thick] coordinates {
"""
for ed, na, _ in noise_data:
    tikz_noise += f"    ({ed}, {na:.4f})\n"
tikz_noise += r"""};
\addlegendentry{No Alignment}

% With Alignment
\addplot[color=green!70!black, mark=square, thick] coordinates {
"""
for ed, _, a in noise_data:
    tikz_noise += f"    ({ed}, {a:.4f})\n"
tikz_noise += r"""};
\addlegendentry{With Alignment}

% Random baseline
\addplot[color=gray, dashed, domain=0:0.4] {0.5};
\addlegendentry{Random}

\end{axis}
\end{tikzpicture}"""

tikz_sup = r"""\begin{tikzpicture}
\begin{axis}[
    width=0.48\textwidth,
    height=6cm,
    xlabel={Fraction of Positive Labels},
    ylabel={Test AUROC},
    title={Effect of Data Scarcity},
    legend pos=south west,
    grid=major,
    ymin=0.48, ymax=0.57,
    xmode=log,
    xmin=0.08, xmax=1.2,
    xtick={0.1, 0.2, 0.5, 1.0},
    xticklabels={10\%, 20\%, 50\%, 100\%},
    ymajorgrids=true,
    mark size=3pt
]

% No Alignment
\addplot[color=red, mark=o, thick] coordinates {
"""
for frac, na, _ in sup_data:
    tikz_sup += f"    ({frac}, {na:.4f})\n"
tikz_sup += r"""};
\addlegendentry{No Alignment}

% With Alignment
\addplot[color=green!70!black, mark=square, thick] coordinates {
"""
for frac, _, a in sup_data:
    tikz_sup += f"    ({frac}, {a:.4f})\n"
tikz_sup += r"""};
\addlegendentry{With Alignment}

% Random baseline
\addplot[color=gray, dashed, domain=0.1:1.0] {0.5};
\addlegendentry{Random}

\end{axis}
\end{tikzpicture}"""

# Save TikZ code
with open(out_dir / 'noise_sweep_tikz.tex', 'w') as f:
    f.write(tikz_noise)
print(f"✓ Saved: {out_dir / 'noise_sweep_tikz.tex'}")

with open(out_dir / 'supervision_sweep_tikz.tex', 'w') as f:
    f.write(tikz_sup)
print(f"✓ Saved: {out_dir / 'supervision_sweep_tikz.tex'}")

# ============================================================
# Generate combined figure LaTeX code
# ============================================================
tikz_combined = r"""\begin{figure*}[t]
\centering
\begin{subfigure}{0.48\textwidth}
""" + tikz_noise + r"""
\caption{Training-time graph noise}
\label{fig:noise_sweep}
\end{subfigure}
\hfill
\begin{subfigure}{0.48\textwidth}
""" + tikz_sup + r"""
\caption{Data scarcity}
\label{fig:sup_sweep}
\end{subfigure}
\caption{Controlled sweeps demonstrating the necessity of alignment. Without alignment (red circles), performance collapses to near-random (~0.505 AUROC) regardless of noise or supervision level. With alignment (green squares), performance is consistently elevated (0.53--0.55 AUROC) across all conditions.}
\label{fig:sweeps}
\end{figure*}"""

with open(out_dir / 'combined_sweeps_tikz.tex', 'w') as f:
    f.write(tikz_combined)
print(f"✓ Saved: {out_dir / 'combined_sweeps_tikz.tex'}")

# ============================================================
# Print ASCII visualization
# ============================================================
print("\n" + "="*70)
print("NOISE SWEEP (Edge Dropout vs AUROC)")
print("="*70)
print(f"{'Dropout':<10} {'No Align':<12} {'With Align':<12} {'Delta':<10}")
print("-"*70)
for ed, na, a in noise_data:
    delta = a - na
    bar_na = '█' * int(na * 100)
    bar_a = '█' * int(a * 100)
    print(f"{ed:<10.1f} {na:<12.4f} {a:<12.4f} +{delta:<9.4f}")
print()

print("="*70)
print("SUPERVISION SWEEP (Positive Fraction vs AUROC)")
print("="*70)
print(f"{'Pos Frac':<10} {'No Align':<12} {'With Align':<12} {'Delta':<10}")
print("-"*70)
for frac, na, a in sup_data:
    delta = a - na
    print(f"{frac:<10.1f} {na:<12.4f} {a:<12.4f} +{delta:<9.4f}")

print("\n" + "="*70)
print("✓ TikZ figures ready for LaTeX!")
print(f"  Add to your paper: \\input{{figures/combined_sweeps_tikz.tex}}")
print(f"  Or use individual files:")
print(f"    - figures/noise_sweep_tikz.tex")
print(f"    - figures/supervision_sweep_tikz.tex")
print("="*70)

#!/usr/bin/env python3
"""Generate publication-quality figures for submission paper."""

import csv
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not available. Will generate gnuplot scripts instead.")

# Set publication-quality defaults
if HAS_MPL:
    mpl.rcParams['font.size'] = 11
    mpl.rcParams['axes.labelsize'] = 12
    mpl.rcParams['axes.titlesize'] = 13
    mpl.rcParams['xtick.labelsize'] = 10
    mpl.rcParams['ytick.labelsize'] = 10
    mpl.rcParams['legend.fontsize'] = 10
    mpl.rcParams['figure.dpi'] = 300
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
else:
    print("Matplotlib not available; will generate data files only.")

# Load data
csv_path = Path('analysis_runs/collected_final/collected_wide.csv')
rows = list(csv.DictReader(csv_path.open()))

# Filter sweep runs (exclude demo)
sweep_rows = [r for r in rows if 'demo' not in r.get('run_name', '')]
align_rows = [r for r in sweep_rows if 'align' in r.get('run_name', '') and 'noalign' not in r.get('run_name', '')]
noalign_rows = [r for r in sweep_rows if 'noalign' in r.get('run_name', '')]

# Output directory
out_dir = Path('../Overleaf/TCR---Bioinformatics/figures')
out_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# Figure 1: Noise Sweep (Edge Dropout)
# ============================================================
noise_configs = [0.0, 0.1, 0.2, 0.3, 0.4]
noise_noalign_auroc = []
noise_align_auroc = []

for ed in noise_configs:
    ed_str = str(ed)
    noalign_row = [r for r in noalign_rows if r.get('edge_dropout') == ed_str and 'noise' in r.get('run_name', '')]
    align_row = [r for r in align_rows if r.get('edge_dropout') == ed_str and 'noise' in r.get('run_name', '')]
    
    if noalign_row and align_row:
        noise_noalign_auroc.append(float(noalign_row[0].get('AUROC', 0)))
        noise_align_auroc.append(float(align_row[0].get('AUROC', 0)))

# Save data to file for external plotting
data_dir = Path('analysis_runs/plot_data')
data_dir.mkdir(parents=True, exist_ok=True)
with open(data_dir / 'noise_sweep.dat', 'w') as f:
    f.write('# edge_dropout noalign_auroc align_auroc\n')
    for ed, na, a in zip(noise_configs, noise_noalign_auroc, noise_align_auroc):
        f.write(f'{ed} {na:.6f} {a:.6f}\n')
print(f"Saved: {data_dir / 'noise_sweep.dat'}")

if not HAS_MPL:
    print("Skipping matplotlib plots (not installed)")
else:
ax.plot(noise_configs, noise_noalign_auroc, 'o-', color='#d62728', linewidth=2.5, 
        markersize=8, label='No Alignment', alpha=0.9)
ax.plot(noise_configs, noise_align_auroc, 's-', color='#2ca02c', linewidth=2.5, 
        markersize=8, label='With Alignment', alpha=0.9)
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Random')
ax.set_xlabel('Edge Dropout Rate', fontweight='bold')
ax.set_ylabel('Test AUROC', fontweight='bold')
ax.set_title('Effect of Training-Time Graph Noise', fontweight='bold', pad=15)
ax.legend(loc='best', framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle=':')
ax.set_ylim([0.48, 0.57])
plt.tight_layout()
plt.savefig(out_dir / 'noise_sweep.pdf', bbox_inches='tight')
plt.savefig(out_dir / 'noise_sweep.png', bbox_inches='tight')
print(f"Saved: {out_dir / 'noise_sweep.pdf'}")
plt.close()

# ============================================================
# Figure 2: Supervision Sweep (Positive Downsampling)
# ============================================================
sup_configs = [0.1, 0.2, 0.5, 1.0]
sup_noalign_auroc = []
sup_align_auroc = []

for frac in sup_configs:
    frac_name = str(frac).replace('.', 'p')
    noalign_row = [r for r in noalign_rows if f'pos{frac_name}' in r.get('run_name', '')]
    align_row = [r for r in align_rows if f'pos{frac_name}' in r.get('run_name', '')]
    
    if noalign_row and align_row:
        sup_noalign_auroc.append(float(noalign_row[0].get('AUROC', 0)))
        sup_align_auroc.append(float(align_row[0].get('AUROC', 0)))

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(sup_configs, sup_noalign_auroc, 'o-', color='#d62728', linewidth=2.5, 
        markersize=8, label='No Alignment', alpha=0.9)
ax.plot(sup_configs, sup_align_auroc, 's-', color='#2ca02c', linewidth=2.5, 
        markersize=8, label='With Alignment', alpha=0.9)
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Random')
ax.set_xlabel('Fraction of Positive Training Labels', fontweight='bold')
ax.set_ylabel('Test AUROC', fontweight='bold')
ax.set_title('Effect of Data Scarcity', fontweight='bold', pad=15)
ax.legend(loc='best', framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle=':')
ax.set_ylim([0.48, 0.57])
ax.set_xscale('log')
ax.set_xticks(sup_configs)
ax.set_xticklabels(['10%', '20%', '50%', '100%'])
plt.tight_layout()
plt.savefig(out_dir / 'supervision_sweep.pdf', bbox_inches='tight')
plt.savefig(out_dir / 'supervision_sweep.png', bbox_inches='tight')
print(f"Saved: {out_dir / 'supervision_sweep.pdf'}")
plt.close()

# ============================================================
# Figure 3: Combined view (2x1 subplots)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Left: Noise sweep
ax1.plot(noise_configs, noise_noalign_auroc, 'o-', color='#d62728', linewidth=2.5, 
         markersize=8, label='No Alignment', alpha=0.9)
ax1.plot(noise_configs, noise_align_auroc, 's-', color='#2ca02c', linewidth=2.5, 
         markersize=8, label='With Alignment', alpha=0.9)
ax1.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Random')
ax1.set_xlabel('Edge Dropout Rate', fontweight='bold')
ax1.set_ylabel('Test AUROC', fontweight='bold')
ax1.set_title('(a) Training-Time Graph Noise', fontweight='bold', pad=10)
ax1.legend(loc='best', framealpha=0.95, fontsize=9)
ax1.grid(True, alpha=0.3, linestyle=':')
ax1.set_ylim([0.48, 0.57])

# Right: Supervision sweep
ax2.plot(sup_configs, sup_noalign_auroc, 'o-', color='#d62728', linewidth=2.5, 
         markersize=8, label='No Alignment', alpha=0.9)
ax2.plot(sup_configs, sup_align_auroc, 's-', color='#2ca02c', linewidth=2.5, 
         markersize=8, label='With Alignment', alpha=0.9)
ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Random')
ax2.set_xlabel('Fraction of Positive Training Labels', fontweight='bold')
ax2.set_ylabel('Test AUROC', fontweight='bold')
ax2.set_title('(b) Data Scarcity', fontweight='bold', pad=10)
ax2.legend(loc='best', framealpha=0.95, fontsize=9)
ax2.grid(True, alpha=0.3, linestyle=':')
ax2.set_ylim([0.48, 0.57])
ax2.set_xscale('log')
ax2.set_xticks(sup_configs)
ax2.set_xticklabels(['10%', '20%', '50%', '100%'])

plt.tight_layout()
plt.savefig(out_dir / 'combined_sweeps.pdf', bbox_inches='tight')
plt.savefig(out_dir / 'combined_sweeps.png', bbox_inches='tight')
print(f"Saved: {out_dir / 'combined_sweeps.pdf'}")
plt.close()

# ============================================================
# Figure 4: Delta AUROC Bar Chart
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Noise deltas
noise_deltas = [a - n for a, n in zip(noise_align_auroc, noise_noalign_auroc)]
noise_labels = [f'{x:.1f}' for x in noise_configs]
colors_noise = ['#2ca02c' if d > 0 else '#d62728' for d in noise_deltas]
bars1 = ax1.bar(noise_labels, noise_deltas, color=colors_noise, alpha=0.8, edgecolor='black', linewidth=1.2)
ax1.axhline(y=0, color='black', linewidth=1)
ax1.set_xlabel('Edge Dropout Rate', fontweight='bold')
ax1.set_ylabel('Δ AUROC (Alignment - No Alignment)', fontweight='bold')
ax1.set_title('(a) Improvement from Alignment: Noise Sweep', fontweight='bold', pad=10)
ax1.grid(True, alpha=0.3, axis='y', linestyle=':')
ax1.set_ylim([-0.01, 0.05])

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
             f'+{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Supervision deltas
sup_deltas = [a - n for a, n in zip(sup_align_auroc, sup_noalign_auroc)]
sup_labels = ['10%', '20%', '50%', '100%']
colors_sup = ['#2ca02c' if d > 0 else '#d62728' for d in sup_deltas]
bars2 = ax2.bar(sup_labels, sup_deltas, color=colors_sup, alpha=0.8, edgecolor='black', linewidth=1.2)
ax2.axhline(y=0, color='black', linewidth=1)
ax2.set_xlabel('Fraction of Positive Training Labels', fontweight='bold')
ax2.set_ylabel('Δ AUROC (Alignment - No Alignment)', fontweight='bold')
ax2.set_title('(b) Improvement from Alignment: Data Scarcity', fontweight='bold', pad=10)
ax2.grid(True, alpha=0.3, axis='y', linestyle=':')
ax2.set_ylim([-0.01, 0.05])

# Add value labels on bars
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
             f'+{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(out_dir / 'delta_auroc_bars.pdf', bbox_inches='tight')
plt.savefig(out_dir / 'delta_auroc_bars.png', bbox_inches='tight')
print(f"Saved: {out_dir / 'delta_auroc_bars.pdf'}")
plt.close()

print("\n✓ All figures generated successfully!")
print(f"  Output directory: {out_dir}")
print("  Files: noise_sweep.pdf, supervision_sweep.pdf, combined_sweeps.pdf, delta_auroc_bars.pdf")

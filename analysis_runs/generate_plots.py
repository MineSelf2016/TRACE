#!/usr/bin/env python3
"""Generate publication-quality figures for submission paper - Simple version."""

import csv
from pathlib import Path
import subprocess
import sys

# Load data
csv_path = Path('analysis_runs/collected_final/collected_wide.csv')
rows = list(csv.DictReader(csv_path.open()))

# Filter sweep runs (exclude demo)
sweep_rows = [r for r in rows if 'demo' not in r.get('run_name', '')]
align_rows = [r for r in sweep_rows if 'align' in r.get('run_name', '') and 'noalign' not in r.get('run_name', '')]
noalign_rows = [r for r in sweep_rows if 'noalign' in r.get('run_name', '')]

# Output directories
out_dir = Path('../Overleaf/TCR---Bioinformatics/figures')
out_dir.mkdir(parents=True, exist_ok=True)
data_dir = Path('analysis_runs/plot_data')
data_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# Extract data for Noise Sweep
# ============================================================
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

# Save noise sweep data
with open(data_dir / 'noise_sweep.dat', 'w') as f:
    f.write('# edge_dropout noalign_auroc align_auroc delta\n')
    for ed, na, a in noise_data:
        f.write(f'{ed} {na:.6f} {a:.6f} {a-na:.6f}\n')
print(f"✓ Saved: {data_dir / 'noise_sweep.dat'}")

# ============================================================
# Extract data for Supervision Sweep
# ============================================================
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

# Save supervision sweep data
with open(data_dir / 'supervision_sweep.dat', 'w') as f:
    f.write('# pos_fraction noalign_auroc align_auroc delta\n')
    for frac, na, a in sup_data:
        f.write(f'{frac} {na:.6f} {a:.6f} {a-na:.6f}\n')
print(f"✓ Saved: {data_dir / 'supervision_sweep.dat'}")

# ============================================================
# Generate gnuplot scripts
# ============================================================
gnuplot_noise = f"""#!/usr/bin/gnuplot
set terminal pdfcairo enhanced color font 'Arial,11' size 6,4
set output '{out_dir}/noise_sweep.pdf'
set style line 1 lc rgb '#d62728' lt 1 lw 2.5 pt 7 ps 1.2
set style line 2 lc rgb '#2ca02c' lt 1 lw 2.5 pt 5 ps 1.2
set style line 3 lc rgb 'gray' lt 2 lw 1 dt 2
set xlabel 'Edge Dropout Rate' font 'Arial-Bold,12'
set ylabel 'Test AUROC' font 'Arial-Bold,12'
set title 'Effect of Training-Time Graph Noise' font 'Arial-Bold,13'
set key top right
set grid ytics linestyle 0 linewidth 0.5 linecolor rgb '#dddddd'
set yrange [0.48:0.57]
set xtics 0.1
plot '{data_dir}/noise_sweep.dat' using 1:2 with linespoints ls 1 title 'No Alignment', \\
     '' using 1:3 with linespoints ls 2 title 'With Alignment', \\
     0.5 with lines ls 3 title 'Random'
"""

with open(data_dir / 'plot_noise.gp', 'w') as f:
    f.write(gnuplot_noise)
print(f"✓ Saved: {data_dir / 'plot_noise.gp'}")

gnuplot_sup = f"""#!/usr/bin/gnuplot
set terminal pdfcairo enhanced color font 'Arial,11' size 6,4
set output '{out_dir}/supervision_sweep.pdf'
set style line 1 lc rgb '#d62728' lt 1 lw 2.5 pt 7 ps 1.2
set style line 2 lc rgb '#2ca02c' lt 1 lw 2.5 pt 5 ps 1.2
set style line 3 lc rgb 'gray' lt 2 lw 1 dt 2
set xlabel 'Fraction of Positive Training Labels' font 'Arial-Bold,12'
set ylabel 'Test AUROC' font 'Arial-Bold,12'
set title 'Effect of Data Scarcity' font 'Arial-Bold,13'
set key top right
set grid ytics linestyle 0 linewidth 0.5 linecolor rgb '#dddddd'
set yrange [0.48:0.57]
set logscale x
set xtics ('10%%' 0.1, '20%%' 0.2, '50%%' 0.5, '100%%' 1.0)
plot '{data_dir}/supervision_sweep.dat' using 1:2 with linespoints ls 1 title 'No Alignment', \\
     '' using 1:3 with linespoints ls 2 title 'With Alignment', \\
     0.5 with lines ls 3 title 'Random'
"""

with open(data_dir / 'plot_supervision.gp', 'w') as f:
    f.write(gnuplot_sup)
print(f"✓ Saved: {data_dir / 'plot_supervision.gp'}")

# Try to run gnuplot if available
try:
    result = subprocess.run(['gnuplot', '--version'], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"\\n✓ gnuplot found: {result.stdout.strip()}")
        print("  Generating plots...")
        subprocess.run(['gnuplot', str(data_dir / 'plot_noise.gp')], check=True)
        print(f"  → {out_dir / 'noise_sweep.pdf'}")
        subprocess.run(['gnuplot', str(data_dir / 'plot_supervision.gp')], check=True)
        print(f"  → {out_dir / 'supervision_sweep.pdf'}")
        print("\\n✓ All plots generated successfully!")
    else:
        raise FileNotFoundError
except (FileNotFoundError, subprocess.CalledProcessError):
    print("\\n⚠ gnuplot not available.")
    print(f"  To generate plots manually, run:")
    print(f"    cd {data_dir} && gnuplot plot_noise.gp && gnuplot plot_supervision.gp")
    print(f"  Or use the data files with your preferred plotting tool:")
    print(f"    {data_dir / 'noise_sweep.dat'}")
    print(f"    {data_dir / 'supervision_sweep.dat'}")

#!/usr/bin/env python3
"""Create and submit one Slurm job per command in sweep_commands.sh.

This is designed for CPU-only runs on an HPC cluster.

It:
- parses a sweep script and extracts lines starting with `python `
- writes one `sbatch` script per command under analysis_runs/slurm_jobs/
- submits them via `sbatch`

Logging:
- Slurm stdout/stderr go to /home/cq5/start/slurm_logs/%x.%j.{out,err}
- Each run directory also contains run_meta.json + evaluation_*.csv

Usage:
  conda activate spaVAE_env
  cd /project/zhiwei/cq5/PythonWorkSpace/TCR-submission
  python analysis_runs/submit_sweep_slurm.py --sweep ./analysis_runs/sweep_commands.sh

Re-running:
- Safe: job scripts are deterministic; if you re-submit with the same run_name,
  add `--resume` in generated commands (see `--add-resume`).
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


def _sanitize_job_name(name: str) -> str:
    # Slurm job names should be simple; keep alnum and underscores.
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name[:200] if name else "icml_job"


def _extract_run_name(cmd: str) -> str | None:
    m = re.search(r"--run_name\s+([^\s]+)", cmd)
    return m.group(1) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="./analysis_runs/sweep_commands.sh")
    ap.add_argument("--jobs-dir", default="./analysis_runs/slurm_jobs")
    ap.add_argument("--submit", action="store_true", help="Actually call sbatch (default: true)")
    ap.add_argument("--no-submit", dest="submit", action="store_false")
    ap.set_defaults(submit=True)

    ap.add_argument("--cpus", type=int, default=4)
    ap.add_argument("--mem-per-cpu", default="4000M")
    ap.add_argument("--time", default="12:00:00")
    ap.add_argument("--partition", default="general")
    ap.add_argument("--qos", default="standard")
    ap.add_argument("--account", default="zhiwei")

    ap.add_argument("--log-dir", default="/home/cq5/start/slurm_logs")
    ap.add_argument("--add-resume", action="store_true", help="Append --resume to each command")
    ap.add_argument(
        "--record",
        default="./analysis_runs/submitted_jobs.txt",
        help="Write created scripts and submitted job IDs to this file.",
    )

    args = ap.parse_args()

    sweep_path = Path(args.sweep)
    if not sweep_path.exists():
        raise SystemExit(f"Missing sweep file: {sweep_path}")

    jobs_dir = Path(args.jobs_dir)
    jobs_dir.mkdir(parents=True, exist_ok=True)

    lines = sweep_path.read_text(encoding="utf-8").splitlines()
    commands = [ln.strip() for ln in lines if ln.strip().startswith("python ")]
    if not commands:
        raise SystemExit(f"No commands found in {sweep_path}")

    created = []
    for i, cmd in enumerate(commands, start=1):
        run_name = _extract_run_name(cmd) or f"run{i:03d}"
        job_name = _sanitize_job_name(f"icml_{run_name}")
        cmd2 = cmd + (" --resume" if args.add_resume and "--resume" not in cmd else "")

        sbatch_path = jobs_dir / f"{job_name}.sbatch"
        sbatch_path.write_text(
            "\n".join(
                [
                    "#!/bin/bash -l",
                    f"#SBATCH --job-name={job_name}",
                    f"#SBATCH --output={args.log_dir}/%x.%j.out",
                    f"#SBATCH --error={args.log_dir}/%x.%j.err",
                    f"#SBATCH --partition={args.partition}",
                    "#SBATCH --nodes=1",
                    "#SBATCH --ntasks=1",
                    f"#SBATCH --cpus-per-task={args.cpus}",
                    f"#SBATCH --mem-per-cpu={args.mem_per_cpu}",
                    f"#SBATCH --qos={args.qos}",
                    f"#SBATCH --account={args.account}",
                    f"#SBATCH --time={args.time}",
                    "",
                    "set -e",
                    "set -o pipefail",
                    "",
                    "# Threading: respect Slurm CPU allocation",
                    "export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}",
                    "export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}",
                    "export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}",
                    "export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}",
                    "",
                    "module purge",
                    "module load wulver",
                    "",
                    "# Ensure slurm log dir exists",
                    f"mkdir -p {args.log_dir}",
                    "",
                    "# Robust shell init (clusters differ in filename)",
                    "if [ -f $HOME/.bash_rc ]; then source $HOME/.bash_rc; fi",
                    "if [ -f $HOME/.bashrc ]; then source $HOME/.bashrc; fi",
                    "if [ -f $HOME/.bash_profile ]; then source $HOME/.bash_profile; fi",
                    "",
                    "# Robust conda init (works even if conda isn't a shell function yet)",
                    "if command -v conda >/dev/null 2>&1; then",
                    "  CONDA_BASE=$(conda info --base)",
                    "  if [ -f $CONDA_BASE/etc/profile.d/conda.sh ]; then source $CONDA_BASE/etc/profile.d/conda.sh; fi",
                    "else",
                    "  if [ -f $HOME/miniconda3/etc/profile.d/conda.sh ]; then source $HOME/miniconda3/etc/profile.d/conda.sh; fi",
                    "  if [ -f $HOME/anaconda3/etc/profile.d/conda.sh ]; then source $HOME/anaconda3/etc/profile.d/conda.sh; fi",
                    "fi",
                    "conda activate spaVAE_env",
                    "",
                    "cd /project/zhiwei/cq5/PythonWorkSpace/TCR-submission",
                    "",
                    f"echo '[JOB] Host='$(hostname) 'Start='$(date)",
                    f"echo '[JOB] Cmd: {cmd2}'",
                    "",
                    # Tee to a stable per-run log file as a convenience.
                    "# Also tee stdout to a per-run log for quick grep without slurm logs",
                    f"mkdir -p ./analysis_runs/run_logs",
                    f"{cmd2} 2>&1 | tee ./analysis_runs/run_logs/{job_name}.$SLURM_JOB_ID.log",
                    "",
                    "echo '[JOB] End='$(date)",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        created.append(sbatch_path)

    record_path = Path(args.record)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with record_path.open("w", encoding="utf-8") as rf:
        rf.write(f"sweep={sweep_path}\n")
        rf.write(f"jobs_dir={jobs_dir}\n")
        rf.write(f"created={len(created)}\n")
        for sb in created:
            rf.write(f"SCRIPT\t{sb}\n")

    print(f"Created {len(created)} sbatch scripts under {jobs_dir}")
    print(f"Record file: {record_path}")

    if not args.submit:
        print("--no-submit specified; not submitting jobs.")
        return

    # Submit all jobs.
    job_ids = []
    for sb in created:
        out = subprocess.check_output(["sbatch", str(sb)], text=True).strip()
        print(out)
        m = re.search(r"Submitted batch job (\d+)", out)
        if m:
            job_ids.append(m.group(1))

    # Append job IDs to record file
    with record_path.open("a", encoding="utf-8") as rf:
        for jid in job_ids:
            rf.write(f"JOB\t{jid}\n")

    if job_ids:
        print("Submitted job IDs:", " ".join(job_ids))


if __name__ == "__main__":
    main()

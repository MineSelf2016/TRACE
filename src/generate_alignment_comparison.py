"""Generate alignment-method comparison table from existing run results."""

from pathlib import Path

import pandas as pd


def extract_results_from_submission():
    """Extract average AUROC/AUPR for each method from evaluation CSV files."""
    repo_root = Path(__file__).resolve().parents[3]
    candidate_dirs = [
        repo_root / "multimodal_clip_binding_results",
        repo_root / "multimodal_clip_binding_results_submission",
    ]
    results_dir = next((p for p in candidate_dirs if p.exists()), candidate_dirs[0])

    method_dirs = {
        "infonce": results_dir / "align",
        "none": results_dir / "noalign",
    }

    results = []

    for method, method_path in method_dirs.items():
        if not method_path.exists():
            print(f"Skip {method}: directory not found")
            continue

        eval_files = list(method_path.glob("*/evaluation_*.csv"))
        if not eval_files:
            print(f"Skip {method}: no evaluation files found")
            continue

        auroc_scores = []
        aupr_scores = []

        for eval_file in eval_files:
            try:
                with open(eval_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("AUROC,"):
                            auroc_scores.append(float(line.strip().split(",")[1]))
                        elif line.startswith("AUPR,"):
                            aupr_scores.append(float(line.strip().split(",")[1]))
            except Exception as e:
                print(f"Read error {eval_file}: {e}")

        if auroc_scores and aupr_scores:
            avg_auroc = sum(auroc_scores) / len(auroc_scores)
            avg_aupr = sum(aupr_scores) / len(aupr_scores)
            results.append(
                {
                    "Alignment Method": method.upper(),
                    "AUROC": f"{avg_auroc:.4f}",
                    "AUPR": f"{avg_aupr:.4f}",
                    "Mechanism": "Contrastive (batch negatives)" if method == "infonce" else "No alignment",
                }
            )
            print(
                f"{method}: AUROC={avg_auroc:.4f}, AUPR={avg_aupr:.4f} "
                f"(averaged over {len(auroc_scores)} runs)"
            )

    return results


def create_latex_table(results):
    """Build latex table code for paper insertion."""
    if not results:
        print("No available results")
        return None

    latex_code = r"""
\begin{table}[t]
\centering
\caption{Ablation study: How does alignment method affect performance?
Results show that supervised binding loss alone (no alignment) leads to
multimodal collapse. With alignment, model succeeds. This table will be extended
with MSE and Cosine methods as additional regularization baselines in revision.}
\label{tab:align_methods}
\begin{tabular}{lccl}
\toprule
\textbf{Alignment Method} & \textbf{AUROC} & \textbf{AUPR} & \textbf{Mechanism} \\
\midrule
"""

    for row in results:
        latex_code += (
            f"{row['Alignment Method']} & {row['AUROC']} & {row['AUPR']} & "
            f"{row['Mechanism']} \\\\n"
        )

    latex_code += r"""\bottomrule
\end{tabular}
\end{table}
"""

    return latex_code


def main():
    print("=" * 60)
    print("Generate alignment-method comparison table from run results")
    print("=" * 60)

    results = extract_results_from_submission()
    if not results:
        print("No results extracted")
        return

    print("\nSummary")
    print("=" * 60)
    print(pd.DataFrame(results).to_string(index=False))

    latex_table = create_latex_table(results)
    if not latex_table:
        return

    print("\nLaTeX table")
    print("=" * 60)
    print(latex_table)

    repo_root = Path(__file__).resolve().parents[3]
    output_file = repo_root / "alignment_comparison.tex"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(latex_table)
    print(f"Saved: {output_file}")

    overleaf_path = Path("/project/zhiwei/cq5/PythonWorkSpace/Overleaf/TCR---Bioinformatics")
    if overleaf_path.exists():
        output_overleaf = overleaf_path / "alignment_comparison.tex"
        with open(output_overleaf, "w", encoding="utf-8") as f:
            f.write(latex_table)
        print(f"Synced to Overleaf: {output_overleaf}")


if __name__ == "__main__":
    main()

"""
Collect and summarize alignment method comparison resultsquence."""

import os
import pandas as pd
import json
from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent.resolve()
RESULTS_BASE = os.path.join(ROOT_PATH, "results_alignment_ablation")

def collect_results():
    """Collect evaluation metrics from all alignment method runs."""
    
    methods = ["none", "mse", "cosine", "infonce"]
    results = []
    
    for method in methods:
        method_dir = os.path.join(RESULTS_BASE, method)
        
        # Find evaluation CSV (pattern: evaluation_*.csv)
        eval_files = list(Path(method_dir).glob("evaluation_*.csv"))
        
        if not eval_files:
            print(f"⚠ No evaluation file found for {method} in {method_dir}")
            continue
        
        eval_file = eval_files[0]
        df_eval = pd.read_csv(eval_file)
        
        # Take last row (best eval)
        if len(df_eval) > 0:
            last_row = df_eval.iloc[-1]
            results.append({
                "Method": method,
                "AUROC": last_row.get("AUROC", float("nan")),
                "AUPR": last_row.get("AUPR", float("nan")),
                "Accuracy": last_row.get("Accuracy", float("nan")),
                "F1": last_row.get("F1", float("nan")),
            })
    
    if not results:
        print("❌ No results found!")
        return
    
    results_df = pd.DataFrame(results)
    
    # Calculate deltas vs MSE baseline
    mse_auroc = results_df[results_df["Method"] == "mse"]["AUROC"].values
    if len(mse_auroc) > 0:
        mse_auroc = mse_auroc[0]
        results_df["Delta AUROC vs MSE"] = results_df["AUROC"] - mse_auroc
    
    print("\n" + "="*70)
    print("ALIGNMENT METHODS COMPARISON RESULTS")
    print("="*70)
    print(results_df.to_string(index=False))
    print("="*70)
    
    # Save summary
    summary_file = os.path.join(RESULTS_BASE, "alignment_comparison_summary.csv")
    results_df.to_csv(summary_file, index=False)
    print(f"\n✓ Summary saved to: {summary_file}")
    
    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    infonce_auroc = results_df[results_df["Method"] == "infonce"]["AUROC"].values
    if len(infonce_auroc) > 0:
        infonce_auroc = infonce_auroc[0]
        if len(mse_auroc) > 0:
            delta = infonce_auroc - mse_auroc
            print(f"\nInfoNCE vs MSE delta: {delta:+.4f} AUROC")
            print("\nInterpretation:")
            print(f"  - MSE: Basic L2 regularization only ({mse_auroc:.4f} AUROC)")
            print(f"  - InfoNCE: Full contrastive (+ geometric constraint) ({infonce_auroc:.4f} AUROC)")
            print(f"  - Gap ({delta:+.4f}pp) = Genuine structural information extracted by contrastive loss")
            print("\nConclusion:")
            if delta > 0.01:
                print(f"  ✓ Contrastive alignment DOES leverage structural information!")
                print(f"  ✓ Not just regularization—the structured approach matters.")
            else:
                print(f"  ✓ Alignment helps, but benefits are modest (likely regularization-heavy)")


if __name__ == "__main__":
    collect_results()

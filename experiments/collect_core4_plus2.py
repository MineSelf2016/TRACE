import os
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_BASE = ROOT / "results_core4_plus2"

ORDER = ["none", "mse", "cosine", "infonce", "seq_mlp", "seq_linear"]


def read_eval_csv(eval_path: Path) -> dict:
    df = pd.read_csv(eval_path)
    out = {
        "AUROC": float("nan"),
        "AUPR": float("nan"),
        "Accuracy": float("nan"),
        "Recall": float("nan"),
        "Precision": float("nan"),
        "F1": float("nan"),
        "Loss": float("nan"),
    }
    if "metrics" in df.columns and "score" in df.columns:
        for _, row in df.iterrows():
            m = str(row["metrics"])
            if m in out:
                out[m] = float(row["score"])
    return out


def main():
    rows = []
    for name in ORDER:
        run_dir = RESULTS_BASE / name / "run_seed12"
        eval_file = run_dir / "evaluation_1.csv"
        if not eval_file.exists():
            rows.append({"Method": name, "Status": "missing"})
            continue
        metrics = read_eval_csv(eval_file)
        rows.append({"Method": name, "Status": "ok", **metrics})

    res = pd.DataFrame(rows)

    if "AUROC" in res.columns:
        res["AUROC"] = pd.to_numeric(res["AUROC"], errors="coerce")
    if "AUPR" in res.columns:
        res["AUPR"] = pd.to_numeric(res["AUPR"], errors="coerce")

    # delta vs MSE for reviewer-facing interpretation
    mse_row = res[res["Method"] == "mse"]
    if len(mse_row) > 0 and pd.notna(mse_row["AUROC"].iloc[0]):
        mse_auroc = float(mse_row["AUROC"].iloc[0])
        res["Delta_AUROC_vs_MSE"] = res["AUROC"] - mse_auroc

    if len(mse_row) > 0 and pd.notna(mse_row["AUPR"].iloc[0]):
        mse_aupr = float(mse_row["AUPR"].iloc[0])
        res["Delta_AUPR_vs_MSE"] = res["AUPR"] - mse_aupr

    out_csv = RESULTS_BASE / "core4_plus2_summary.csv"
    res.to_csv(out_csv, index=False)

    print("\n=== Core4 + Extra2 Summary ===")
    print(res.to_string(index=False))
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()

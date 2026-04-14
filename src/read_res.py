from pathlib import Path
import pandas as pd

def load_all_results(root_dir):
    """
    读取所有实验结果
    """
    records = []

    root_dir = Path(root_dir)

    for exp_dir in root_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        if not exp_dir.name.startswith("grid_"):
            continue

        for csv_file in exp_dir.glob("*.csv"):
            df = pd.read_csv(csv_file)

            df["exp_dir"] = exp_dir.name
            df["dataset"] = csv_file.stem  # 文件名即数据集名

            records.append(df)

    all_df = pd.concat(records, ignore_index=True)
    return all_df


def best_metric_per_dataset(all_df, metric_name):
    """
    对每个 dataset，找某个 metric 的最优结果
    """
    df_metric = all_df[all_df["metrics"] == metric_name]

    # 找每个 dataset 中 score 最大的那一行
    idx = df_metric.groupby("dataset")["score"].idxmax()

    best_df = df_metric.loc[idx].sort_values(
        by="score", ascending=False
    )

    return best_df.reset_index(drop=True)


if __name__ == "__main__":
    ROOT = "multimodal_clip_binding_results"  # 改成你的根目录

    all_results = load_all_results(ROOT)

    all_results.to_csv(f"{ROOT}/all_results_combined.csv", index=False)

    # === 1. 每个数据集最高 AUROC ===
    best_auroc = best_metric_per_dataset(all_results, "AUROC")
    print("\n===== Best AUROC per dataset =====")
    print(best_auroc[[
        "dataset", "score", "exp_dir",
        "proj_dim", "graph_hidden", "graph_layers",
        "dropout", "learning_rate", "batch_size",
        "lambda_clip", "temperature", "optimizer",
        "weight_decay", "lambda_bind"
    ]])

    # === 2. 每个数据集最高 Accuracy ===
    best_acc = best_metric_per_dataset(all_results, "Accuracy")
    print("\n===== Best Accuracy per dataset =====")
    print(best_acc[[
        "dataset", "score", "exp_dir",
        "proj_dim", "graph_hidden", "graph_layers",
        "dropout", "learning_rate", "batch_size",
        "lambda_clip", "temperature", "optimizer",
        "weight_decay", "lambda_bind"
    ]])

    # === 3. 每个数据集最高 Recall ===
    best_recall = best_metric_per_dataset(all_results, "Recall")
    print("\n===== Best Recall per dataset =====")
    print(best_recall[[
        "dataset", "score", "exp_dir",
        "proj_dim", "graph_hidden", "graph_layers",
        "dropout", "learning_rate", "batch_size",
        "lambda_clip", "temperature", "optimizer",
        "weight_decay", "lambda_bind"
    ]])

    # === 4. 每个数据集最高 Precision ===
    best_precision = best_metric_per_dataset(all_results, "Precision")
    print("\n===== Best Precision per dataset =====")
    print(best_precision[[
        "dataset", "score", "exp_dir",
        "proj_dim", "graph_hidden", "graph_layers",
        "dropout", "learning_rate", "batch_size",
        "lambda_clip", "temperature", "optimizer",
        "weight_decay", "lambda_bind"
    ]])

    # === 5. 每个数据集最高 F1 ===
    best_f1 = best_metric_per_dataset(all_results, "F1")
    print("\n===== Best F1 per dataset =====")
    print(best_f1[[
        "dataset", "score", "exp_dir",
        "proj_dim", "graph_hidden", "graph_layers",
        "dropout", "learning_rate", "batch_size",
        "lambda_clip", "temperature", "optimizer",
        "weight_decay", "lambda_bind"
    ]])

    # === 6. 每个数据集最高 AUPR ===
    best_aupr = best_metric_per_dataset(all_results, "AUPR")
    print("\n===== Best AUPR per dataset =====")
    print(best_aupr[[
        "dataset", "score", "exp_dir",
        "proj_dim", "graph_hidden", "graph_layers",
        "dropout", "learning_rate", "batch_size",
        "lambda_clip", "temperature", "optimizer",
        "weight_decay", "lambda_bind"
    ]])



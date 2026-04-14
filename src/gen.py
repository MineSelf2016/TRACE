import pandas as pd

fracs = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]

for i in range(5):
    # /project/zhiwei/g013857/TCR-Multi/dataset/ds.hard-splits/pep+cdr3b/train/only-sampled-negs/train-0.csv

    for j, frac in enumerate(fracs):
        df_a = pd.read_csv(f'/project/zhiwei/g013857/TCR-Multi/dataset/ds.hard-splits/pep+cdr3b/train/only-sampled-negs/train-{i}.csv', dtype=str)
        df_b = pd.read_csv(f'/project/zhiwei/g013857/TCR-Multi/dataset/ds.hard-splits/pep+cdr3b/test/only-sampled-negs/test-{i}.csv', dtype=str)

        # df_b = df_b[df_b["label"] == '1']
        df_b = df_b.sample(frac=0.95, random_state=42)

        df_combined = pd.concat([df_a, df_b], ignore_index=True)
        df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

        value_counts = df_combined['label'].value_counts()
        print(value_counts)

        to_file = f"/project/zhiwei/g013857/TCR-Multi/dataset/ds.hard-splits/pep+cdr3b2/train/only-sampled-negs/train-{i}-{frac}.csv"
        df_combined.to_csv(to_file, index=False)
        print(to_file)
        # print(len(df_b_sample))
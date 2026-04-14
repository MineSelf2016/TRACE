import pandas as pd
import pickle

df = pd.read_csv('dataset/ds.hard-splits/pep+cdr3b/test/only-sampled-negs/test-1.csv')

with open('embs/tcr_seq_dict.pkl', 'rb') as f:
    tcr_dict = pickle.load(f)
with open('embs/peptide_seq_dict.pkl', 'rb') as f:
    pep_dict = pickle.load(f)

tcrs_in_csv = set(df['cdr3.beta'].unique())
peps_in_csv = set(df['antigen.epitope'].unique())
tcrs_in_dict = set(tcr_dict.keys())
peps_in_dict = set(pep_dict.keys())

overlap_tcr = tcrs_in_csv & tcrs_in_dict
overlap_pep = peps_in_csv & peps_in_dict

print(f"CSV has {len(df)} rows")
print(f"\nTCR: {len(tcrs_in_csv)} unique, {len(overlap_tcr)} have embeddings ({len(tcrs_in_csv-overlap_tcr)} missing)")
print(f"Peptide: {len(peps_in_csv)} unique, {len(overlap_pep)} have embeddings ({len(peps_in_csv-overlap_pep)} missing)")

# Filter to only rows where both TCR and peptide have embeddings
df_filtered = df[df['cdr3.beta'].isin(overlap_tcr) & df['antigen.epitope'].isin(overlap_pep)]
print(f"\nFiltered: {len(df_filtered)} / {len(df)} rows have both embeddings")
df_filtered.to_csv('dataset/ds.hard-splits/pep+cdr3b/test/only-sampled-negs/test-1-filtered.csv', index=False)
print("Saved to test-1-filtered.csv")

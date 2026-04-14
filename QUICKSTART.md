# Quick Start Guide

## 60-Second Setup

```bash
# 1. Clone/Download
cd TCR-submission/submission/github

# 2. Install
pip install -r requirements.txt
pip install torch torch-geometric

# 3. Download data
# [See DATA_README.md in figshare/]

# 4. Train TRACE
python src/train.py --dataset tchardx --model_name TRACE
```

## First Run: TRACE Training (10 minutes)

```bash
# Quick CPU test
python src/train.py \
  --device cpu \
  --epochs 5 \
  --batch_size 32 \
  --model_name TRACE

# Output:
# Epoch 1/5 | Loss: 0.512 | AUROC: 0.580 | AUPR: 0.380
# ...
# Best AUROC: 0.612 | Model saved to logs/
```

## Reproduce Paper Results

### 1. Main Results (Table 1)
```bash
# TRACE vs baselines on TCHard RN
python src/train.py \
  --dataset tchardx \
  --split_type hard-splits \
  --model_name TRACE \
  --lambda_align 0.2 \
  --epochs 100
```

Expected: **AUROC 0.689**

### 2. Ablation (Table 2)
```bash
# Seq-only baseline
python src/train.py \
  --dataset tchardx \
  --model_name seq_only

# Seq+Graph without alignment
python src/train.py \
  --dataset tchardx \
  --model_name seq_graph \
  --lambda_align 0.0

# TRACE with alignment
python src/train.py \
  --dataset tchardx \
  --model_name TRACE \
  --lambda_align 0.2
```

Expected AUROC: **0.662** → **0.506** → **0.689**

### 3. Noise Sweep (Figure 3a)
```bash
# Train with 20% edge dropout
python src/train.py \
  --dataset tchardx \
  --edge_dropout 0.2 \
  --model_name TRACE

# Try 0%, 10%, 20%, 30%, 40% dropout
for dropout in 0.0 0.1 0.2 0.3 0.4; do
  python src/train.py \
    --dataset tchardx \
    --edge_dropout $dropout \
    --model_name TRACE \
    --run_tag "noise_sweep_${dropout}"
done

# Collect results
python experiments/collect_core4_plus2.py
python analysis_runs/generate_sweep_plots.py
```

### 4. Supervision Sweep (Figure 3b)
```bash
# Train with 50% positive examples
python src/train.py \
  --dataset tchardx \
  --pos_downsample 0.5 \
  --model_name TRACE

# Try 10%, 20%, 50%, 100%
for frac in 0.1 0.2 0.5 1.0; do
  python src/train.py \
    --dataset tchardx \
    --pos_downsample $frac \
    --model_name TRACE \
    --run_tag "sup_sweep_${frac}"
done
```

### 5. Alignment Geometry Analysis (Figure 4)
```bash
# Generate alignment quality plots
python analysis_runs/plot_enhanced_geometry.py

# Output: figures/cosine_similarity_violin.png
```

### 6. Biological Interpretability (Figure 5)
```bash
# Validate that alignment captures biological signals
python analysis_runs/biological_interpretability.py

# Output: figures/binding_pair_discrimination.png
```

## Directory Structure

After running:
```
.
├── logs/                          # Training logs & checkpoints
│   ├── TRACE_2024_04_12.log
│   ├── best_model.pt
│   └── metrics.json
├── results/                       # Aggregated results
│   ├── sweep_results.csv
│   └── ablation_results.csv
└── figures/                       # Generated plots
    ├── noise_sweep.png
    ├── supervision_sweep.png
    └── alignment_geometry.png
```

## Typical Output

```
Training TRACE on TCHard RN...
Data loaded: 48000 train | 6000 val | 6000 test samples
Model initialized: 2-layer GNN, embedding_dim=256
Training with Adam (lr=5e-4), batch_size=64, epochs=100

Epoch   1/100 | Loss: 0.486 | Bind: 0.421 | Align: 0.065 | LR: 5.0e-04
      Val AUROC: 0.612 | AUPR: 0.485 | ECE: 0.089
      
Epoch  10/100 | Loss: 0.215 | Bind: 0.187 | Align: 0.028 | LR: 5.0e-04
      Val AUROC: 0.668 | AUPR: 0.515 | ECE: 0.072
      
Epoch  80/100 | Loss: 0.124 | Bind: 0.118 | Align: 0.006 | LR: 2.5e-04
      Val AUROC: 0.689 | AUPR: 0.538 | ECE: 0.067 ← Best!
      
Epoch  88/100 | Early stopping triggered (no improvement for 8 epochs)

Final Results:
  Test AUROC:  0.689
  Test AUPR:   0.538
  Test F1:     0.531
  Saved model: logs/best_model.pt
```

## Monitoring

### View Training in Real-time
```bash
# Terminal 1: Run training
python src/train.py --logdir ./runs/trace_001

# Terminal 2: Start TensorBoard
tensorboard --logdir ./runs
# Open http://localhost:6006
```

## Debugging Common Issues

### Error: "No data files found"
```bash
# Check symlinks in figshare/
ls -la submission/figshare/
# Should show: dataset -> ..., embeddings -> ...

# If missing, create them:
cd submission/figshare
ln -s ../../TCR-ISMB/dataset dataset
ln -s ../../TCR-ISMB/embs embeddings
```

### Error: "CUDA out of memory"
```bash
# Reduce batch size
python src/train.py --batch_size 16 --device cuda:0

# Or use CPU for debugging
python src/train.py --device cpu
```

### Error: "Can't import torch_geometric"
```bash
# Reinstall PyTorch Geometric
pip install --upgrade torch-geometric -f https://data.pyg.org/whl/
```

## Next Steps

1. **Understand the model**: Read `README.md` → Model Architecture section
2. **Modify hyperparameters**: Edit `--lambda_align`, `--learning_rate`, etc.
3. **Run full experiments**: See "Reproduce Paper Results" section
4. **Analyze results**: Use scripts in `analysis_runs/`
5. **Adapt to your data**: Update data loading in `src/train.py`

## Performance Benchmarks

| Setting | Device | Time/Epoch | Total |
|---------|--------|-----------|--------|
| Seq-only | GPU (A100) | ~30s | 50 min (100 epochs) |
| TRACE | GPU (A100) | ~35s | 60 min |
| TRACE | GPU (V100) | ~45s | 75 min |
| TRACE | CPU | ~5 min | 8 hrs |

## Key Parameters

```python
# Model
--embedding_dim 256              # Shared latent space
--gnn_layers 2                   # Message-passing depth
--batch_size 64                  # Batch size

# Training
--learning_rate 5e-4             # Adam learning rate
--epochs 100                     # Total epochs
--patience 8                     # Early stopping

# Alignment (critical!)
--lambda_align 0.2               # Alignment loss weight
--lambda_bind 1.0                # Binding loss weight
--temperature 0.07               # InfoNCE temperature

# Data augmentation
--edge_dropout 0.0               # Noise level (0-0.4)
--pos_downsample 1.0             # Positive fraction (0.1-1.0)
```

## Citation

```
@article{qi2024trace,
  title={When Multimodal Fusion Fails: Contrastive Alignment as a Necessary Stabilizer for TCR-Peptide Binding Prediction},
  author={Qi, Cong and Wang, Wenbo and Fang, Hanzhang and Wei, Zhi},
  journal={arXiv preprint},
  year={2024}
}
```

---

For more details, see `README.md` and `INSTALL.md`

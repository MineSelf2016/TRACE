# TRACE: TCR Robust Alignment via Contrastive Encoding

Official implementation for the paper: **"When Multimodal Fusion Fails: Contrastive Alignment as a Necessary Stabilizer for TCR-Peptide Binding Prediction"**

## Overview

This repository contains the implementation of TRACE, a lightweight multimodal framework that addresses the problem of unstable multimodal fusion in TCR-peptide binding prediction. Our key finding is that naive fusion of sequence embeddings and structure-derived residue graphs can degrade performance below sequence-only baselines when structural inputs are noisy. TRACE stabilizes multimodal learning through **intra-entity contrastive alignment**.

### Key Contributions
- **Identifies a systematic failure mode**: Naive multimodal fusion with imperfect structural information can hurt performance
- **Proposes TRACE**: Uses CLIP-style intra-entity contrastive alignment to regularize representation geometry
- **Provides comprehensive validation**: Including controlled noise/supervision sweeps, calibration analysis, and biological interpretability studies
- **Demonstrates stability**: TRACE consistently outperforms sequence+graph fusion without alignment across challenging evaluation protocols

## Project Structure

```
.
├── src/                              # Core implementation
│   ├── train.py                      # Main training script with TRACE model
│   ├── train2.py, train3.py         # Alternative training variants
│   ├── analysis.py                   # Performance analysis utilities
│   ├── analysis_rn_hardness.py      # TCHard protocol analysis
│   ├── models/                       # Model definitions (generated from train.py)
│   ├── utils/                        # Utility functions
│   └── ...
│
├── experiments/                      # Experimental scripts
│   ├── collect_alignment_results.py  # Collect alignment ablation results
│   ├── collect_core4_plus2.py       # Collect core+2 comparison results
│   └── ...
│
├── analysis_runs/                        # submission-specific experiments
│   ├── biological_interpretability.py # Biological validation
│   ├── plot_enhanced_geometry.py     # Visualizations
│   ├── generate_sweep_plots.py       # Noise/supervision sweep plots
│   ├── make_sweep_commands.py        # Generate sweep command scripts
│   └── ...
│
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
└── LICENSE                           # License information

```

## Installation

### Prerequisites
- Python 3.9+
- CUDA 11.8+ (for GPU support)
- conda (recommended)

### Setup

1. **Clone and navigate to repository:**
   ```bash
   cd TCR-submission/submission/github
   ```

2. **Create conda environment:**
   ```bash
   conda create -n trace_env python=3.10
   conda activate trace_env
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install PyTorch (with CUDA support):**
   ```bash
   # For CUDA 11.8
   pip install torch::2.0.0+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # For CPU only
   pip install torch torchvision torchaudio
   ```

5. **Install PyTorch Geometric:**
   ```bash
   pip install pyg-lib torch-geometric
   ```

## Usage

### Basic Training (TRACE Model)

```bash
# Example: Train TRACE on TCHard RN dataset with alignment
python src/train.py \
  --dataset tchardx \
  --dataset_index 1 \
  --split_type hard-splits \
  --model_name TRACE \
  --batch_size 64 \
  --learning_rate 5e-4 \
  --epochs 100 \
  --device cuda:0

# Key flags:
# --dataset: Dataset choice (default: tchardx)
# --split_type: hard-splits | validation_split | few_shot_split
# --model_name: TRACE | seq_only | seq+graph
# --lambda_align: Alignment weight (default: 0.2)
```

### Controlled Experiments

#### Noise Sweep (Edge Dropout)
```bash
python src/train.py \
  --dataset tchardx \
  --edge_dropout 0.2 \
  --model_name TRACE
```

#### Supervision Sweep (Positive Downsampling)
```bash
python src/train.py \
  --dataset tchardx \
  --pos_downsample 0.5 \
  --model_name TRACE
```

#### Generate Sweep Commands
```bash
python analysis_runs/make_sweep_commands.py \
  --mode only-sampled-negs \
  --dataset-index 1 \
  --write ./sweep_commands.sh

# Review and run manually:
# bash ./sweep_commands.sh
```

### Analysis and Visualization

```bash
# Collect results from multiple runs
python experiments/collect_core4_plus2.py

# Generate alignment geometry analysis
python analysis_runs/plot_enhanced_geometry.py

# Generate biological interpretability plots
python analysis_runs/biological_interpretability.py

# Create sweep visualization
python analysis_runs/generate_sweep_plots.py
```

## Model Architecture

### TRACE Framework
```
Entity (TCR or Peptide)
    ├─ Sequence Tower
    │  └─ Pretrained PLM → Projection → z_seq
    │
    └─ Graph Tower
       └─ Residue Graph (from AlphaFold) → GNN → Projection → z_graph
           │
           └─ L message-passing layers
           
         ↓ [Intra-entity Contrastive Alignment - InfoNCE Loss]
         
    Fusion: z = Concat(z_seq, z_graph) → MLP → h
    
    ↓ [Sequence + Graph Encoder (TCR & Peptide)]
    
    Interaction Features → MLP → Binding Prediction (binary)
```

### Key Components

**1. Sequence Tower:**
- Global embedding from pretrained protein language model (ProtT5, ESM-2, etc.)
- Projection to shared latent space dimension D

**2. Graph Tower:**
- Residue-level graph from predicted AlphaFold structures
- Node features: 20D one-hot amino-acid identity
- Edges: sequence adjacency + spatial proximity (8Å cutoff)
- Lightweight GNN with L message-passing layers

**3. Intra-entity Contrastive Alignment (Novel):**
- CLIP-style InfoNCE loss
- Aligns sequence and graph embeddings **within** each entity
- Acts as geometric regularizer preventing structure noise from dominating
- Enforces representation consistency on shared hypersphere (after L2 normalization)

**4. Binding Classifier:**
- Explicit interaction features: [h_tcr, h_pep, |h_tcr - h_pep|, h_tcr ⊙ h_pep]
- Small MLP for binary classification
- Class-weighted cross-entropy loss (handles imbalance)

## Training Details

### Loss Function
```
L = λ_bind × L_CE + λ_align × L_align

where:
  L_CE: Class-weighted cross-entropy (with class weights for imbalance)
  L_align: Symmetric InfoNCE loss (seq→graph + graph→seq)
```

### Hyperparameters (Default)
| Parameter | Value | Notes |
|-----------|-------|------|
| Learning rate | 5×10⁻⁴ | Adam optimizer |
| Weight decay | 1×10⁻³ | L2 regularization |
| Batch size | 64 | Affects alignment denominator |
| Temperature τ | 0.07 | InfoNCE temperature (from CLIP) |
| λ_bind | 1.0 | Binding loss weight |
| λ_align | 0.2 | Alignment loss weight |
| GNN layers | 2 | Message-passing depth |
| Hidden dim | 256 | Per-layer GNN dimension |
| Embedding dim D | 256 | Shared latent space dimension |
| Early stopping patience | 8 | Epochs without improvement |

### Data & Protocol

**Dataset:** TCHard (controlled split protocol, stress-tests distribution shift)
- Focus on RN (random negatives) setting
- Hard negatives from random TCR-peptide pairing
- Substantial class imbalance (~7-10% positives)

**Splits:**
- `hard-splits`: Main reported results
- `validation_split`: Held-out epitope generalization
- `few_shot_split`: Low-data regime

## Key Results Summary

### Main Performance Comparison (TCHard RN, AUROC)
| Model | AUROC | Notes |
|-------|-------|-------|
| Sequence-only | 0.662 | Baseline |
| Seq + Graph (no alignment) | 0.506 | **Multimodal failure→ random** |
| **TRACE (with alignment)** | **0.689** | **+3.8% over sequence** |

### Ablation Study
| Configuration | AUROC |
|---|---|
| Seq-only | 0.662 |
| Seq+Graph (no alignment) | 0.506 |
| MSE regularization | 0.501 |
| Cosine regularization | 0.505 |
| **InfoNCE alignment (TRACE)** | **0.689** |

### Robustness Under Noise
- **Edge dropout**: 0-40% dropout rate
  - Without alignment: AUROC ~0.505 (random)
  - With alignment: AUROC 0.53-0.55 (stable)
  
- **Data scarcity**: 10-100% positive examples
  - Without alignment: AUROC ~0.505 (random)
  - With alignment: AUROC 0.53-0.55 (consistent)

## Dataset & Embeddings

**Note:** This repository includes code and documentation. Data files should be downloaded from Figshare.

### Required Data Files (place in `data/`)
```
data/
├── all_data.csv                    # Main dataset (53 MB)
├── ds.hard-splits/                 # TCHard split data
├── validation_split/               # Held-out epitope validation
├── few_shot_split/                 # Few-shot learning split
└── embeddings/
    ├── tcr_seq_dict.pkl            # TCR sequence embeddings (975 MB)
    ├── tcr_graph_dict.pkl          # TCR residue graphs (509 MB)
    ├── peptide_seq_dict.pkl        # Peptide embeddings (6.8 MB)
    └── peptide_graph_dict.pkl      # Peptide graphs (2.7 MB)
```

**Download from:** [Figshare Link] (see supplementary materials)

## Experiments & Reproduction

### Quick Validation Run (CPU, ~5 min)
```bash
python src/train.py \
  --device cpu \
  --epochs 2 \
  --batch_size 32 \
  --model_name TRACE \
  --quick_check True
```

### Full Sweep Experiments (Recommended: GPU cluster)

1. **Noise sweep** (edge dropout 0%, 10%, 20%, 30%, 40%):
   ```bash
   cd analysis_runs
   python make_sweep_commands.py --mode edge-dropout --write sweep_noise.sh
   bash sweep_noise.sh
   python plot_enhanced_geometry.py
   ```

2. **Supervision sweep** (positive fraction 10%, 20%, 50%, 100%):
   ```bash
   python make_sweep_commands.py --mode pos-downsample --write sweep_supervision.sh
   bash sweep_supervision.sh
   ```

3. **Alignment method comparison** (No alignment, MSE, Cosine, InfoNCE):
   ```bash
   python collect_core4_plus2.py
   ```

### Visualization
```bash
# Generate all main figures
cd analysis_runs
python generate_sweep_plots.py           # Noise & supervision sweeps (Fig 3)
python plot_enhanced_geometry.py         # Alignment geometry (Fig 4)
python biological_interpretability.py    # Biological analysis (Fig 5)
```

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir ./logs
```

### Typical Training Output
```
Epoch 1/100 | Loss: 0.486 | AUROC: 0.612 | AUPR: 0.421
Epoch 2/100 | Loss: 0.392 | AUROC: 0.645 | AUPR: 0.468
...
Epoch 80/100 | Loss: 0.124 | AUROC: 0.689 | AUPR: 0.538
Best Val AUROC: 0.689 → saving checkpoint
Early stopping triggered after patience=8
```

## Citation

If you use TRACE in your research, please cite:

```bibtex
@article{qi2024trace,
  title={When Multimodal Fusion Fails: Contrastive Alignment as a Necessary Stabilizer for TCR-Peptide Binding Prediction},
  author={Qi, Cong and Wang, Wenbo and Fang, Hanzhang and Wei, Zhi},
  journal={arXiv preprint arXiv:2406.xxxxx},
  year={2024}
}
```

## Contact & Support

For questions or issues:
- Create an issue on GitHub
- Contact: zhi.wei@njit.edu

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- Based on prior work on TCR-peptide binding prediction (NetTCR, ERGO, etc.)
- CLIP framework for contrastive learning (Radford et al., 2021)
- AlphaFold for structure prediction (Jumper et al., 2021)
- PyTorch Geometric for graph neural networks

---

**Last Updated:** April 2024
**Paper Status:** [Submitted to submission]

# Installation & Setup Guide

## Quick Start

### Option 1: Using Conda (Recommended)

```bash
# Create environment
conda env create -f environment.yml -n trace_env
conda activate trace_env

# Or manually:
conda create -n trace_env python=3.10
conda activate trace_env
pip install -r requirements.txt
```

### Option 2: Using pip with venv

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Detailed Dependency Installation

### GPU Support (CUDA 11.8)

```bash
conda activate trace_env

# PyTorch with CUDA
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# PyTorch Geometric
pip install pyg-lib torch-geometric -f https://data.pyg.org/whl/
```

### CPU Only

```bash
pip install torch torchvision torchaudio
pip install pyg-lib torch-geometric
```

### Other Important Packages

```bash
# Scientific computing
pip install numpy pandas scikit-learn scipy

# Visualization
pip install matplotlib seaborn plotly

# Deep learning utilities
pip install torch-optim-lookahead torch-lr-finder

# Graph neural networks
pip install torch-geometric torch-scatter torch-sparse

# Logging & monitoring
pip install tensorboard wandb

# Structure prediction (optional)
pip install biopython biotite

# Development
pip install jupyter ipython pytest
```

## Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import torch_geometric; print(f'PyG {torch_geometric.__version__}')"
python -c "import sklearn; print(f'sklearn {sklearn.__version__}')"
```

## Environment Configuration

For slurm-based clusters, create a submission script:

```bash
#!/bin/bash
#SBATCH --job-name=trace_train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_%j.log

# Load modules
module load cuda/11.8
module load pytorch/2.0.0

# Activate environment
conda activate trace_env

# Run training
cd /path/to/TCR-submission/submission/github
python src/train.py --device cuda:0 [other arguments]
```

## Data Setup

Download data from Figshare and organize as follows:

```
data/
├── all_data.csv
├── ds.hard-splits/
│   ├── test.csv
│   ├── train.csv
│   └── val.csv
├── validation_split/
├── few_shot_split/
└── embeddings/
    ├── tcr_seq_dict.pkl
    ├── tcr_graph_dict.pkl
    ├── peptide_seq_dict.pkl
    └── peptide_graph_dict.pkl
```

Update `src/train.py` to point to your data directory:

```python
# In train.py
DATA_PATH = "/path/to/data"
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `--batch_size` (from 64 to 32 or 16)
- Use `--device cpu` for debugging

### Import Errors
```bash
# Reinstall PyTorch Geometric
pip install --upgrade torch-geometric -f https://data.pyg.org/whl/
```

### Version Conflicts
```bash
# Clear pip cache and reinstall
pip cache purge
pip install -r requirements.txt --force-reinstall
```

## Testing Your Setup

```bash
# Quick validation
python src/train.py \
  --device cpu \
  --epochs 1 \
  --batch_size 16 \
  --quick_check True

# Should complete in < 5 minutes
```

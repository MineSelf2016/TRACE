# Code File Index

## Core Training & Models

### `src/train.py` ⭐ (Main File)
**Primary training script for TRACE model**
- Dataset loading (TCHard splits)
- Model architecture (Sequence + Graph towers with alignment)
- Training loop with binding + alignment losses
- Evaluation metrics (AUROC, AUPR, F1, calibration)
- Key arguments:
  - `--model_name`: TRACE, seq_only, seq_graph
  - `--lambda_align`: 0.0-1.0 (alignment weight)
  - `--edge_dropout`: 0.0-0.4 (structural noise)
  - `--pos_downsample`: 0.1-1.0 (data scarcity)

### `src/train2.py`
**Alternative training variant** (experimental, for comparison)
- Similar structure to train.py
- Different hyperparameter configurations
- Used for hyperparam tuning ablations

### `src/train3.py`
**Another training variant** (experimental)
- May test different loss formulations or optimizers

---

## Analysis & Metrics

### `src/analysis.py`
**Post-training analysis utilities**
- Load trained models
- Compute metrics on test sets
- Generate prediction files
- Model inference on new data

### `src/analysis_rn_hardness.py`
**TCHard protocol-specific analysis**
- Quantifies hardness of random negative (RN) sampling
- Analyzes negative distribution properties
- Validates that RN protocol is genuinely hard

### `src/check_embeddings.py`
**Validate pre-computed embeddings**
- Verify embedding dimensions
- Check for missing sequences
- Validate graph structure consistency
- Debug encoding issues

### `src/read_res.py`
**Read and parse results files**
- Load metrics from training logs
- Extract AUROC, AUPR, F1, calibration
- Format for comparisons

### `src/extract_res.py`
**Extract results from multiple runs**
- Batch process training outputs
- Aggregate metrics across runs
- Generate summary statistics

---

## Data & Utilities

### `src/gen.py`
**Data generation utilities**
- TCR/peptide sequence processing
- Graph construction helpers
- Embedding generation wrappers

### `src/generate_alignment_comparison.py`
**Generate alignment comparison data**
- Run models with different alignment methods (MSE, Cosine, InfoNCE)
- Compare vector similarity distributions
- Output alignment quality metrics

---

## Experiments & Analysis Scripts

### `experiments/collect_core4_plus2.py` ⭐
**Collect results from "core4+2" comparison**
- Compares: No align, MSE, Cosine, InfoNCE, Seq-MLP, Seq-Linear
- Aggregates metrics from 6 model variants
- Produces comparison table (Table 1 in paper)
- Output: `core4_plus2_results.csv`

### `experiments/collect_alignment_results.py`
**Collect results from alignment ablation experiments**
- Gathers alignment geometry metrics
- Cosine similarity statistics
- Cross-view correlation analysis

---

## submission-Specific Analysis & Visualization

### `analysis_runs/make_sweep_commands.py` ⭐ (Key)
**Generate single experiment sweep commands**
- Creates bash scripts for noise sweep (edge dropout 0-40%)
- Creates scripts for supervision sweep (positive fraction 10-100%)
- Output: `sweep_commands.sh` (executable script)

### `analysis_runs/generate_sweep_plots.py` ⭐ (Visualization)
**Visualize noise and supervision sweeps (Figure 3)**
- Plots AUROC vs. edge dropout percentage
- Plots AUROC vs. positive fraction
- Shows aligned vs. non-aligned comparison
- Outputs: `noise_sweep.png`, `supervision_sweep.png`

### `analysis_runs/plot_enhanced_geometry.py` ⭐ (Visualization)
**Create alignment geometry plots (Figure 4)**
- Violin plots: cosine similarity distributions
- Shows mean & std of seq-graph similarity
- Compares aligned vs. non-aligned models
- Output: `cosine_violins.png`

### `analysis_runs/biological_interpretability.py` ⭐ (Validation)
**Biological validation experiments (Figure 5)**
- Tests whether aligned models discriminate binding pairs
- Computes seq-graph similarity for binding vs. non-binding
- T-tests for statistical significance
- Embedding norm correlation analysis
- Output: `biological_interpretability.png`

### `analysis_runs/collect_results.py`
**Aggregate results from all batch jobs**
- Parses SLURM logs
- Collects metrics from multiple runs
- Generates unified results table

### `analysis_runs/plot_results.py`
**General plotting utilities**
- Visualize training curves
- ROC/PR curves
- Calibration plots
- Loss component decomposition

### `analysis_runs/plot_loss_components.py`
**Decompose loss during training**
- Separate binding loss vs. alignment loss over epochs
- Visualize loss trade-offs
- Output: `loss_components.png`

### `analysis_runs/plot_align_geometry.py`
**Earlier version of alignment geometry plotting**
- Predecessor to `plot_enhanced_geometry.py`
- Can generate alternative visualizations

### `analysis_runs/generate_plots.py`
**Generate multiple plots in batch**
- Calls other visualization functions
- Creates comprehensive figure panel
- Output: `figures/combined_analysis.png`

### `analysis_runs/align_geometry.py`
**Compute alignment geometry metrics**
- Calculate cosine similarity statistics
- Embedded norm distributions
- Interaction space analysis

### `analysis_runs/regenerate_figure_final.py`
**Final figure generation** (publication-ready)
- High-quality plots for paper submission
- Consistent styling and labeling

### `analysis_runs/regenerate_figure_v3.py`
**Version 3 of figure regeneration**
- Iterative improvement version
- May have alternative layouts

### `analysis_runs/regenerate_figure.py`
**Original figure regeneration attempt**

### `analysis_runs/generate_tikz.py`
**Generate TikZ plots** (for LaTeX integration)
- Can output vector graphics for paper
- Exact font matching with LaTeX

### `analysis_runs/submit_sweep_slurm.py`
**Submit sweep experiments to SLURM cluster**
- Generates SLURM job submission scripts
- Manages job queue
- Monitors completion

---

## Documentation Files

### `README.md` ⭐ (Start Here!)
Comprehensive guide covering:
- Project overview & motivation
- Installation & setup
- Usage examples
- Model architecture details
- Key results summary
- Dataset information
- Citation information

### `QUICKSTART.md`
**Get running in 60 seconds**
- Quick setup instructions
- Minimal reproducible examples
- Common use cases
- Debugging tips

### `INSTALL.md`
**Detailed installation guide**
- Conda / pip / venv setup
- GPU/CPU installation options
- Dependency resolution
- Troubleshooting common issues

### `PAPER_REVISION_SUMMARY.md` (Reference)
Editorial notes from paper revision process

### `ICML_EXPERIMENT_RESULTS.md` (Reference)
Summary of submission experiments

### `COMPLETION_REPORT.md` (Reference)
Project completion status

---

## Directory Structure

```
github/
├── src/                          # Core implementation
│   ├── train.py                  # ⭐ Main training script
│   ├── train2.py, train3.py     # Alternative variants
│   ├── analysis*.py              # Analysis utilities
│   ├── check_embeddings.py       # Embedding validation
│   ├── gen.py                    # Data generation
│   └── ...
│
├── experiments/                  # Experimental aggregation
│   ├── collect_core4_plus2.py   # ⭐ Core4+2 results
│   └── collect_alignment_results.py
│
├── analysis_runs/                    # submission analysis & viz
│   ├── make_sweep_commands.py   # ⭐ Generate sweep jobs
│   ├── generate_sweep_plots.py  # ⭐ Noise/supervision plots
│   ├── plot_enhanced_geometry.py # ⭐ Alignment geometry (Fig 4)
│   ├── biological_interpretability.py  # ⭐ Validation (Fig 5)
│   ├── plot_results.py           # General plotting
│   ├── collect_results.py        # Aggregate results
│   └── ...
│
├── README.md                     # ⭐ Start here!
├── QUICKSTART.md                 # Quick setup (60s)
├── INSTALL.md                    # Detailed installation
├── requirements.txt              # Dependencies
└── LICENSE
```

---

## Quick Reference

### Running Experiments

1. **Single TRACE run:**
   ```bash
   python src/train.py --model_name TRACE --epochs 100
   ```

2. **Generate sweep commands:**
   ```bash
   python analysis_runs/make_sweep_commands.py --write sweep.sh
   bash sweep.sh
   ```

3. **Collect results:**
   ```bash
   python experiments/collect_core4_plus2.py
   ```

4. **Generate plots:**
   ```bash
   python analysis_runs/generate_sweep_plots.py
   python analysis_runs/plot_enhanced_geometry.py
   python analysis_runs/biological_interpretability.py
   ```

### Key Output Files

- **Model checkpoint:** `logs/best_model.pt`
- **Metrics:** `logs/metrics.json`
- **Results table:** `results/comparison.csv`
- **Figures:** `figures/*.png`

### Important Classes/Functions

(From main train.py)
- `TraceModel` - Main architecture
- `SequenceTower` - Seq embedding projection
- `GraphTower` - GNN residue graph encoder
- `BindingClassifier` - Final interaction head
- `train()` - Main training loop
- `evaluate()` - Validation/test metrics

---

## Paper Figures Mapping

| Figure | Script(s) | Purpose |
|--------|-----------|---------|
| Fig 1  | - | Architecture diagram (drawn manually) |
| Fig 2  | - | Data/protocol comparison |
| Fig 3 (a-b) | `generate_sweep_plots.py` | Noise & supervision sweeps |
| Fig 3 (c) | `plot_enhanced_geometry.py` | Alignment quality |
| Fig 3 (d) | `generate_sweep_plots.py` | Model complexity robustness |
| Fig 4 | `plot_enhanced_geometry.py` | Cosine similarity violins |
| Fig 5 | `biological_interpretability.py` | Binding pair discrimination |

---

**Last Updated:** April 2024

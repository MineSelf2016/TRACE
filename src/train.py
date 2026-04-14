import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    auc, precision_score, recall_score, f1_score, roc_auc_score,
    accuracy_score, precision_recall_curve
)
import torch.nn as nn
import torch.optim as optim
import argparse
import random
import os
from datetime import datetime
import logging
from pathlib import Path
import itertools
import glob
import pickle
import json
from typing import Dict, Any, List, Tuple
import platform
import sys
import time

# Backward compatibility for pickle files saved with older numpy module paths.
sys.modules.setdefault("numpy._core", np.core)
sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
sys.modules.setdefault("numpy._core.umath", np.core.umath)

# -----------------------
# Reproducibility
# -----------------------
seed = 12

# ROOT_PATH = Path(Path.cwd()).resolve().parent
ROOT_PATH = Path(Path.cwd()).resolve()
NOW_TIME = datetime.now().strftime("%Y%m%d_%H%M")


def init_logger(log_filename):
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # avoid duplicated handlers in some environments
    if logger.handlers:
        return logger

    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


os.makedirs("logs", exist_ok=True)
log_filename = os.path.join("logs", f"{NOW_TIME}.log")
logger = init_logger(log_filename)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"running on {device}")


# -----------------------
# Args
# -----------------------
parser = argparse.ArgumentParser(description="Multimodal TCR–peptide binding (Seq+Graph + CLIP alignment).")
parser.add_argument('--train_base', type=str, default=os.path.join(ROOT_PATH, "dataset/ds.hard-splits/pep+cdr3b"),
                    help='Root path of the train dataset')
parser.add_argument('--embedbase', type=str, default=os.path.join(ROOT_PATH, "embs"),
                    help='Path to the embeddings output directory')
parser.add_argument('--mode', type=str, default="only-sampled-negs", help='random or hard split')
parser.add_argument('--results_dir', type=str, default="./multimodal_clip_binding_results",
                    help='Path to save outputs')

# model dims
parser.add_argument('--proj_dim', type=int, default=256, help='Projection dim for CLIP space')
parser.add_argument('--graph_hidden', type=int, default=256, help='Hidden dim inside graph encoder')
parser.add_argument('--graph_layers', type=int, default=2, help='Number of message passing layers')

# losses
parser.add_argument('--lambda_clip', type=float, default=0.2, help='Weight for CLIP alignment loss')
parser.add_argument('--lambda_bind', type=float, default=1.0, help='Weight for binding classification loss')
parser.add_argument('--temperature', type=float, default=0.07, help='CLIP temperature')
parser.add_argument('--alignment_mode', type=str, default="infonce",
                    choices=["none", "mse", "cosine", "infonce"],
                    help='Alignment method for multimodal model: none, mse, cosine, infonce')
parser.add_argument('--model_variant', type=str, default="multimodal",
                    choices=["multimodal", "seq_mlp", "seq_linear"],
                    help='Model variant: multimodal (Seq+Graph), seq_mlp (sequence-only MLP head), seq_linear (sequence-only linear head)')

# training
parser.add_argument('--dropout', type=float, default=0.2, help='drop out rate')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate')
parser.add_argument('--num_epochs', type=int, default=60, help='max epochs')
parser.add_argument('--patience', type=int, default=20, help='early stop patience')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay')
parser.add_argument('--optimizer', type=str, default="adam", choices=["adam", "sgd"], help='optimizer type')

# experiment bookkeeping
parser.add_argument('--seed', type=int, default=12, help='Random seed for reproducibility')
parser.add_argument('--run_name', type=str, default='',
                    help='Unique run name. If empty, a name is derived from key args to avoid overwrites.')
parser.add_argument('--resume', action='store_true',
                    help='Resume training from output_dir/checkpoint_last.pth if it exists.')

# grid (optional)
parser.add_argument('--do_grid', action='store_true', help='enable hyperparam grid search')

# allow selecting dataset index from CLI
parser.add_argument('--dataset_index', type=int, default=1, help='dataset index to run')
parser.add_argument('--few_shot', type=int, default=1, help='few shot setting index')
parser.add_argument('--frac', type=float, default=0.0, help='fraction index for few-shot')

# submission: controlled sweeps (do not change defaults)
parser.add_argument('--edge_dropout', type=float, default=0.0,
                    help='Drop this fraction of graph edges during TRAINING only (noise sweep).')
parser.add_argument('--pos_downsample', type=float, default=1.0,
                    help='Keep this fraction of positive training pairs (supervision sweep). 1.0 keeps all.')
parser.add_argument('--neg_downsample', type=float, default=1.0,
                    help='Keep this fraction of negative training pairs (optional; default keeps all).')

# CPU demo / timing controls (optional)
parser.add_argument('--max_train_batches', type=int, default=0,
                    help='If >0, limit number of train batches per epoch (for quick CPU demo / timing).')
parser.add_argument('--max_eval_batches', type=int, default=0,
                    help='If >0, limit number of eval batches in val/test (for quick CPU demo / timing).')

# performance knobs (CPU/HPC)
parser.add_argument('--num_workers', type=int, default=0,
                    help='DataLoader num_workers. For CPU Slurm runs, try 2-4 to utilize allocated CPUs.')
parser.add_argument('--torch_num_threads', type=int, default=0,
                    help='torch.set_num_threads(). 0 keeps default; on CPU Slurm we auto-set from SLURM_CPUS_PER_TASK.')
parser.add_argument('--torch_num_interop_threads', type=int, default=0,
                    help='torch.set_num_interop_threads(). 0 keeps default.')

args = parser.parse_args()
logger.info("args:\n" + json.dumps(vars(args), indent=2))

# Try to respect Slurm CPU allocation on CPU runs.
if device.type == "cpu":
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if args.torch_num_threads and int(args.torch_num_threads) > 0:
        torch.set_num_threads(int(args.torch_num_threads))
    elif slurm_cpus and slurm_cpus.isdigit():
        torch.set_num_threads(int(slurm_cpus))

    if args.torch_num_interop_threads and int(args.torch_num_interop_threads) > 0:
        torch.set_num_interop_threads(int(args.torch_num_interop_threads))
    else:
        # Default interop threads can be surprisingly high on some clusters;
        # keep it small to reduce overhead for small/medium CPU jobs.
        if slurm_cpus and slurm_cpus.isdigit():
            torch.set_num_interop_threads(1)

logger.info(
    f"perf: torch_num_threads={torch.get_num_threads()} torch_num_interop_threads={torch.get_num_interop_threads()} "
    f"num_workers={args.num_workers}"
)

# Apply seeds after parsing args
seed = int(args.seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

results_dir = args.results_dir
os.makedirs(results_dir, exist_ok=True)

mode = args.mode
embedbase = os.path.join(args.embedbase)

TCR_col = "cdr3.beta"
PEPTIDE_col = "antigen.epitope"

# ============================================================
# 0) Load precomputed embeddings (Seq) and graphs (Graph)
# ============================================================
with open(os.path.join(embedbase, "peptide_seq_dict.pkl"), 'rb') as f:
    peptide_seq_dict = pickle.load(f)
print("Loaded peptide seq dict from ", os.path.join(embedbase, "peptide_seq_dict.pkl"))
with open(os.path.join(embedbase, "tcr_seq_dict.pkl"), 'rb') as f:
    tcrb_seq_dict = pickle.load(f)
print("Loaded tcr seq dict from ", os.path.join(embedbase, "tcr_seq_dict.pkl"))
"""
tcrb_graph_dict.pkl:  key = tcrb string
peptide_graph_dict.pkl: key = peptide string

每个 value 必须是一个 dict，至少包含：
{
  "x": np.ndarray or torch.Tensor, shape [num_nodes, node_feat_dim]
       - 推荐用 residue-level ESM embedding 作为节点特征（冻结即可）
       - 如果暂时没有 residue-level embedding，也可以先用 one-hot / physicochemical 特征
  "edge_index": np.ndarray or torch.LongTensor, shape [2, num_edges]
       - 0-indexed，双向边建议都放进去（i->j and j->i）
}
"""

graph_tcr_path = os.path.join(embedbase, "tcr_graph_dict.pkl")
graph_pep_path = os.path.join(embedbase, "peptide_graph_dict.pkl")

if not (os.path.exists(graph_tcr_path) and os.path.exists(graph_pep_path)):
    raise FileNotFoundError(
        "Missing graph dict pickles.\n"
        f"Expected:\n  {graph_tcr_path}\n  {graph_pep_path}\n\n"
        "Please generate residue graphs and save them as described in the comment block."
    )

with open(graph_tcr_path, 'rb') as f:
    tcrb_graph_dict = pickle.load(f)

with open(graph_pep_path, 'rb') as f:
    peptide_graph_dict = pickle.load(f)


# ============================================================
# 1) Dataset + Collate (variable-size graphs)
# ============================================================
class MultiModalPairDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        # tcrb = row["tcrb"]
        # pep = row["peptide"]
        tcrb = row[TCR_col]
        pep = row[PEPTIDE_col]
        y = int(row["label"])

        # seq embeddings (global)
        tcr_seq = np.asarray(tcrb_seq_dict[tcrb], dtype=np.float32)
        pep_seq = np.asarray(peptide_seq_dict[pep], dtype=np.float32)

        # graph inputs
        tcr_g = tcrb_graph_dict[tcrb]
        pep_g = peptide_graph_dict[pep]

        # x: [N, F], edge_index: [2, E]
        tcr_x = np.asarray(tcr_g["x"], dtype=np.float32)
        tcr_edge = np.asarray(tcr_g["edge_index"], dtype=np.int64)

        pep_x = np.asarray(pep_g["x"], dtype=np.float32)
        pep_edge = np.asarray(pep_g["edge_index"], dtype=np.int64)

        sample = {
            "tcr_seq": torch.from_numpy(tcr_seq),
            "pep_seq": torch.from_numpy(pep_seq),
            "tcr_x": torch.from_numpy(tcr_x),
            "tcr_edge": torch.from_numpy(tcr_edge),
            "pep_x": torch.from_numpy(pep_x),
            "pep_edge": torch.from_numpy(pep_edge),
            "y": torch.tensor(y, dtype=torch.long),
        }
        return sample


def _batch_graph(graph_x_list: List[torch.Tensor],
                 edge_index_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batch variable graphs into one big disjoint graph.
    Returns:
      X: [sumN, F]
      edge_index: [2, sumE] with node offsets applied
      batch: [sumN] graph id per node (0..B-1)
    """
    xs = []
    eis = []
    batch = []

    node_offset = 0
    for g_id, (x, ei) in enumerate(zip(graph_x_list, edge_index_list)):
        n = x.size(0)
        xs.append(x)
        batch.append(torch.full((n,), g_id, dtype=torch.long))

        ei2 = ei.clone()
        ei2 = ei2 + node_offset
        eis.append(ei2)

        node_offset += n

    X = torch.cat(xs, dim=0)
    edge_index = torch.cat(eis, dim=1) if len(eis) else torch.zeros((2, 0), dtype=torch.long)
    batch = torch.cat(batch, dim=0)
    return X, edge_index, batch


def _edge_dropout(edge_index: torch.Tensor, p: float) -> torch.Tensor:
    """Randomly drop edges from a single graph.

    edge_index: [2, E]
    p: dropout probability in [0,1)
    """
    if p <= 0.0 or edge_index.numel() == 0:
        return edge_index
    if p >= 1.0:
        return edge_index[:, :0]

    E = edge_index.size(1)
    keep = torch.rand((E,), device=edge_index.device) >= p
    return edge_index[:, keep]


def _collate_fn_impl(batch_samples: List[Dict[str, Any]], *, edge_dropout: float) -> Dict[str, Any]:
    tcr_seq = torch.stack([b["tcr_seq"] for b in batch_samples], dim=0)  # [B, Dseq]
    pep_seq = torch.stack([b["pep_seq"] for b in batch_samples], dim=0)  # [B, Dseq]
    y = torch.stack([b["y"] for b in batch_samples], dim=0)              # [B]

    # Apply edge dropout per-graph (training-time noise only).
    if edge_dropout > 0.0:
        tcr_edges = [_edge_dropout(b["tcr_edge"], edge_dropout) for b in batch_samples]
        pep_edges = [_edge_dropout(b["pep_edge"], edge_dropout) for b in batch_samples]
    else:
        tcr_edges = [b["tcr_edge"] for b in batch_samples]
        pep_edges = [b["pep_edge"] for b in batch_samples]

    tcr_x, tcr_edge, tcr_batch = _batch_graph(
        [b["tcr_x"] for b in batch_samples],
        tcr_edges
    )
    pep_x, pep_edge, pep_batch = _batch_graph(
        [b["pep_x"] for b in batch_samples],
        pep_edges
    )

    return {
        "tcr_seq": tcr_seq,
        "pep_seq": pep_seq,
        "tcr_x": tcr_x, "tcr_edge": tcr_edge, "tcr_batch": tcr_batch,
        "pep_x": pep_x, "pep_edge": pep_edge, "pep_batch": pep_batch,
        "y": y
    }


def make_collate_fn(*, edge_dropout: float):
    return lambda batch_samples: _collate_fn_impl(batch_samples, edge_dropout=edge_dropout)


# ============================================================
# 2) Encoders: Seq tower + Graph tower (simple message passing)
# ============================================================
class SeqTower(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleMPNNLayer(nn.Module):
    """
    A lightweight message passing layer without torch_geometric.
    Aggregation: mean of neighbor node features.
    """
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.lin_self = nn.Linear(dim, dim)
        self.lin_nei = nn.Linear(dim, dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x: [N, D], edge_index: [2, E]
        N, D = x.size()
        if edge_index.numel() == 0:
            out = self.lin_self(x)
            return self.norm(self.drop(self.act(out)))

        src = edge_index[0]  # [E]
        dst = edge_index[1]  # [E]

        # sum neighbor messages
        agg = torch.zeros((N, D), device=x.device, dtype=x.dtype)
        agg.index_add_(0, dst, x[src])

        # mean by degree
        deg = torch.zeros((N,), device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        deg = torch.clamp(deg, min=1.0).unsqueeze(-1)
        agg = agg / deg

        out = self.lin_self(x) + self.lin_nei(agg)
        out = self.norm(self.drop(self.act(out)))
        return out


class GraphTower(nn.Module):
    def __init__(self, node_in_dim: int, hidden_dim: int, proj_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.layers = nn.ModuleList([SimpleMPNNLayer(hidden_dim, dropout) for _ in range(num_layers)])
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        x: [sumN, Fin], edge_index: [2, sumE], batch: [sumN] in [0..B-1]
        returns: graph embedding [B, proj_dim]
        """
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h, edge_index)

        # global mean pool per graph id
        B = int(batch.max().item()) + 1 if batch.numel() else 0
        if B == 0:
            return torch.zeros((0, self.out_proj[-1].normalized_shape[0]), device=x.device)

        pooled = torch.zeros((B, h.size(-1)), device=h.device, dtype=h.dtype)
        pooled.index_add_(0, batch, h)

        counts = torch.zeros((B,), device=h.device, dtype=h.dtype)
        counts.index_add_(0, batch, torch.ones_like(batch, dtype=h.dtype))
        pooled = pooled / torch.clamp(counts.unsqueeze(-1), min=1.0)

        z = self.out_proj(pooled)
        return z


# ============================================================
# 3) Alignment losses + Binding head
# ============================================================
def infonce_loss(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    z_a, z_b: [B, D]
    InfoNCE / CLIP loss (symmetric).
    """
    z_a = nn.functional.normalize(z_a, dim=-1)
    z_b = nn.functional.normalize(z_b, dim=-1)
    logits = (z_a @ z_b.t()) / temperature  # [B, B]
    labels = torch.arange(z_a.size(0), device=z_a.device)
    loss_ab = nn.functional.cross_entropy(logits, labels)
    loss_ba = nn.functional.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_ab + loss_ba)


def mse_align_loss(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(z_a, z_b)


def cosine_align_loss(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    z_a = nn.functional.normalize(z_a, dim=-1)
    z_b = nn.functional.normalize(z_b, dim=-1)
    return -torch.mean(torch.sum(z_a * z_b, dim=1))


def get_alignment_loss(alignment_mode: str, z_a: torch.Tensor, z_b: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    if alignment_mode == "none":
        return torch.tensor(0.0, device=z_a.device, dtype=z_a.dtype)
    if alignment_mode == "mse":
        return mse_align_loss(z_a, z_b)
    if alignment_mode == "cosine":
        return cosine_align_loss(z_a, z_b)
    if alignment_mode == "infonce":
        return infonce_loss(z_a, z_b, temperature)
    raise ValueError(f"Unknown alignment_mode: {alignment_mode}")


def clip_loss(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float) -> torch.Tensor:
    return infonce_loss(z_a, z_b, temperature)


class BindingHead(nn.Module):
    """
    Use both entity embeddings (fused seq+graph) for binding classification.
    """
    def __init__(self, d: int, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 * d, hidden),  # [t, p, |t-p|, t*p]
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([t, p, torch.abs(t - p), t * p], dim=-1)
        return self.net(feat)


class LinearBindingHead(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.fc = nn.Linear(4 * d, 2)

    def forward(self, t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([t, p, torch.abs(t - p), t * p], dim=-1)
        return self.fc(feat)


class MultiModalBindingModel(nn.Module):
    def __init__(self,
                 tcr_seq_dim: int,
                 pep_seq_dim: int,
                 tcr_node_dim: int,
                 pep_node_dim: int,
                 proj_dim: int,
                 graph_hidden: int,
                 graph_layers: int,
                 dropout: float):
        super().__init__()
        # Seq towers -> CLIP space
        self.tcr_seq_tower = SeqTower(tcr_seq_dim, proj_dim, dropout)
        self.pep_seq_tower = SeqTower(pep_seq_dim, proj_dim, dropout)

        # Graph towers -> CLIP space
        self.tcr_graph_tower = GraphTower(tcr_node_dim, graph_hidden, proj_dim, graph_layers, dropout)
        self.pep_graph_tower = GraphTower(pep_node_dim, graph_hidden, proj_dim, graph_layers, dropout)

        # fuse within entity (seq + graph)
        self.tcr_fuse = nn.Sequential(
            nn.Linear(2 * proj_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(proj_dim),
        )
        self.pep_fuse = nn.Sequential(
            nn.Linear(2 * proj_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(proj_dim),
        )

        self.binding_head = BindingHead(proj_dim, hidden=256, dropout=dropout)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        tcr_seq = batch["tcr_seq"]
        pep_seq = batch["pep_seq"]

        tcr_x, tcr_edge, tcr_b = batch["tcr_x"], batch["tcr_edge"], batch["tcr_batch"]
        pep_x, pep_edge, pep_b = batch["pep_x"], batch["pep_edge"], batch["pep_batch"]

        z_tcr_seq = self.tcr_seq_tower(tcr_seq)  # [B, D]
        z_pep_seq = self.pep_seq_tower(pep_seq)  # [B, D]
        z_tcr_g = self.tcr_graph_tower(tcr_x, tcr_edge, tcr_b)  # [B, D]
        z_pep_g = self.pep_graph_tower(pep_x, pep_edge, pep_b)  # [B, D]

        tcr = self.tcr_fuse(torch.cat([z_tcr_seq, z_tcr_g], dim=-1))  # [B, D]
        pep = self.pep_fuse(torch.cat([z_pep_seq, z_pep_g], dim=-1))  # [B, D]

        logits = self.binding_head(tcr, pep)  # [B, 2]
        return {
            "logits": logits,
            "z_tcr_seq": z_tcr_seq, "z_tcr_g": z_tcr_g,
            "z_pep_seq": z_pep_seq, "z_pep_g": z_pep_g,
            "tcr": tcr, "pep": pep
        }


class SeqOnlyBindingModel(nn.Module):
    def __init__(self,
                 tcr_seq_dim: int,
                 pep_seq_dim: int,
                 proj_dim: int,
                 dropout: float,
                 head_type: str = "mlp"):
        super().__init__()
        self.tcr_seq_tower = SeqTower(tcr_seq_dim, proj_dim, dropout)
        self.pep_seq_tower = SeqTower(pep_seq_dim, proj_dim, dropout)
        if head_type == "linear":
            self.binding_head = LinearBindingHead(proj_dim)
        else:
            self.binding_head = BindingHead(proj_dim, hidden=256, dropout=dropout)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        z_tcr_seq = self.tcr_seq_tower(batch["tcr_seq"])
        z_pep_seq = self.pep_seq_tower(batch["pep_seq"])
        logits = self.binding_head(z_tcr_seq, z_pep_seq)
        return {
            "logits": logits,
            "tcr": z_tcr_seq,
            "pep": z_pep_seq,
        }


# ============================================================
# 4) Data loading (csv) -> Dataset
# ============================================================
def load_split_dfs(dataset_index: int):
    train_path = args.train_base
    # /project/zhiwei/g013857/TCR-Multi/dataset/ds.hard-splits/pep+cdr3b/train/only-sampled-negs/train-0.csv
    # train_file = f'{train_path}/train/{mode}/train-{dataset_index}.csv'
    if args.frac > 0:
        train_file = f'{train_path}2/train/{mode}/train-{dataset_index}-{args.frac}.csv'
    else:
        train_file = f'{train_path}2/train/{mode}/train-{dataset_index}.csv'

    train_df = pd.read_csv(train_file, low_memory=False).drop_duplicates()
    values_to_remove = ["CASSQETDIVFNXPQHF", "CASSLRTRTDTQYX", "CASSILGWSEAFX", "CSARTGDRTEAFX", "CASSQETDIVFNOPQHF"]
    train_df = train_df[~train_df[TCR_col].isin(values_to_remove)] # remove bad data entry
    logger.info(f"train file: {train_file}")

    # Optional supervision sweep: downsample positives/negatives in TRAIN only.
    # This keeps val/test unchanged so results reflect supervision strength.
    if args.pos_downsample < 1.0 or args.neg_downsample < 1.0:
        pos_df = train_df[train_df["label"].astype(int) == 1]
        neg_df = train_df[train_df["label"].astype(int) == 0]

        if args.pos_downsample < 1.0:
            pos_df = pos_df.sample(frac=float(args.pos_downsample), random_state=seed)
        if args.neg_downsample < 1.0:
            neg_df = neg_df.sample(frac=float(args.neg_downsample), random_state=seed)

        train_df = pd.concat([pos_df, neg_df], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        logger.info(
            f"Applied train downsampling: pos_frac={args.pos_downsample} neg_frac={args.neg_downsample} -> n={len(train_df)}"
        )
    
    # No validation split available; set val_df to None
    val_df = None
    test_file = f'{train_path}/test/{mode}/test-{dataset_index}.csv'
    test_df = pd.read_csv(test_file, low_memory=False)
    test_df = test_df[~test_df[TCR_col].isin(values_to_remove)] # remove bad data entry
    
    
    return train_df, val_df, test_df


# ============================================================
# 5) Train / Eval
# ============================================================
@torch.no_grad()
def eval_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int = 0,
):
    model.eval()
    probs = []
    ys = []
    running_loss = 0.0

    for b_idx, batch in enumerate(loader):
        if max_batches and b_idx >= max_batches:
            break
        # move
        for k in batch:
            v = batch[k]
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        out = model(batch)
        logits = out["logits"]
        y = batch["y"]
        loss = criterion(logits, y)
        running_loss += loss.item() * y.size(0)

        prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        probs.extend(prob)
        ys.extend(y.detach().cpu().numpy())

    ys = np.asarray(ys)
    probs = np.asarray(probs)
    denom = max(1, len(ys))
    avg_loss = running_loss / denom

    auroc = roc_auc_score(ys, probs) if len(np.unique(ys)) > 1 else float("nan")
    precision, recall, _ = precision_recall_curve(ys, probs)
    aupr = auc(recall, precision)

    preds = (probs > 0.5).astype(int)
    metrics = {
        "Loss": avg_loss,
        "AUROC": auroc,
        "Accuracy": accuracy_score(ys, preds),
        "Recall": recall_score(ys, preds, zero_division=0),
        "Precision": precision_score(ys, preds, zero_division=0),
        "F1": f1_score(ys, preds, zero_division=0),
        "AUPR": aupr,
    }
    return metrics, ys, probs


def train_one_run(dataset_index: int,
                  param_dict: Dict[str, Any],
                  output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Dataset {dataset_index} params={param_dict}")

    # Persist run metadata for paper reproducibility.
    meta = {
        "timestamp": NOW_TIME,
        "run_name": args.run_name,
        "dataset_index": dataset_index,
        "seed": seed,
        "cmd": " ".join(sys.argv),
        "cwd": str(Path.cwd()),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": getattr(torch, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
        "args": vars(args),
        "param_dict": param_dict,
    }
    with open(os.path.join(output_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    train_df, val_df, test_df = load_split_dfs(dataset_index)

    train_ds = MultiModalPairDataset(train_df)
    test_ds = MultiModalPairDataset(test_df)

    # If no validation dataframe is provided, fall back to using the test set
    # as the validation set to allow training to proceed.
    if val_df is None:
        val_ds = test_ds
        logger.info(f"No validation set found for dataset {dataset_index}; using test set as validation.")
    else:
        val_ds = MultiModalPairDataset(val_df)

    train_loader = DataLoader(train_ds, batch_size=param_dict["batch_size"], shuffle=True,
                                                            drop_last=True,
                                                            num_workers=int(args.num_workers),
                                                            persistent_workers=bool(args.num_workers and int(args.num_workers) > 0),
                                                            collate_fn=make_collate_fn(edge_dropout=param_dict.get("edge_dropout", 0.0)))
    val_loader = DataLoader(val_ds, batch_size=param_dict["batch_size"], shuffle=False,
                                                        num_workers=int(args.num_workers),
                                                        persistent_workers=bool(args.num_workers and int(args.num_workers) > 0),
                                                        collate_fn=make_collate_fn(edge_dropout=0.0))
    test_loader = DataLoader(test_ds, batch_size=param_dict["batch_size"], shuffle=False,
                                                         num_workers=int(args.num_workers),
                                                         persistent_workers=bool(args.num_workers and int(args.num_workers) > 0),
                                                         collate_fn=make_collate_fn(edge_dropout=0.0))

    # infer dims from one sample
    sample0 = train_ds[0]
    tcr_seq_dim = sample0["tcr_seq"].numel()
    pep_seq_dim = sample0["pep_seq"].numel()
    tcr_node_dim = sample0["tcr_x"].size(-1)
    pep_node_dim = sample0["pep_x"].size(-1)

    if args.model_variant == "multimodal":
        model = MultiModalBindingModel(
            tcr_seq_dim=tcr_seq_dim,
            pep_seq_dim=pep_seq_dim,
            tcr_node_dim=tcr_node_dim,
            pep_node_dim=pep_node_dim,
            proj_dim=param_dict["proj_dim"],
            graph_hidden=param_dict["graph_hidden"],
            graph_layers=param_dict["graph_layers"],
            dropout=param_dict["dropout"]
        ).to(device)
    elif args.model_variant == "seq_mlp":
        model = SeqOnlyBindingModel(
            tcr_seq_dim=tcr_seq_dim,
            pep_seq_dim=pep_seq_dim,
            proj_dim=param_dict["proj_dim"],
            dropout=param_dict["dropout"],
            head_type="mlp",
        ).to(device)
    elif args.model_variant == "seq_linear":
        model = SeqOnlyBindingModel(
            tcr_seq_dim=tcr_seq_dim,
            pep_seq_dim=pep_seq_dim,
            proj_dim=param_dict["proj_dim"],
            dropout=param_dict["dropout"],
            head_type="linear",
        ).to(device)
    else:
        raise ValueError(f"Unknown model_variant: {args.model_variant}")

    # class weights (binding labels)
    y_train = train_df["label"].values
    class_weights = torch.tensor(
        compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train),
        dtype=torch.float32, device=device
    )
    ce = nn.CrossEntropyLoss(weight=class_weights)

    if param_dict["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=param_dict["learning_rate"],
                               weight_decay=param_dict["weight_decay"])
    else:
        optimizer = optim.SGD(model.parameters(), lr=param_dict["learning_rate"],
                              momentum=0.9, weight_decay=param_dict["weight_decay"])

    # NOTE: ReduceLROnPlateau expects a metric; we step on val AUROC (higher better)
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.1, verbose=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.1)

    best_val_auroc = -1.0
    patience = args.patience
    counter = 0
    best_path = os.path.join(output_dir, f"best_model_{dataset_index}.pth")
    ckpt_last_path = os.path.join(output_dir, "checkpoint_last.pth")
    ckpt_best_path = os.path.join(output_dir, "checkpoint_best.pth")
    start_epoch = 0

    # Optional resume (for 12h Slurm wall time).
    if args.resume and os.path.exists(ckpt_last_path):
        logger.info(f"Resuming from checkpoint: {ckpt_last_path}")
        ckpt = torch.load(ckpt_last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_auroc = float(ckpt.get("best_val_auroc", -1.0))
        counter = int(ckpt.get("counter", 0))
        logger.info(
            f"Resumed: start_epoch={start_epoch} best_val_auroc={best_val_auroc:.4f} counter={counter}"
        )
        # If best model path exists, keep it; otherwise, write best_path from checkpoint if present.
        if "best_state_dict" in ckpt and not os.path.exists(best_path):
            torch.save(ckpt["best_state_dict"], best_path)

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        running = 0.0

        # track component losses for analysis (per-epoch averages)
        sum_bind = 0.0
        sum_clip_tcr = 0.0
        sum_clip_pep = 0.0
        sum_clip = 0.0

        t_train0 = time.time()
        n_train_samples = 0
        n_train_batches = 0

        for b_idx, batch in enumerate(train_loader):
            if args.max_train_batches and b_idx >= args.max_train_batches:
                break
            for k in batch:
                v = batch[k]
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            out = model(batch)
            logits = out["logits"]
            y = batch["y"]

            # binding loss
            loss_bind = ce(logits, y)

            # Alignment within each entity: (seq <-> graph), only for multimodal outputs
            has_alignment_views = all(k in out for k in ["z_tcr_seq", "z_tcr_g", "z_pep_seq", "z_pep_g"])
            if args.model_variant == "multimodal" and has_alignment_views:
                loss_clip_tcr = get_alignment_loss(args.alignment_mode, out["z_tcr_seq"], out["z_tcr_g"], param_dict["temperature"])
                loss_clip_pep = get_alignment_loss(args.alignment_mode, out["z_pep_seq"], out["z_pep_g"], param_dict["temperature"])
                loss_clip = 0.5 * (loss_clip_tcr + loss_clip_pep)
            else:
                loss_clip_tcr = torch.tensor(0.0, device=device)
                loss_clip_pep = torch.tensor(0.0, device=device)
                loss_clip = torch.tensor(0.0, device=device)

            loss = param_dict["lambda_bind"] * loss_bind + param_dict["lambda_clip"] * loss_clip

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item() * y.size(0)

            # accumulate raw (unweighted) component losses
            bs = int(y.size(0))
            sum_bind += float(loss_bind.item()) * bs
            sum_clip_tcr += float(loss_clip_tcr.item()) * bs
            sum_clip_pep += float(loss_clip_pep.item()) * bs
            sum_clip += float(loss_clip.item()) * bs

            n_train_samples += int(y.size(0))
            n_train_batches += 1

        t_train1 = time.time()
        train_loss = running / max(1, n_train_samples)
        avg_bind = sum_bind / max(1, n_train_samples)
        avg_clip_tcr = sum_clip_tcr / max(1, n_train_samples)
        avg_clip_pep = sum_clip_pep / max(1, n_train_samples)
        avg_clip = sum_clip / max(1, n_train_samples)

        t_eval0 = time.time()
        val_metrics, _, _ = eval_model(
            model,
            val_loader,
            ce,
            device,
            max_batches=int(args.max_eval_batches or 0),
        )
        t_eval1 = time.time()
        val_auroc = val_metrics["AUROC"]

        logger.info(
            (
                f"Epoch {epoch+1:03d} | train_loss={train_loss:.4f} "
                f"(bind={avg_bind:.4f}, clip={avg_clip:.4f}; tcr={avg_clip_tcr:.4f}, pep={avg_clip_pep:.4f}) "
                f"| val_loss={val_metrics['Loss']:.4f} val_AUROC={val_auroc:.4f} val_AUPR={val_metrics['AUPR']:.4f}"
                f" | train_time={t_train1-t_train0:.1f}s ({n_train_batches} batches)"
                f" eval_time={t_eval1-t_eval0:.1f}s (max_eval_batches={args.max_eval_batches})"
            )
        )

        # append to per-epoch CSV
        try:
            comp_row = {
                "epoch": int(epoch + 1),
                "train_loss": float(train_loss),
                "bind": float(avg_bind),
                "clip": float(avg_clip),
                "clip_tcr": float(avg_clip_tcr),
                "clip_pep": float(avg_clip_pep),
                "val_loss": float(val_metrics["Loss"]),
                "val_AUROC": float(val_auroc),
                "val_AUPR": float(val_metrics.get("AUPR", float("nan"))),
            }
            comp_path = os.path.join(output_dir, "loss_components.csv")
            if not os.path.exists(comp_path):
                with open(comp_path, "w", encoding="utf-8") as f:
                    f.write(
                        "epoch,train_loss,bind,clip,clip_tcr,clip_pep,val_loss,val_AUROC,val_AUPR\n"
                    )
            with open(comp_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{comp_row['epoch']},{comp_row['train_loss']:.6f},{comp_row['bind']:.6f},{comp_row['clip']:.6f},{comp_row['clip_tcr']:.6f},{comp_row['clip_pep']:.6f},{comp_row['val_loss']:.6f},{comp_row['val_AUROC']:.6f},{comp_row['val_AUPR']:.6f}\n"
                )
        except Exception as _:
            pass

        # early stop on AUROC
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            torch.save(model.state_dict(), best_path)
            counter = 0

            # Save a "best" checkpoint for robustness.
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_val_auroc": best_val_auroc,
                    "counter": counter,
                    "seed": seed,
                    "param_dict": param_dict,
                    "args": vars(args),
                },
                ckpt_best_path,
            )
        else:
            counter += 1
            if counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}. Best val_AUROC={best_val_auroc:.4f}")
                break

        scheduler.step(val_auroc)

        # Always write a last checkpoint so we can resume after walltime.
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_auroc": best_val_auroc,
                "counter": counter,
                "seed": seed,
                "param_dict": param_dict,
                "args": vars(args),
                # optional convenience: keep a copy of best weights
                "best_state_dict": torch.load(best_path, map_location=device) if os.path.exists(best_path) else None,
            },
            ckpt_last_path,
        )

    # test eval
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics, y_true, y_prob = eval_model(
        model,
        test_loader,
        ce,
        device,
        max_batches=int(args.max_eval_batches or 0),
    )

    logger.info(f"[TEST] dataset={dataset_index} metrics={test_metrics}")

    # save metrics
    metrics_df = pd.DataFrame([
        {"metrics": k, "score": float(v), "experiment": dataset_index, **param_dict}
        for k, v in test_metrics.items()
    ])
    metrics_df.to_csv(os.path.join(output_dir, f"evaluation_{dataset_index}.csv"), index=False)

    return best_path, test_metrics


# ============================================================
# 6) Main (single run OR grid)
# ============================================================
dataset_indices = [args.dataset_index]   # set via --dataset_index

base_param = {
    "proj_dim": args.proj_dim,
    "graph_hidden": args.graph_hidden,
    "graph_layers": args.graph_layers,
    "dropout": args.dropout,
    "learning_rate": args.learning_rate,
    "batch_size": args.batch_size,
    "weight_decay": args.weight_decay,
    "optimizer": args.optimizer,
    "lambda_clip": args.lambda_clip,
    "lambda_bind": args.lambda_bind,
    "temperature": args.temperature,
    "edge_dropout": args.edge_dropout,
    "model_variant": args.model_variant,
    "alignment_mode": args.alignment_mode,
}


def _default_run_name() -> str:
    # Include only a few keys that define the sweep axes and alignment setting.
    return (
        f"ds{args.dataset_index}_mode{args.mode}"
        f"_mv{args.model_variant}_am{args.alignment_mode}"
        f"_clip{args.lambda_clip}_ed{args.edge_dropout}"
        f"_pos{args.pos_downsample}_neg{args.neg_downsample}"
        f"_seed{seed}_{NOW_TIME}"
    )


if args.model_variant != "multimodal":
    if args.alignment_mode != "none":
        logger.info(f"model_variant={args.model_variant} has no graph view; overriding alignment_mode {args.alignment_mode} -> none")
        args.alignment_mode = "none"
        base_param["alignment_mode"] = "none"
    if float(args.lambda_clip) != 0.0:
        logger.info(f"model_variant={args.model_variant} has no graph view; overriding lambda_clip {args.lambda_clip} -> 0.0")
        args.lambda_clip = 0.0
        base_param["lambda_clip"] = 0.0


if not args.run_name:
    args.run_name = _default_run_name()

if not args.do_grid:
    # single run
    output_dir = os.path.join(results_dir, args.run_name)
    os.makedirs(output_dir, exist_ok=True)
    all_rows = []
    for di in dataset_indices:
        best_path, test_metrics = train_one_run(di, base_param, output_dir)
        for k, v in test_metrics.items():
            all_rows.append({"experiment": di, "metrics": k, "score": v, **base_param})
    pd.DataFrame(all_rows).to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    logger.info(f"Done. log file: {log_filename}")

else:
    # grid search (lightweight)
    param_grid = {
        "proj_dim": [256, 512],
        "graph_hidden": [256],
        "graph_layers": [2, 3],
        "dropout": [0.2, 0.3],
        "learning_rate": [0.001, 0.0005],
        "batch_size": [32, 64],
        "lambda_clip": [0.1, 0.2, 0.5],
        "temperature": [0.07],
        "optimizer": [args.optimizer],
        "weight_decay": [args.weight_decay],
        "lambda_bind": [1.0],
    }
    names = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))

    best_mean_auroc = -1
    best_params = None
    all_results = []

    for values in combos:
        p = dict(zip(names, values))
        param_str = "_".join([f"{k}-{v}" for k, v in p.items()])
        output_dir = os.path.join(results_dir, f"grid_{param_str}")
        os.makedirs(output_dir, exist_ok=True)

        aurocs = []
        for di in dataset_indices:
            _, test_metrics = train_one_run(di, p, output_dir)
            aurocs.append(test_metrics.get("AUROC", float("nan")))
            for k, v in test_metrics.items():
                all_results.append({"experiment": di, "metrics": k, "score": v, **p})

        mean_auroc = float(np.nanmean(aurocs))
        if mean_auroc > best_mean_auroc:
            best_mean_auroc = mean_auroc
            best_params = p.copy()

        logger.info(f"[GRID] {param_str} mean_AUROC={mean_auroc:.4f} best={best_mean_auroc:.4f}")

    pd.DataFrame(all_results).to_csv(os.path.join(results_dir, "all_grid_results.csv"), index=False)
    with open(os.path.join(results_dir, "best_params.txt"), "w") as f:
        f.write(json.dumps(best_params, indent=2))
    logger.info(f"Best mean AUROC={best_mean_auroc:.4f} best_params={best_params}")
    logger.info(f"Done. log file: {log_filename}")

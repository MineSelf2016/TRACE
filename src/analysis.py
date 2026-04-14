import os
import json
import pickle
import random
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score,
    brier_score_loss
)

import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA
from collections import Counter

# -----------------------
# Reproducibility
# -----------------------
def set_seed(seed: int = 12):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# NEW: Paper style + palette (more vivid, paper-friendly)
# ============================================================
def set_paper_style():
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
    })

PALETTE = {
    "pos": "#D81B60",     # vivid magenta
    "neg": "#1E88E5",     # vivid blue
    "tcr": "#8E24AA",     # purple
    "pep": "#00ACC1",     # cyan
    "accent": "#FB8C00",  # orange
    "diag": "#424242",    # dark gray
    "muted": "#9E9E9E",
    "ink": "#111111",
}

def _annotate_stats(ax, x, y=0.98, label=None):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        txt = f"{label}: n=0" if label else "n=0"
    else:
        txt = (
            f"{label}: n={x.size}, mean={x.mean():.3f}, med={np.median(x):.3f}, std={x.std(ddof=1):.3f}"
            if label
            else f"n={x.size}, mean={x.mean():.3f}, med={np.median(x):.3f}, std={x.std(ddof=1):.3f}"
        )
    ax.text(0.02, y, txt, transform=ax.transAxes, ha="left", va="top")

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.digitize(y_prob, bins) - 1
    ids = np.clip(ids, 0, n_bins - 1)
    ece = 0.0
    N = len(y_true)
    for i in range(n_bins):
        m = (ids == i)
        if m.sum() == 0:
            continue
        acc = y_true[m].mean()
        conf = y_prob[m].mean()
        ece += (m.sum() / max(1, N)) * abs(acc - conf)
    return float(ece)

def cohens_d(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    va = a.var(ddof=1)
    vb = b.var(ddof=1)
    sp = np.sqrt(((a.size - 1) * va + (b.size - 1) * vb) / max(1, (a.size + b.size - 2)))
    if sp < 1e-12:
        return float("nan")
    return float((a.mean() - b.mean()) / sp)

def violin_box_jitter(ax, data_list, labels, colors, ylabel, title):
    positions = np.arange(1, len(data_list) + 1)

    vp = ax.violinplot(data_list, positions=positions, showmeans=False, showmedians=False, showextrema=False)
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(colors[i])
        body.set_edgecolor("none")
        body.set_alpha(0.35)

    bp = ax.boxplot(
        data_list,
        positions=positions,
        widths=0.25,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": PALETTE["ink"], "linewidth": 1.2},
        whiskerprops={"color": "#444444", "linewidth": 1.0},
        capprops={"color": "#444444", "linewidth": 1.0},
    )
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(colors[i])
        box.set_alpha(0.55)
        box.set_edgecolor("none")

    rng = np.random.RandomState(0)
    for i, vals in enumerate(data_list):
        vals = np.asarray(vals, dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        x = rng.normal(loc=positions[i], scale=0.04, size=vals.size)
        ax.scatter(x, vals, s=10, alpha=0.55, linewidths=0, color=colors[i])

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


# ============================================================
# 1) Dataset + Collate (same as your training)
# ============================================================
class MultiModalPairDataset(torch.utils.data.Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 tcr_col: str,
                 pep_col: str,
                 tcrb_seq_dict: Dict[str, Any],
                 peptide_seq_dict: Dict[str, Any],
                 tcrb_graph_dict: Dict[str, Any],
                 peptide_graph_dict: Dict[str, Any]):
        self.df = df.reset_index(drop=True)
        self.tcr_col = tcr_col
        self.pep_col = pep_col
        self.tcrb_seq_dict = tcrb_seq_dict
        self.peptide_seq_dict = peptide_seq_dict
        self.tcrb_graph_dict = tcrb_graph_dict
        self.peptide_graph_dict = peptide_graph_dict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        tcrb = row[self.tcr_col]
        pep = row[self.pep_col]
        y = int(row["label"])

        # seq embeddings (global)
        tcr_seq = np.asarray(self.tcrb_seq_dict[tcrb], dtype=np.float32)
        pep_seq = np.asarray(self.peptide_seq_dict[pep], dtype=np.float32)

        # graph inputs
        tcr_g = self.tcrb_graph_dict[tcrb]
        pep_g = self.peptide_graph_dict[pep]

        tcr_x = np.asarray(tcr_g["x"], dtype=np.float32)
        tcr_edge = np.asarray(tcr_g["edge_index"], dtype=np.int64)

        pep_x = np.asarray(pep_g["x"], dtype=np.float32)
        pep_edge = np.asarray(pep_g["edge_index"], dtype=np.int64)

        sample = {
            "tcrb": tcrb,
            "pep": pep,
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


def collate_fn(batch_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    tcr_seq = torch.stack([b["tcr_seq"] for b in batch_samples], dim=0)
    pep_seq = torch.stack([b["pep_seq"] for b in batch_samples], dim=0)
    y = torch.stack([b["y"] for b in batch_samples], dim=0)

    tcr_x, tcr_edge, tcr_batch = _batch_graph(
        [b["tcr_x"] for b in batch_samples],
        [b["tcr_edge"] for b in batch_samples]
    )
    pep_x, pep_edge, pep_batch = _batch_graph(
        [b["pep_x"] for b in batch_samples],
        [b["pep_edge"] for b in batch_samples]
    )

    return {
        "tcr_seq": tcr_seq,
        "pep_seq": pep_seq,
        "tcr_x": tcr_x, "tcr_edge": tcr_edge, "tcr_batch": tcr_batch,
        "pep_x": pep_x, "pep_edge": pep_edge, "pep_batch": pep_batch,
        "y": y,
        "tcrb_list": [b["tcrb"] for b in batch_samples],
        "pep_list": [b["pep"] for b in batch_samples],
    }


# ============================================================
# 2) Model (copied from your training script)
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
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.lin_self = nn.Linear(dim, dim)
        self.lin_nei = nn.Linear(dim, dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        N, D = x.size()
        if edge_index.numel() == 0:
            out = self.lin_self(x)
            return self.norm(self.drop(self.act(out)))

        src = edge_index[0]
        dst = edge_index[1]

        agg = torch.zeros((N, D), device=x.device, dtype=x.dtype)
        agg.index_add_(0, dst, x[src])

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
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h, edge_index)

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


class BindingHead(nn.Module):
    def __init__(self, d: int, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 * d, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([t, p, torch.abs(t - p), t * p], dim=-1)
        return self.net(feat)


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
        self.tcr_seq_tower = SeqTower(tcr_seq_dim, proj_dim, dropout)
        self.pep_seq_tower = SeqTower(pep_seq_dim, proj_dim, dropout)

        self.tcr_graph_tower = GraphTower(tcr_node_dim, graph_hidden, proj_dim, graph_layers, dropout)
        self.pep_graph_tower = GraphTower(pep_node_dim, graph_hidden, proj_dim, graph_layers, dropout)

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

        z_tcr_seq = self.tcr_seq_tower(tcr_seq)
        z_pep_seq = self.pep_seq_tower(pep_seq)
        z_tcr_g = self.tcr_graph_tower(tcr_x, tcr_edge, tcr_b)
        z_pep_g = self.pep_graph_tower(pep_x, pep_edge, pep_b)

        tcr = self.tcr_fuse(torch.cat([z_tcr_seq, z_tcr_g], dim=-1))
        pep = self.pep_fuse(torch.cat([z_pep_seq, z_pep_g], dim=-1))

        logits = self.binding_head(tcr, pep)
        return {
            "logits": logits,
            "z_tcr_seq": z_tcr_seq, "z_tcr_g": z_tcr_g,
            "z_pep_seq": z_pep_seq, "z_pep_g": z_pep_g,
            "tcr": tcr, "pep": pep
        }


# ============================================================
# 3) Eval + helpers
# ============================================================
def save_pdf(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = torch.nn.functional.normalize(a, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)
    return (a * b).sum(dim=-1)


@torch.no_grad()
def predict_all(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    ys = []
    probs = []

    z_tcr_seq_all, z_tcr_g_all = [], []
    z_pep_seq_all, z_pep_g_all = [], []
    tcr_all, pep_all = [], []

    for batch in loader:
        y = batch["y"].numpy()
        for k in ["tcr_seq", "pep_seq", "tcr_x", "tcr_edge", "tcr_batch", "pep_x", "pep_edge", "pep_batch", "y"]:
            batch[k] = batch[k].to(device)

        out = model(batch)
        logits = out["logits"]
        prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

        ys.append(y)
        probs.append(prob)

        z_tcr_seq_all.append(out["z_tcr_seq"].detach().cpu())
        z_tcr_g_all.append(out["z_tcr_g"].detach().cpu())
        z_pep_seq_all.append(out["z_pep_seq"].detach().cpu())
        z_pep_g_all.append(out["z_pep_g"].detach().cpu())
        tcr_all.append(out["tcr"].detach().cpu())
        pep_all.append(out["pep"].detach().cpu())

    ys = np.concatenate(ys, axis=0)
    probs = np.concatenate(probs, axis=0)

    aux = {
        "z_tcr_seq": torch.cat(z_tcr_seq_all, dim=0),
        "z_tcr_g": torch.cat(z_tcr_g_all, dim=0),
        "z_pep_seq": torch.cat(z_pep_seq_all, dim=0),
        "z_pep_g": torch.cat(z_pep_g_all, dim=0),
        "tcr": torch.cat(tcr_all, dim=0),
        "pep": torch.cat(pep_all, dim=0),
    }
    return ys, probs, aux


def add_panel_label(ax: plt.Axes, label: str,
                    dx: float = -36, dy: float = 10):
    """
    Place panel label slightly outside the axes (in points),
    robust against content overlap in combined figures.
    dx, dy are in display points.
    """
    ax.annotate(
        label,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(dx, dy),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )



# ============================================================
# 4) Paper figures (refactored: draw on Axes)
#    UPDATED: higher information density + vivid palette
# ============================================================
def plot_calibration(ax: plt.Axes, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    conf = np.full(n_bins, np.nan, dtype=np.float64)
    acc = np.full(n_bins, np.nan, dtype=np.float64)
    cnt = np.zeros(n_bins, dtype=np.int64)

    for i in range(n_bins):
        mask = (bin_ids == i)
        cnt[i] = int(mask.sum())
        if cnt[i] > 0:
            conf[i] = float(y_prob[mask].mean())
            acc[i] = float(y_true[mask].mean())

    brier = brier_score_loss(y_true, y_prob)
    ece = expected_calibration_error(y_true, y_prob, n_bins=n_bins)

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color=PALETTE["diag"], alpha=0.8)
    ax.plot(conf, acc, marker="o", linewidth=1.6, color=PALETTE["accent"], markersize=4)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical fraction positive")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # marker size encodes counts
    sizes = 20 + 120 * (cnt / max(1, cnt.max()))
    for i in range(n_bins):
        if cnt[i] > 0 and np.isfinite(conf[i]) and np.isfinite(acc[i]):
            ax.scatter(conf[i], acc[i], s=float(sizes[i]), color=PALETTE["accent"], alpha=0.35, linewidths=0)

    # secondary axis: counts bars
    ax2 = ax.twinx()
    centers = 0.5 * (bins[:-1] + bins[1:])
    ax2.bar(centers, cnt, width=(bins[1] - bins[0]) * 0.9, alpha=0.18, color=PALETTE["muted"])
    ax2.set_ylabel("Bin count")
    ax2.grid(False)

    ax.set_title(f"Calibration  Brier={brier:.3f}  ECE={ece:.3f}")


def plot_clip_alignment_distributions(ax: plt.Axes, aux: Dict[str, torch.Tensor], bins: int = 40):
    sim_tcr = cosine_sim(aux["z_tcr_seq"], aux["z_tcr_g"]).numpy()
    sim_pep = cosine_sim(aux["z_pep_seq"], aux["z_pep_g"]).numpy()

    # hist
    ax.hist(sim_tcr, bins=bins, alpha=0.45, label="TCR", density=True, color=PALETTE["tcr"])
    ax.hist(sim_pep, bins=bins, alpha=0.45, label="Peptide", density=True, color=PALETTE["pep"])

    # mean/median lines
    for arr, c in [(sim_tcr, PALETTE["tcr"]), (sim_pep, PALETTE["pep"])]:
        arr = arr[np.isfinite(arr)]
        if arr.size:
            ax.axvline(arr.mean(), color=c, linewidth=1.4, alpha=0.95)
            ax.axvline(np.median(arr), color=c, linewidth=1.2, alpha=0.7, linestyle="--")

    # labels
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Density")
    ax.set_title("Cross-view alignment (seq vs graph)")

    # legend: keep it clean, move away from title/stats
    ax.legend(frameon=False, loc="lower left")

    # stats text box (top-right), avoids overlaps
    tcr = sim_tcr[np.isfinite(sim_tcr)]
    pep = sim_pep[np.isfinite(sim_pep)]
    tcr_line = f"TCR: n={tcr.size}, mean={tcr.mean():.3f}, med={np.median(tcr):.3f}, std={tcr.std(ddof=1):.3f}" if tcr.size else "TCR: n=0"
    pep_line = f"Peptide: n={pep.size}, mean={pep.mean():.3f}, med={np.median(pep):.3f}, std={pep.std(ddof=1):.3f}" if pep.size else "Peptide: n=0"
    textstr = tcr_line + "\n" + pep_line

    ax.text(
        0.98, 0.98, textstr,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="none")
    )



def plot_interaction_score_separation(ax: plt.Axes, aux: Dict[str, torch.Tensor], y_true: np.ndarray, bins: int = 40):
    sim_tp = cosine_sim(aux["tcr"], aux["pep"]).numpy()
    pos = sim_tp[y_true == 1]
    neg = sim_tp[y_true == 0]

    ax.hist(neg, bins=bins, alpha=0.45, label=f"Neg (n={len(neg)})", density=True, color=PALETTE["neg"])
    ax.hist(pos, bins=bins, alpha=0.45, label=f"Pos (n={len(pos)})", density=True, color=PALETTE["pos"])

    if len(pos) > 0:
        ax.axvline(np.mean(pos), color=PALETTE["pos"], linewidth=1.4, alpha=0.95)
    if len(neg) > 0:
        ax.axvline(np.mean(neg), color=PALETTE["neg"], linewidth=1.4, alpha=0.95)

    d = cohens_d(pos, neg)

    sim_auroc = float("nan")
    if len(np.unique(y_true)) > 1:
        try:
            sim_auroc = float(roc_auc_score(y_true, sim_tp))
        except Exception:
            sim_auroc = float("nan")

    ax.set_xlabel(r"Cosine similarity $\cos(\mathrm{tcr}_{\mathrm{fused}},\, \mathrm{pep}_{\mathrm{fused}})$")
    ax.set_ylabel("Density")
    ax.set_title(f"Interaction geometry  d={d:.2f}  AUROC(sim)={sim_auroc:.3f}")
    ax.legend(frameon=False, fontsize=8, loc="upper left")


def fig_calibration(y_true: np.ndarray, y_prob: np.ndarray, out_pdf: str, n_bins: int = 10):
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    plot_calibration(ax, y_true, y_prob, n_bins=n_bins)
    fig.tight_layout()
    save_pdf(fig, out_pdf)


def fig_clip_alignment_distributions(aux: Dict[str, torch.Tensor], out_pdf: str):
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    plot_clip_alignment_distributions(ax, aux)
    fig.tight_layout()
    save_pdf(fig, out_pdf)


def fig_interaction_score_separation(aux: Dict[str, torch.Tensor], y_true: np.ndarray, out_pdf: str):
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    plot_interaction_score_separation(ax, aux, y_true)
    fig.tight_layout()
    save_pdf(fig, out_pdf)


def fig_roc(y_true: np.ndarray, y_prob: np.ndarray, out_pdf: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUROC = {auroc:.3f}", color=PALETTE["accent"], linewidth=1.8)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color=PALETTE["diag"], alpha=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    save_pdf(fig, out_pdf)


def fig_pr(y_true: np.ndarray, y_prob: np.ndarray, out_pdf: str):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    aupr = auc(recall, precision)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.plot(recall, precision, label=f"AUPR = {aupr:.3f}", color=PALETTE["accent"], linewidth=1.8)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    save_pdf(fig, out_pdf)


def fig_prob_hist(y_true: np.ndarray, y_prob: np.ndarray, out_pdf: str, bins: int = 30):
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.hist(neg, bins=bins, alpha=0.45, label=f"Neg (n={len(neg)})", density=True, color=PALETTE["neg"])
    ax.hist(pos, bins=bins, alpha=0.45, label=f"Pos (n={len(pos)})", density=True, color=PALETTE["pos"])
    ax.set_xlabel("Predicted P(binding)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_pdf(fig, out_pdf)


def fig_figure3_combined(y_true: np.ndarray,
                         y_prob: np.ndarray,
                         aux: Dict[str, torch.Tensor],
                         out_pdf: str,
                         figsize: Tuple[float, float] = (13.5, 3.8)):
    set_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    plot_calibration(axes[0], y_true, y_prob, n_bins=10)
    add_panel_label(axes[0], "(a)")

    plot_clip_alignment_distributions(axes[1], aux, bins=45)
    add_panel_label(axes[1], "(b)")

    plot_interaction_score_separation(axes[2], aux, y_true, bins=45)
    add_panel_label(axes[2], "(c)")

    fig.tight_layout()
    save_pdf(fig, out_pdf)


def build_interaction_features(aux: Dict[str, torch.Tensor]) -> np.ndarray:
    t = aux["tcr"].detach().cpu().numpy()
    p = aux["pep"].detach().cpu().numpy()
    feat = np.concatenate([t, p, np.abs(t - p), t * p], axis=1)
    return feat


def pca_2d(feat: np.ndarray, seed: int = 12) -> np.ndarray:
    pca = PCA(n_components=2, random_state=seed)
    xy = pca.fit_transform(feat)
    return xy


def plot_pseudo_spatial_scatter(ax: plt.Axes, xy: np.ndarray, values: np.ndarray, title: str):
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=values, s=10, alpha=0.85, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    return sc


def plot_pseudo_spatial_hexbin(ax: plt.Axes, xy: np.ndarray, values: np.ndarray, title: str, gridsize: int = 45):
    hb = ax.hexbin(
        xy[:, 0], xy[:, 1],
        C=values,
        reduce_C_function=np.mean,
        gridsize=gridsize,
        mincnt=1
    )
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    return hb


def plot_marker_peptide_map(ax: plt.Axes,
                            xy: np.ndarray,
                            y_prob: np.ndarray,
                            pep_list: List[str],
                            marker_pep: str,
                            title_prefix: str = "Peptide"):
    pep_arr = np.asarray(pep_list)
    mask = (pep_arr == marker_pep)

    ax.scatter(xy[:, 0], xy[:, 1], s=6, alpha=0.10, linewidths=0, color=PALETTE["muted"])

    if mask.sum() > 0:
        ax.scatter(xy[mask, 0], xy[mask, 1], c=y_prob[mask], s=14, alpha=0.95, linewidths=0)

    ax.set_title(f"{title_prefix}: {marker_pep}\n(n={int(mask.sum())})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")


def fig_pseudo_spatial_atlas(y_true: np.ndarray,
                             y_prob: np.ndarray,
                             aux: Dict[str, torch.Tensor],
                             pep_list: List[str],
                             out_pdf: str,
                             seed: int = 12,
                             top_k_peptides: int = 4,
                             gridsize: int = 45,
                             figsize: Tuple[float, float] = (13.5, 6.5)):
    feat = build_interaction_features(aux)
    xy = pca_2d(feat, seed=seed)

    cnt = Counter(pep_list)
    top_peps = [p for p, _ in cnt.most_common(top_k_peptides)]
    while len(top_peps) < 3:
        top_peps.append(top_peps[-1] if top_peps else "NA")

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    sc1 = plot_pseudo_spatial_scatter(axes[0, 0], xy, y_prob, "Pseudo-spatial map: predicted P(binding)")
    add_panel_label(axes[0, 0], "(a)")
    fig.colorbar(sc1, ax=axes[0, 0], fraction=0.046, pad=0.04)

    hb = plot_pseudo_spatial_hexbin(axes[0, 1], xy, y_prob, "Neighborhood enrichment: mean P(binding)", gridsize=gridsize)
    add_panel_label(axes[0, 1], "(b)")
    fig.colorbar(hb, ax=axes[0, 1], fraction=0.046, pad=0.04)

    sc2 = plot_pseudo_spatial_scatter(axes[0, 2], xy, y_true.astype(float), "Pseudo-spatial map: ground-truth label")
    add_panel_label(axes[0, 2], "(c)")
    fig.colorbar(sc2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    for j in range(3):
        pep_name = top_peps[j] if j < len(top_peps) else top_peps[-1]
        plot_marker_peptide_map(axes[1, j], xy, y_prob, pep_list, pep_name, title_prefix="Marker peptide")
        add_panel_label(axes[1, j], f"({chr(ord('d') + j)})")

    fig.tight_layout()
    save_pdf(fig, out_pdf)


# ============================================================
# 5) Node saliency (single sample) + perturbation utilities
# ============================================================
def score_prob_binding(model: nn.Module, batch: Dict[str, torch.Tensor]) -> float:
    out = model(batch)
    logits = out["logits"]
    prob = torch.softmax(logits, dim=1)[0, 1]
    return float(prob.detach().cpu().item())


def compute_node_saliency(model: nn.Module,
                          sample: Dict[str, Any],
                          device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    batch = collate_fn([sample])
    for k in ["tcr_seq", "pep_seq", "tcr_x", "tcr_edge", "tcr_batch", "pep_x", "pep_edge", "pep_batch", "y"]:
        batch[k] = batch[k].to(device)

    batch["tcr_x"].requires_grad_(True)
    batch["pep_x"].requires_grad_(True)

    out = model(batch)
    logits = out["logits"]
    base_prob = float(torch.softmax(logits, dim=1)[0, 1].detach().cpu().item())

    score = logits[0, 1]
    model.zero_grad(set_to_none=True)
    score.backward()

    tcr_grad = batch["tcr_x"].grad.detach()
    pep_grad = batch["pep_x"].grad.detach()
    tcr_imp = torch.norm(tcr_grad, p=2, dim=-1).cpu().numpy()
    pep_imp = torch.norm(pep_grad, p=2, dim=-1).cpu().numpy()

    return {"tcr_imp": tcr_imp, "pep_imp": pep_imp, "base_prob": base_prob}


def _topk_indices(x: np.ndarray, k: int) -> List[int]:
    k = max(1, min(k, len(x)))
    return np.argsort(-x)[:k].tolist()


def _bottomk_indices(x: np.ndarray, k: int) -> List[int]:
    k = max(1, min(k, len(x)))
    return np.argsort(x)[:k].tolist()


def _randomk_indices(n: int, k: int, rng: np.random.RandomState) -> List[int]:
    k = max(1, min(k, n))
    return rng.choice(np.arange(n), size=k, replace=False).tolist()


def build_single_batch(sample: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    batch = collate_fn([sample])
    for k in ["tcr_seq", "pep_seq", "tcr_x", "tcr_edge", "tcr_batch", "pep_x", "pep_edge", "pep_batch", "y"]:
        batch[k] = batch[k].to(device)
    return batch


def mask_nodes_inplace(batch: Dict[str, torch.Tensor],
                       tcr_ids: List[int],
                       pep_ids: List[int],
                       mode: str = "zero",
                       rng: Optional[np.random.RandomState] = None):
    if mode not in ["zero", "noise", "shuffle"]:
        raise ValueError(f"Unknown mode: {mode}")

    def _apply(x: torch.Tensor, ids: List[int]):
        if len(ids) == 0:
            return
        if mode == "zero":
            x[ids] = 0.0
        elif mode == "noise":
            std = torch.std(x, dim=0, keepdim=True)
            std = torch.clamp(std, min=1e-6)
            noise = torch.randn((len(ids), x.size(1)), device=x.device, dtype=x.dtype) * std
            x[ids] = noise
        else:
            if len(ids) == 1:
                return
            perm = ids.copy()
            if rng is None:
                random.shuffle(perm)
            else:
                perm = rng.permutation(perm).tolist()
            x[ids] = x[perm]

    _apply(batch["tcr_x"], tcr_ids)
    _apply(batch["pep_x"], pep_ids)


def shuffle_node_features_inplace(batch: Dict[str, torch.Tensor], rng: np.random.RandomState):
    def _shuffle(x: torch.Tensor):
        n = x.size(0)
        perm = torch.from_numpy(rng.permutation(n)).to(x.device)
        x[:] = x[perm]
    _shuffle(batch["tcr_x"])
    _shuffle(batch["pep_x"])


def plot_node_saliency_curve(values: np.ndarray, out_pdf: str, xlabel: str):
    v = values.astype(np.float64)
    if v.max() - v.min() > 1e-12:
        v = (v - v.min()) / (v.max() - v.min())
    else:
        v = v * 0.0

    fig = plt.figure(figsize=(max(6, 0.15 * len(v)), 2.6))
    plt.plot(np.arange(1, len(v) + 1), v, marker="o", linewidth=1, color=PALETTE["accent"])
    plt.xlabel(xlabel)
    plt.ylabel("Saliency (normalized)")
    plt.tight_layout()
    save_pdf(fig, out_pdf)


# ============================================================
# 6) Sanity checks (Deletion / Randomization / Counterfactual)
# ============================================================
def run_sanity_checks(model: nn.Module,
                      ds: MultiModalPairDataset,
                      y_true: np.ndarray,
                      y_prob: np.ndarray,
                      device: torch.device,
                      outdir: str,
                      num_examples: int = 3,
                      topk_ratio: float = 0.2,
                      seed: int = 12,
                      save_individual: bool = True):
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.RandomState(seed)

    idxs = np.where((y_true == 1) & (y_prob >= 0.5))[0].tolist()
    idxs = sorted(idxs, key=lambda i: float(y_prob[i]), reverse=True)
    if len(idxs) < num_examples:
        fallback = list(np.argsort(-y_prob))
        seen = set(idxs)
        for i in fallback:
            if int(i) not in seen:
                idxs.append(int(i))
                seen.add(int(i))
            if len(idxs) >= num_examples:
                break
    idxs = idxs[:num_examples]

    rows = []
    deltas_expl, deltas_rand, deltas_bottom = [], [], []
    cf_delta_top, cf_delta_bottom = [], []
    rand_jacc_tcr = []
    rand_jacc_pep = []

    for rank, i in enumerate(idxs, start=1):
        sample = ds[i]
        info = compute_node_saliency(model, sample, device)
        tcr_imp = info["tcr_imp"]
        pep_imp = info["pep_imp"]

        k_tcr = max(1, int(np.ceil(topk_ratio * len(tcr_imp))))
        k_pep = max(1, int(np.ceil(topk_ratio * len(pep_imp))))

        tcr_top = _topk_indices(tcr_imp, k_tcr)
        pep_top = _topk_indices(pep_imp, k_pep)
        tcr_bottom = _bottomk_indices(tcr_imp, k_tcr)
        pep_bottom = _bottomk_indices(pep_imp, k_pep)
        tcr_rand = _randomk_indices(len(tcr_imp), k_tcr, rng)
        pep_rand = _randomk_indices(len(pep_imp), k_pep, rng)

        # Deletion test (mask to zero)
        b0 = build_single_batch(sample, device)
        prob0 = score_prob_binding(model, b0)

        b_expl = build_single_batch(sample, device)
        mask_nodes_inplace(b_expl, tcr_top, pep_top, mode="zero", rng=rng)
        prob_expl = score_prob_binding(model, b_expl)

        b_rand = build_single_batch(sample, device)
        mask_nodes_inplace(b_rand, tcr_rand, pep_rand, mode="zero", rng=rng)
        prob_rand = score_prob_binding(model, b_rand)

        b_bot = build_single_batch(sample, device)
        mask_nodes_inplace(b_bot, tcr_bottom, pep_bottom, mode="zero", rng=rng)
        prob_bottom = score_prob_binding(model, b_bot)

        delta_expl = prob0 - prob_expl
        delta_rand = prob0 - prob_rand
        delta_bottom = prob0 - prob_bottom

        deltas_expl.append(delta_expl)
        deltas_rand.append(delta_rand)
        deltas_bottom.append(delta_bottom)

        # Counterfactual (noise perturb)
        b_top = build_single_batch(sample, device)
        mask_nodes_inplace(b_top, tcr_top, pep_top, mode="noise", rng=rng)
        prob_cf_top = score_prob_binding(model, b_top)

        b_bottom = build_single_batch(sample, device)
        mask_nodes_inplace(b_bottom, tcr_bottom, pep_bottom, mode="noise", rng=rng)
        prob_cf_bottom = score_prob_binding(model, b_bottom)

        cf_delta_top.append(abs(prob0 - prob_cf_top))
        cf_delta_bottom.append(abs(prob0 - prob_cf_bottom))

        # Randomization test (feature shuffle)
        b_shuf = build_single_batch(sample, device)
        shuffle_node_features_inplace(b_shuf, rng=rng)

        shuf_sample = {
            "tcrb": sample["tcrb"],
            "pep": sample["pep"],
            "tcr_seq": sample["tcr_seq"],
            "pep_seq": sample["pep_seq"],
            "tcr_x": b_shuf["tcr_x"].detach().cpu(),
            "tcr_edge": sample["tcr_edge"],
            "pep_x": b_shuf["pep_x"].detach().cpu(),
            "pep_edge": sample["pep_edge"],
            "y": sample["y"],
        }
        shuf_info = compute_node_saliency(model, shuf_sample, device)
        tcr_imp_shuf = shuf_info["tcr_imp"]
        pep_imp_shuf = shuf_info["pep_imp"]

        tcr_top_shuf = set(_topk_indices(tcr_imp_shuf, k_tcr))
        pep_top_shuf = set(_topk_indices(pep_imp_shuf, k_pep))

        tcr_top_set = set(tcr_top)
        pep_top_set = set(pep_top)

        def jacc(a: set, b: set) -> float:
            if len(a) == 0 and len(b) == 0:
                return 1.0
            return float(len(a & b) / max(1, len(a | b)))

        j_tcr = jacc(tcr_top_set, tcr_top_shuf)
        j_pep = jacc(pep_top_set, pep_top_shuf)
        rand_jacc_tcr.append(j_tcr)
        rand_jacc_pep.append(j_pep)

        rows.append({
            "rank": rank,
            "idx": i,
            "tcrb": sample["tcrb"],
            "pep": sample["pep"],
            "y_true": int(sample["y"].item()),
            "prob0": prob0,
            "k_tcr": k_tcr,
            "k_pep": k_pep,

            "prob_expl_del": prob_expl,
            "prob_rand_del": prob_rand,
            "prob_bottom_del": prob_bottom,
            "delta_expl_del": delta_expl,
            "delta_rand_del": delta_rand,
            "delta_bottom_del": delta_bottom,

            "prob_cf_top": prob_cf_top,
            "prob_cf_bottom": prob_cf_bottom,
            "abs_delta_cf_top": abs(prob0 - prob_cf_top),
            "abs_delta_cf_bottom": abs(prob0 - prob_cf_bottom),

            "jaccard_topk_tcr_after_shuffle": j_tcr,
            "jaccard_topk_pep_after_shuffle": j_pep,
        })

        plot_node_saliency_curve(
            tcr_imp,
            os.path.join(outdir, f"fig_saliency_ex{rank}_tcr.pdf"),
            xlabel="TCR node index (approx residue position)"
        )
        plot_node_saliency_curve(
            pep_imp,
            os.path.join(outdir, f"fig_saliency_ex{rank}_pep.pdf"),
            xlabel="Peptide node index (approx residue position)"
        )

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(outdir, "sanity_results.csv"), index=False)

    if save_individual:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        violin_box_jitter(
            ax,
            [deltas_expl, deltas_rand, deltas_bottom],
            ["Explain top-k", "Random-k", "Bottom-k"],
            [PALETTE["accent"], PALETTE["muted"], PALETTE["neg"]],
            ylabel="Score drop Δ = P0 - P(after mask)",
            title="Deletion test"
        )
        fig.tight_layout()
        save_pdf(fig, os.path.join(outdir, "fig_deletion_test.pdf"))

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        violin_box_jitter(
            ax,
            [cf_delta_top, cf_delta_bottom],
            ["Perturb top-k", "Perturb bottom-k"],
            [PALETTE["pos"], PALETTE["neg"]],
            ylabel="|Score change|",
            title="Counterfactual test"
        )
        fig.tight_layout()
        save_pdf(fig, os.path.join(outdir, "fig_counterfactual.pdf"))

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        violin_box_jitter(
            ax,
            [rand_jacc_tcr, rand_jacc_pep],
            ["TCR top-k", "Peptide top-k"],
            [PALETTE["tcr"], PALETTE["pep"]],
            ylabel="Top-k overlap after feature shuffle (Jaccard)",
            title="Randomization test"
        )
        ax.set_ylim(0.0, 1.0)
        fig.tight_layout()
        save_pdf(fig, os.path.join(outdir, "fig_randomization_test.pdf"))

    return {
        "deltas_expl": deltas_expl,
        "deltas_rand": deltas_rand,
        "deltas_bottom": deltas_bottom,
        "cf_delta_top": cf_delta_top,
        "cf_delta_bottom": cf_delta_bottom,
        "rand_jacc_tcr": rand_jacc_tcr,
        "rand_jacc_pep": rand_jacc_pep,
    }


def fig_figure4_combined(sanity_data: Dict[str, List[float]],
                         out_pdf: str,
                         figsize: Tuple[float, float] = (13.5, 3.8)):
    set_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    data_a = [sanity_data["deltas_expl"], sanity_data["deltas_rand"], sanity_data["deltas_bottom"]]
    violin_box_jitter(
        axes[0], data_a,
        ["Explain top-k", "Random-k", "Bottom-k"],
        [PALETTE["accent"], PALETTE["muted"], PALETTE["neg"]],
        ylabel="Score drop Δ = P0 - P(after mask)",
        title="Deletion test"
    )
    add_panel_label(axes[0], "(a)")
    axes[0].text(0.02, 0.02, f"n={len(data_a[0])}", transform=axes[0].transAxes, ha="left", va="bottom")

    data_b = [sanity_data["cf_delta_top"], sanity_data["cf_delta_bottom"]]
    violin_box_jitter(
        axes[1], data_b,
        ["Perturb top-k", "Perturb bottom-k"],
        [PALETTE["pos"], PALETTE["neg"]],
        ylabel="|Score change|",
        title="Counterfactual test"
    )
    add_panel_label(axes[1], "(b)")
    axes[1].text(0.02, 0.02, f"n={len(data_b[0])}", transform=axes[1].transAxes, ha="left", va="bottom")

    data_c = [sanity_data["rand_jacc_tcr"], sanity_data["rand_jacc_pep"]]
    violin_box_jitter(
        axes[2], data_c,
        ["TCR top-k", "Peptide top-k"],
        [PALETTE["tcr"], PALETTE["pep"]],
        ylabel="Top-k overlap after feature shuffle (Jaccard)",
        title="Randomization test"
    )
    axes[2].set_ylim(0.0, 1.0)
    add_panel_label(axes[2], "(c)")
    axes[2].text(0.02, 0.02, f"n={len(data_c[0])}", transform=axes[2].transAxes, ha="left", va="bottom")

    fig.tight_layout()
    save_pdf(fig, out_pdf)


# ============================================================
# 7) Main
# ============================================================
def main():
    ROOT_PATH = "./"

    parser = argparse.ArgumentParser("Load trained TCR-Multi model and generate paper-ready PDF figures.")
    parser.add_argument("--root", type=str, default=str(Path.cwd().resolve()), help="Project root (same as training ROOT_PATH).")

    # data
    parser.add_argument("--train_base", type=str, required=False, default="dataset/ds.hard-splits/pep+cdr3b", help="Same as args.train_base in training.")
    parser.add_argument("--embedbase", type=str, required=False, default=os.path.join(ROOT_PATH, "embs"), help="Directory containing *_dict.pkl files.")
    parser.add_argument("--mode", type=str, default="only-sampled-negs", help="Split mode used in training.")
    parser.add_argument("--dataset_index", type=int, default=1, help="Which dataset index to evaluate.")
    parser.add_argument("--tcr_col", type=str, default="cdr3.beta")
    parser.add_argument("--pep_col", type=str, default="antigen.epitope")
    parser.add_argument("--batch_size", type=int, default=128)

    # checkpoint + model hparams (must match training)
    parser.add_argument("--ckpt", type=str, required=False, default="multimodal_clip_binding_results/multimodal_only-sampled-negs/best_multimodal_0.pth", help="Path to checkpoint file.")
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--graph_hidden", type=int, default=256)
    parser.add_argument("--graph_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)

    # output
    parser.add_argument("--outdir", type=str, default="./paper_figures", help="Output directory for PDFs.")
    parser.add_argument("--seed", type=int, default=12)

    # sanity checks
    parser.add_argument("--num_examples", type=int, default=3, help="Sanity check sample count (default 3).")
    parser.add_argument("--topk_ratio", type=float, default=0.2, help="Top-k ratio per graph for explanation.")

    # combined figures switches/sizes
    parser.add_argument("--make_combined", action="store_true", help="If set, generate Figure3/Figure4 combined PDFs.")
    parser.add_argument("--combined_width", type=float, default=13.5, help="Combined figure width (inches).")
    parser.add_argument("--combined_height", type=float, default=3.8, help="Combined figure height (inches).")

    args = parser.parse_args()

    set_seed(args.seed)
    set_paper_style()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    # -----------------------
    # Load dict pickles
    # -----------------------
    embedbase = args.embedbase
    with open(os.path.join(embedbase, "peptide_seq_dict.pkl"), "rb") as f:
        peptide_seq_dict = pickle.load(f)
    with open(os.path.join(embedbase, "tcr_seq_dict.pkl"), "rb") as f:
        tcrb_seq_dict = pickle.load(f)

    graph_tcr_path = os.path.join(embedbase, "tcr_graph_dict.pkl")
    graph_pep_path = os.path.join(embedbase, "peptide_graph_dict.pkl")
    with open(graph_tcr_path, "rb") as f:
        tcrb_graph_dict = pickle.load(f)
    with open(graph_pep_path, "rb") as f:
        peptide_graph_dict = pickle.load(f)

    # -----------------------
    # Load test CSV
    # -----------------------
    test_file = f"{args.train_base}/test/{args.mode}/test-{args.dataset_index}.csv"
    df = pd.read_csv(test_file, low_memory=False).drop_duplicates()

    values_to_remove = ["CASSQETDIVFNXPQHF", "CASSLRTRTDTQYX", "CASSILGWSEAFX", "CSARTGDRTEAFX", "CASSQETDIVFNOPQHF"]
    df = df[~df[args.tcr_col].isin(values_to_remove)].reset_index(drop=True)

    ds = MultiModalPairDataset(
        df=df,
        tcr_col=args.tcr_col,
        pep_col=args.pep_col,
        tcrb_seq_dict=tcrb_seq_dict,
        peptide_seq_dict=peptide_seq_dict,
        tcrb_graph_dict=tcrb_graph_dict,
        peptide_graph_dict=peptide_graph_dict
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # infer dims from one sample
    s0 = ds[0]
    tcr_seq_dim = s0["tcr_seq"].numel()
    pep_seq_dim = s0["pep_seq"].numel()
    tcr_node_dim = s0["tcr_x"].size(-1)
    pep_node_dim = s0["pep_x"].size(-1)

    # -----------------------
    # Build model + load ckpt
    # -----------------------
    model = MultiModalBindingModel(
        tcr_seq_dim=tcr_seq_dim,
        pep_seq_dim=pep_seq_dim,
        tcr_node_dim=tcr_node_dim,
        pep_node_dim=pep_node_dim,
        proj_dim=args.proj_dim,
        graph_hidden=args.graph_hidden,
        graph_layers=args.graph_layers,
        dropout=args.dropout
    ).to(device)

    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # -----------------------
    # Predict and metrics
    # -----------------------
    y_true, y_prob, aux = predict_all(model, loader, device)
    preds = (y_prob > 0.5).astype(int)

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    aupr = float(auc(rec, prec))

    metrics = {
        "AUROC": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "AUPR": aupr,
        "Accuracy": float(accuracy_score(y_true, preds)),
        "Precision": float(precision_score(y_true, preds, zero_division=0)),
        "Recall": float(recall_score(y_true, preds, zero_division=0)),
        "F1": float(f1_score(y_true, preds, zero_division=0)),
        "Brier": float(brier_score_loss(y_true, y_prob)),
        "N": int(len(y_true)),
        "Pos": int((y_true == 1).sum()),
        "Neg": int((y_true == 0).sum()),
        "test_file": test_file,
        "ckpt": args.ckpt
    }

    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame({"y_true": y_true, "y_prob": y_prob}).to_csv(
        os.path.join(args.outdir, "predictions.csv"), index=False
    )

    # -----------------------
    # Individual Figures (still saved)
    # -----------------------
    fig_roc(y_true, y_prob, os.path.join(args.outdir, "fig_roc.pdf"))
    fig_pr(y_true, y_prob, os.path.join(args.outdir, "fig_pr.pdf"))
    fig_prob_hist(y_true, y_prob, os.path.join(args.outdir, "fig_prob_hist.pdf"))

    fig_calibration(y_true, y_prob, os.path.join(args.outdir, "fig_calibration.pdf"))
    fig_clip_alignment_distributions(aux, os.path.join(args.outdir, "fig_clip_alignment.pdf"))
    fig_interaction_score_separation(aux, y_true, os.path.join(args.outdir, "fig_cosine_tcr_pep.pdf"))

    # -----------------------
    # Sanity checks (+ optional individual PDFs)
    # -----------------------
    sanity_data = run_sanity_checks(
        model=model,
        ds=ds,
        y_true=y_true,
        y_prob=y_prob,
        device=device,
        outdir=args.outdir,
        num_examples=args.num_examples,
        topk_ratio=args.topk_ratio,
        seed=args.seed,
        save_individual=True
    )

    # -----------------------
    # Combined Figures for paper
    # -----------------------
    if args.make_combined:
        figsize = (args.combined_width, args.combined_height)

        fig_figure3_combined(
            y_true=y_true,
            y_prob=y_prob,
            aux=aux,
            out_pdf=os.path.join(args.outdir, "fig_figure3_abc.pdf"),
            figsize=figsize
        )

        fig_figure4_combined(
            sanity_data=sanity_data,
            out_pdf=os.path.join(args.outdir, "fig_figure4_abc.pdf"),
            figsize=figsize
        )

    print("Done. Outputs written to:", args.outdir)
    print("Metrics:", json.dumps(metrics, indent=2))
    if args.make_combined:
        print("Combined figures:",
              os.path.join(args.outdir, "fig_figure3_abc.pdf"),
              os.path.join(args.outdir, "fig_figure4_abc.pdf"))

    # -----------------------
    # Pseudo-spatial atlas (embedding-space visualization)
    # -----------------------
    fig_pseudo_spatial_atlas(
        y_true=y_true,
        y_prob=y_prob,
        aux=aux,
        pep_list=df[args.pep_col].tolist(),
        out_pdf=os.path.join(args.outdir, "fig_pseudo_spatial_atlas.pdf"),
        seed=args.seed,
        top_k_peptides=3,
        gridsize=45,
        figsize=(args.combined_width, 6.5)
    )


if __name__ == "__main__":
    main()

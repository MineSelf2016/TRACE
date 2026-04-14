#!/usr/bin/env python3
"""Alignment Geometry Analysis

Loads a trained checkpoint, extracts seq/graph/fused embeddings on a chosen split,
computes cosine-similarity distributions, per-batch InfoNCE (CLIP) contributions,
and produces t-SNE/UMAP visualizations. Results are saved under --outdir.

Dependencies: numpy, pandas, torch, sklearn; optional umap-learn for UMAP.
"""

from __future__ import annotations

import os
import json
import pickle
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent directory to path to import analysis.py
sys.path.insert(0, str(Path(__file__).parent.parent))

# Reuse dataset + model from analysis.py to avoid duplication
from analysis import (
    MultiModalPairDataset,
    collate_fn,
    MultiModalBindingModel,
    cosine_sim,
    fig_clip_alignment_distributions,
    fig_interaction_score_separation,
)


def clip_loss(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float) -> torch.Tensor:
    z_a = nn.functional.normalize(z_a, dim=-1)
    z_b = nn.functional.normalize(z_b, dim=-1)
    logits = (z_a @ z_b.t()) / temperature
    labels = torch.arange(z_a.size(0), device=z_a.device)
    loss_ab = nn.functional.cross_entropy(logits, labels)
    loss_ba = nn.functional.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_ab + loss_ba)


@torch.no_grad()
def collect_embeddings(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    ys: List[np.ndarray] = []
    probs: List[np.ndarray] = []

    z_tcr_seq_all, z_tcr_g_all = [], []
    z_pep_seq_all, z_pep_g_all = [], []
    tcr_all, pep_all = [], []

    for batch in loader:
        # move only tensors
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        out = model(batch)
        logits = out["logits"]
        prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

        ys.append(batch["y"].detach().cpu().numpy())
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


def compute_clip_components(aux: Dict[str, torch.Tensor], temperature: float) -> Dict[str, float]:
    # Batch-wise InfoNCE is not defined without batches, but we can compute
    # an approximation by slicing into small chunks.
    def _approx(z1: torch.Tensor, z2: torch.Tensor, temp: float, chunk: int = 256) -> float:
        n = z1.size(0)
        if n == 0:
            return float("nan")
        losses = []
        for i in range(0, n, chunk):
            zs = z1[i:i+chunk]
            zg = z2[i:i+chunk]
            if zs.size(0) < 2:
                continue
            losses.append(float(clip_loss(zs, zg, temp).item()))
        return float(np.mean(losses)) if losses else float("nan")

    return {
        "clip_tcr": _approx(aux["z_tcr_seq"], aux["z_tcr_g"], temperature),
        "clip_pep": _approx(aux["z_pep_seq"], aux["z_pep_g"], temperature),
    }


def try_tsne_umap(fused: np.ndarray, labels: np.ndarray, outdir: str, prefix: str = "fused"):
    from sklearn.manifold import TSNE
    xy_tsne = TSNE(n_components=2, random_state=12, perplexity=30).fit_transform(fused)
    pd.DataFrame({"x": xy_tsne[:, 0], "y": xy_tsne[:, 1], "label": labels}).to_csv(
        os.path.join(outdir, f"{prefix}_tsne.csv"), index=False
    )
    try:
        import umap  # type: ignore
        xy_umap = umap.UMAP(n_components=2, random_state=12).fit_transform(fused)
        pd.DataFrame({"x": xy_umap[:, 0], "y": xy_umap[:, 1], "label": labels}).to_csv(
            os.path.join(outdir, f"{prefix}_umap.csv"), index=False
        )
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser("Alignment Geometry Analysis")
    ap.add_argument("--train_base", default="dataset/ds.hard-splits/pep+cdr3b")
    ap.add_argument("--embedbase", default="embs")
    ap.add_argument("--mode", default="only-sampled-negs")
    ap.add_argument("--dataset_index", type=int, default=1)
    ap.add_argument("--ckpt", required=False, default="multimodal_clip_binding_results/multimodal_only-sampled-negs/best_multimodal_0.pth")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--proj_dim", type=int, default=256)
    ap.add_argument("--graph_hidden", type=int, default=256)
    ap.add_argument("--graph_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--outdir", default="./analysis_runs/align_geometry")

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dict pickles
    with open(os.path.join(args.embedbase, "peptide_seq_dict.pkl"), "rb") as f:
        peptide_seq_dict = pickle.load(f)
    with open(os.path.join(args.embedbase, "tcr_seq_dict.pkl"), "rb") as f:
        tcrb_seq_dict = pickle.load(f)
    with open(os.path.join(args.embedbase, "tcr_graph_dict.pkl"), "rb") as f:
        tcrb_graph_dict = pickle.load(f)
    with open(os.path.join(args.embedbase, "peptide_graph_dict.pkl"), "rb") as f:
        peptide_graph_dict = pickle.load(f)

    # Load test split
    test_file = f"{args.train_base}/test/{args.mode}/test-{args.dataset_index}.csv"
    df = pd.read_csv(test_file, low_memory=False).drop_duplicates()
    values_to_remove = ["CASSQETDIVFNXPQHF", "CASSLRTRTDTQYX", "CASSILGWSEAFX", "CSARTGDRTEAFX", "CASSQETDIVFNOPQHF"]
    df = df[~df["cdr3.beta"].isin(values_to_remove)].reset_index(drop=True)

    ds = MultiModalPairDataset(
        df=df,
        tcr_col="cdr3.beta",
        pep_col="antigen.epitope",
        tcrb_seq_dict=tcrb_seq_dict,
        peptide_seq_dict=peptide_seq_dict,
        tcrb_graph_dict=tcrb_graph_dict,
        peptide_graph_dict=peptide_graph_dict,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Infer dims
    s0 = ds[0]
    model = MultiModalBindingModel(
        tcr_seq_dim=s0["tcr_seq"].numel(),
        pep_seq_dim=s0["pep_seq"].numel(),
        tcr_node_dim=s0["tcr_x"].size(-1),
        pep_node_dim=s0["pep_x"].size(-1),
        proj_dim=args.proj_dim,
        graph_hidden=args.graph_hidden,
        graph_layers=args.graph_layers,
        dropout=args.dropout,
    ).to(device)

    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)

    # Collect
    y_true, y_prob, aux = collect_embeddings(model, loader, device)

    # Distributions and interaction geometry
    fig_clip_alignment_distributions(aux, os.path.join(args.outdir, "alignment_distributions.pdf"))
    fig_interaction_score_separation(aux, y_true, os.path.join(args.outdir, "interaction_geometry.pdf"))

    # Compute CLIP components (approx) and save summary
    comps = compute_clip_components(aux, temperature=args.temperature)
    summary = {
        "test_file": test_file,
        "ckpt": args.ckpt,
        "clip_tcr": comps["clip_tcr"],
        "clip_pep": comps["clip_pep"],
        "mean_cos_seq_graph_tcr": float(torch.mean(cosine_sim(aux["z_tcr_seq"], aux["z_tcr_g"])) .item()),
        "mean_cos_seq_graph_pep": float(torch.mean(cosine_sim(aux["z_pep_seq"], aux["z_pep_g"])) .item()),
    }
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # t-SNE / UMAP of fused embeddings
    fused = torch.cat([aux["tcr"], aux["pep"]], dim=1).detach().cpu().numpy()
    try_tsne_umap(fused, y_true.astype(int), args.outdir, prefix="fused")

    print("✓ Alignment geometry outputs:")
    print("  -", os.path.join(args.outdir, "alignment_distributions.pdf"))
    print("  -", os.path.join(args.outdir, "interaction_geometry.pdf"))
    print("  -", os.path.join(args.outdir, "summary.json"))
    print("  -", os.path.join(args.outdir, "fused_tsne.csv"), "(+ optional fused_umap.csv)")


if __name__ == "__main__":
    main()

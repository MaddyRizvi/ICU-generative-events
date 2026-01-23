from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from src.models.event_transformer import EventTransformer
from src.train.clf_datasets import PatientMortalityDataset


def _load_splits(labels_dir: Path, labels: pd.DataFrame, seed: int = 42):
    """
    Uses make_splits outputs if available; otherwise fallback to random split.
    """
    train_p = labels_dir / "train_patients.parquet"
    val_p = labels_dir / "val_patients.parquet"
    test_p = labels_dir / "test_patients.parquet"

    if train_p.exists() and val_p.exists() and test_p.exists():
        train_ids = pd.read_parquet(train_p)["patient_id"].unique()
        val_ids = pd.read_parquet(val_p)["patient_id"].unique()
        test_ids = pd.read_parquet(test_p)["patient_id"].unique()
        return train_ids, val_ids, test_ids, "hospital_holdout"

    # fallback: random patient split 80/10/10
    pids = labels["patient_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(pids)
    n = len(pids)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_ids = pids[:n_train]
    val_ids = pids[n_train : n_train + n_val]
    test_ids = pids[n_train + n_val :]
    return train_ids, val_ids, test_ids, "random"


def _mean_pool(h: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
    """
    h: (B,T,D), attn: (B,T) bool
    """
    attn_f = attn.float().unsqueeze(-1)  # (B,T,1)
    denom = attn_f.sum(dim=1).clamp(min=1.0)
    return (h * attn_f).sum(dim=1) / denom


@torch.no_grad()
def evaluate(model, head, loader, device):
    model.eval()
    head.eval()
    ys, ps = [], []

    for var_in, val_in, dt_in, attn, y, pid in loader:
        var_in = var_in.to(device)
        val_in = val_in.to(device)
        dt_in = dt_in.to(device)
        attn = attn.to(device)
        y = y.cpu().numpy()

        h = model.encode(var_in, val_in, dt_in, attn_mask=attn)
        pooled = _mean_pool(h, attn)
        logits = head(pooled).squeeze(-1)
        prob = torch.sigmoid(logits).cpu().numpy()

        ys.append(y)
        ps.append(prob)

    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)

    # Guard for edge cases (all one class)
    metrics = {}
    if len(np.unique(y_true)) > 1:
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
        metrics["auprc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["auroc"] = None
        metrics["auprc"] = None

    metrics["n"] = int(len(y_true))
    metrics["prevalence"] = float(np.mean(y_true))
    return metrics


def main():
    p = argparse.ArgumentParser(description="Fine-tune pretrained ICU event transformer for mortality.")
    p.add_argument("--tokens", type=str, required=True)
    p.add_argument("--labels", type=str, required=True)
    p.add_argument("--pretrained", type=str, required=True)
    p.add_argument("--out", type=str, default="outputs/checkpoints/mortality.pt")

    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tokens = pd.read_parquet(args.tokens)
    labels = pd.read_parquet(args.labels)

    # Checkpoint
    ckpt = torch.load(args.pretrained, map_location="cpu")
    cfg = ckpt["config"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = EventTransformer(
        n_var_tokens=cfg["n_var_tokens"],
        n_val_tokens=cfg["n_val_tokens"],
        max_len=cfg["max_len"],
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    if args.freeze_backbone:
        for p_ in model.parameters():
            p_.requires_grad = False

    # Small classifier head
    head = nn.Sequential(
        nn.Linear(cfg["d_model"], cfg["d_model"]),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(cfg["d_model"], 1),
    ).to(device)

    # Splits
    labels_dir = Path(args.labels).parent
    train_ids, val_ids, test_ids, split_type = _load_splits(labels_dir, labels, seed=args.seed)
    print(f"Split type: {split_type} | train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")

    train_ds = PatientMortalityDataset(tokens, labels, train_ids, max_len=args.max_len)
    val_ds = PatientMortalityDataset(tokens, labels, val_ids, max_len=args.max_len)
    test_ds = PatientMortalityDataset(tokens, labels, test_ids, max_len=args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    params = list(head.parameters()) + ([] if args.freeze_backbone else list(model.parameters()))
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train(not args.freeze_backbone)
        head.train()

        total_loss = 0.0
        total_n = 0

        for var_in, val_in, dt_in, attn, y, pid in train_loader:
            var_in = var_in.to(device)
            val_in = val_in.to(device)
            dt_in = dt_in.to(device)
            attn = attn.to(device)
            y = y.to(device)

            h = model.encode(var_in, val_in, dt_in, attn_mask=attn)
            pooled = _mean_pool(h, attn)
            logits = head(pooled).squeeze(-1)

            loss = loss_fn(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            total_loss += float(loss.item()) * y.shape[0]
            total_n += int(y.shape[0])

        train_loss = total_loss / max(1, total_n)
        val_metrics = evaluate(model, head, val_loader, device)
        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | val={val_metrics}")

        history.append({"epoch": epoch, "train_loss": train_loss, "val": val_metrics})

        # Use AUROC if available, else AUPRC
        score = val_metrics["auroc"] if val_metrics["auroc"] is not None else (val_metrics["auprc"] or 0.0)
        if score > best_val:
            best_val = score
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "backbone_state": model.state_dict(),
                    "head_state": head.state_dict(),
                    "config": cfg,
                    "split_type": split_type,
                    "history": history,
                },
                out_path,
            )
            print(f"Saved best checkpoint: {out_path}")

    # Final test
    test_metrics = evaluate(model, head, test_loader, device)
    print("Test metrics:", test_metrics)

    metrics_path = Path(args.out).with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps({"test": test_metrics, "history": history}, indent=2))
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models.event_transformer import EventTransformer
from .datasets import MaskConfig, PatientSequenceDataset


def load_patient_split(labels_path: Path, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    If train/val files exist, use them.
    Else create a simple random patient split (90/10).
    """
    train_path = labels_path.parent / "train_patients.parquet"
    val_path = labels_path.parent / "val_patients.parquet"

    if train_path.exists() and val_path.exists():
        train_ids = pd.read_parquet(train_path)["patient_id"].unique()
        val_ids = pd.read_parquet(val_path)["patient_id"].unique()
        return train_ids, val_ids

    labels = pd.read_parquet(labels_path)
    pids = labels["patient_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(pids)
    n_val = max(1, int(0.1 * len(pids)))
    return pids[n_val:], pids[:n_val]


def main() -> None:
    p = argparse.ArgumentParser(description="Masked-event pretraining on ICU event tokens.")
    p.add_argument("--tokens", type=str, required=True)
    p.add_argument("--labels", type=str, required=True, help="labels.parquet (for splits)")
    p.add_argument("--out", type=str, default="outputs/checkpoints/pretrain.pt")

    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--mask_prob", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tokens_path = Path(args.tokens)
    labels_path = Path(args.labels)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading tokens...")
    tok = pd.read_parquet(tokens_path)

    required = {"patient_id", "time_hours", "dt_hours", "variable_id", "value_bin"}
    if not required.issubset(tok.columns):
        raise ValueError(f"tokens.parquet must contain {required}. Found: {set(tok.columns)}")

    # Define vocab sizes and reserve last id as MASK
    max_var = int(tok["variable_id"].max())
    max_val = int(tok["value_bin"].max())
    n_var_tokens = max_var + 2  # +1 for inclusive, +1 for MASK id at end
    n_val_tokens = max_val + 2

    # Ensure PAD=0 exists (already), actual tokens start at 1, MASK is last id.
    print(f"n_var_tokens={n_var_tokens} (MASK id={n_var_tokens-1})")
    print(f"n_val_tokens={n_val_tokens} (MASK id={n_val_tokens-1})")

    train_ids, val_ids = load_patient_split(labels_path, seed=args.seed)
    print(f"Train patients: {len(train_ids):,} | Val patients: {len(val_ids):,}")

    mask_cfg = MaskConfig(mask_prob=args.mask_prob)

    train_ds = PatientSequenceDataset(
        tok,
        patient_ids=train_ids,
        max_len=args.max_len,
        n_dt_buckets=16,
        n_var_tokens=n_var_tokens,
        n_val_tokens=n_val_tokens,
        mask_cfg=mask_cfg,
        seed=args.seed,
    )
    val_ds = PatientSequenceDataset(
        tok,
        patient_ids=val_ids,
        max_len=args.max_len,
        n_dt_buckets=16,
        n_var_tokens=n_var_tokens,
        n_val_tokens=n_val_tokens,
        mask_cfg=mask_cfg,
        seed=args.seed + 1,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = EventTransformer(
        n_var_tokens=n_var_tokens,
        n_val_tokens=n_val_tokens,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        max_len=args.max_len,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce_var = nn.CrossEntropyLoss(reduction="none")
    ce_val = nn.CrossEntropyLoss(reduction="none")

    def run_epoch(loader, train: bool):
        model.train(train)
        total_loss = 0.0
        total_count = 0

        for batch in loader:
            var_in, val_in, dt_in, attn_mask, var_tgt, val_tgt, mask_pos = batch
            var_in = var_in.to(device)
            val_in = val_in.to(device)
            dt_in = dt_in.to(device)
            attn_mask = attn_mask.to(device)
            var_tgt = var_tgt.to(device)
            val_tgt = val_tgt.to(device)
            mask_pos = mask_pos.to(device)

            var_logits, val_logits = model(var_in, val_in, dt_in, attn_mask=attn_mask)

            # Compute loss only on masked positions
            B, T, _ = var_logits.shape
            mask_flat = mask_pos.view(-1)
            if mask_flat.sum().item() == 0:
                continue

            var_loss_all = ce_var(var_logits.view(-1, n_var_tokens), var_tgt.view(-1))
            val_loss_all = ce_val(val_logits.view(-1, n_val_tokens), val_tgt.view(-1))

            loss = (var_loss_all[mask_flat].mean() + val_loss_all[mask_flat].mean())

            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            total_loss += float(loss.item()) * int(mask_flat.sum().item())
            total_count += int(mask_flat.sum().item())

        return total_loss / max(1, total_count)

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(train_loader, train=True)
        val_loss = run_epoch(val_loader, train=False)

        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            ckpt = {
                "model_state": model.state_dict(),
                "config": {
                    "n_var_tokens": n_var_tokens,
                    "n_val_tokens": n_val_tokens,
                    "max_len": args.max_len,
                    "d_model": args.d_model,
                    "n_layers": args.n_layers,
                    "n_heads": args.n_heads,
                    "dropout": args.dropout,
                },
                "history": history,
            }
            torch.save(ckpt, out_path)
            print(f"Saved checkpoint: {out_path}")

    # Save training history as JSON for easy plotting
    metrics_path = out_path.with_suffix(".history.json")
    metrics_path.write_text(json.dumps(history, indent=2))
    print(f"Saved history: {metrics_path}")


if __name__ == "__main__":
    main()

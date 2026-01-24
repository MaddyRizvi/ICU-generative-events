from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def bucketize_dt(dt_hours: np.ndarray, n_buckets: int = 16) -> np.ndarray:
    """Bucket time gaps into log-spaced bins."""
    dt = np.clip(dt_hours, 0.0, None)
    # Log-spaced buckets from 0.36 seconds to 48 hours
    edges = np.geomspace(1e-4, 48.0, num=n_buckets)
    idx = np.digitize(dt, edges, right=False) - 1
    return np.clip(idx, 0, n_buckets - 1).astype(np.int64)


@dataclass
class MaskConfig:
    mask_prob: float = 0.15
    replace_with_mask_prob: float = 0.8
    replace_with_random_prob: float = 0.1


class PatientSequenceDataset(Dataset):
    """Builds fixed-length chunks from patient event sequences with BERT-style masking."""

    def __init__(
        self,
        tokens: pd.DataFrame,
        patient_ids: np.ndarray,
        max_len: int = 256,
        n_dt_buckets: int = 16,
        n_var_tokens: int = 0,
        n_val_tokens: int = 0,
        mask_cfg: MaskConfig = MaskConfig(),
        seed: int = 42,
    ):
        self.max_len = max_len
        self.n_dt_buckets = n_dt_buckets
        self.n_var_tokens = n_var_tokens
        self.n_val_tokens = n_val_tokens
        self.mask_cfg = mask_cfg
        self.rng = np.random.default_rng(seed)

        # Filter tokens to selected patients and sort
        df = tokens[tokens["patient_id"].isin(patient_ids)].copy()
        df = df.sort_values(["patient_id", "time_hours"]).reset_index(drop=True)

        # Group events by patient
        self.patient_groups: Dict[int, pd.DataFrame] = {
            int(pid): g for pid, g in df.groupby("patient_id", sort=False)
        }

        # Create sliding windows for training
        self.windows: List[Tuple[int, int]] = []
        for pid, g in self.patient_groups.items():
            n = len(g)
            if n == 0:
                continue
            # Non-overlapping windows
            for start in range(0, n, max_len):
                self.windows.append((pid, start))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        pid, start = self.windows[idx]
        g = self.patient_groups[pid]

        chunk = g.iloc[start : start + self.max_len]
        T = len(chunk)

        var = chunk["variable_id"].to_numpy(np.int64)
        val = chunk["value_bin"].to_numpy(np.int64)
        dt = chunk["dt_hours"].to_numpy(np.float32)

        dt_bucket = bucketize_dt(dt, n_buckets=self.n_dt_buckets)

        # Pad sequences to fixed length
        var_in = np.zeros((self.max_len,), dtype=np.int64)
        val_in = np.zeros((self.max_len,), dtype=np.int64)
        dt_in = np.zeros((self.max_len,), dtype=np.int64)
        attn_mask = np.zeros((self.max_len,), dtype=np.bool_)

        var_in[:T] = var
        val_in[:T] = val
        dt_in[:T] = dt_bucket
        attn_mask[:T] = True

        # Save original values as targets
        var_tgt = var_in.copy()
        val_tgt = val_in.copy()

        # Randomly pick positions to mask
        mask_positions = np.zeros((self.max_len,), dtype=np.bool_)
        real_positions = np.where(attn_mask)[0]
        if len(real_positions) > 0:
            n_mask = max(1, int(len(real_positions) * self.mask_cfg.mask_prob))
            chosen = self.rng.choice(real_positions, size=n_mask, replace=False)
            mask_positions[chosen] = True

            # Apply BERT-style masking (80% mask token, 10% random, 10% keep)
            var_mask_id = self.n_var_tokens - 1
            val_mask_id = self.n_val_tokens - 1

            for pos in chosen:
                p = self.rng.random()
                if p < self.mask_cfg.replace_with_mask_prob:
                    var_in[pos] = var_mask_id
                    val_in[pos] = val_mask_id
                elif p < self.mask_cfg.replace_with_mask_prob + self.mask_cfg.replace_with_random_prob:
                    # Replace with random token
                    var_in[pos] = int(self.rng.integers(1, self.n_var_tokens - 1))
                    val_in[pos] = int(self.rng.integers(0, self.n_val_tokens - 1))
                else:
                    # Keep original value
                    pass

        return (
            torch.from_numpy(var_in),
            torch.from_numpy(val_in),
            torch.from_numpy(dt_in),
            torch.from_numpy(attn_mask),
            torch.from_numpy(var_tgt),
            torch.from_numpy(val_tgt),
            torch.from_numpy(mask_positions),
        )

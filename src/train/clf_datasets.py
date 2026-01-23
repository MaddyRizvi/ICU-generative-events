from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.train.datasets import bucketize_dt


class PatientMortalityDataset(Dataset):
    """
    One sequence window per patient (first max_len events).
    Returns:
      var_ids, val_ids, dt_bucket, attn_mask, label
    """

    def __init__(
        self,
        tokens: pd.DataFrame,
        labels: pd.DataFrame,
        patient_ids: np.ndarray,
        max_len: int = 256,
        n_dt_buckets: int = 16,
    ):
        self.max_len = max_len
        self.n_dt_buckets = n_dt_buckets

        lab = labels[labels["patient_id"].isin(patient_ids)][["patient_id", "mortality"]].copy()
        self.y_map: Dict[int, int] = {int(r.patient_id): int(r.mortality) for r in lab.itertuples(index=False)}

        df = tokens[tokens["patient_id"].isin(patient_ids)].copy()
        df = df.sort_values(["patient_id", "time_hours"]).reset_index(drop=True)

        self.patient_groups: Dict[int, pd.DataFrame] = {int(pid): g for pid, g in df.groupby("patient_id", sort=False)}
        self.patients: List[int] = [pid for pid in self.patient_groups.keys() if pid in self.y_map]

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int):
        pid = self.patients[idx]
        g = self.patient_groups[pid].iloc[: self.max_len]

        T = len(g)
        var = g["variable_id"].to_numpy(np.int64)
        val = g["value_bin"].to_numpy(np.int64)
        dt = g["dt_hours"].to_numpy(np.float32)
        dt_bucket = bucketize_dt(dt, n_buckets=self.n_dt_buckets)

        var_in = np.zeros((self.max_len,), dtype=np.int64)
        val_in = np.zeros((self.max_len,), dtype=np.int64)
        dt_in = np.zeros((self.max_len,), dtype=np.int64)
        attn = np.zeros((self.max_len,), dtype=np.bool_)

        var_in[:T] = var
        val_in[:T] = val
        dt_in[:T] = dt_bucket
        attn[:T] = True

        y = float(self.y_map[pid])

        return (
            torch.from_numpy(var_in),
            torch.from_numpy(val_in),
            torch.from_numpy(dt_in),
            torch.from_numpy(attn),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(pid, dtype=torch.long),
        )
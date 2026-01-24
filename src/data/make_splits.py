from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def make_splits(
    labels_path: Path,
    out_dir: Path,
    test_hospital_frac: float = 0.2,
    val_patient_frac: float = 0.1,
    seed: int = 42,
) -> None:
    """Create hospital-held-out train/val/test splits."""

    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_parquet(labels_path)

    required = {"patient_id", "hospital_id"}
    if not required.issubset(labels.columns):
        raise ValueError(
            f"labels.parquet must contain columns {required}. "
            f"Found: {set(labels.columns)}"
        )

    # Pick hospitals to hold out for testing
    hospitals = labels["hospital_id"].dropna().unique()
    hospitals = hospitals.astype(int)

    rng.shuffle(hospitals)

    n_test_hospitals = max(1, int(len(hospitals) * test_hospital_frac))
    test_hospitals = set(hospitals[:n_test_hospitals])
    train_hospitals = set(hospitals[n_test_hospitals:])

    # Split patients by hospital
    test_patients = labels[labels["hospital_id"].isin(test_hospitals)]
    trainval_patients = labels[labels["hospital_id"].isin(train_hospitals)]

    # Validation split within training hospitals
    patient_ids = trainval_patients["patient_id"].unique()
    rng.shuffle(patient_ids)

    n_val = int(len(patient_ids) * val_patient_frac)
    val_ids = set(patient_ids[:n_val])
    train_ids = set(patient_ids[n_val:])

    train_patients = trainval_patients[trainval_patients["patient_id"].isin(train_ids)]
    val_patients = trainval_patients[trainval_patients["patient_id"].isin(val_ids)]

    # Save splits to disk
    train_patients.to_parquet(out_dir / "train_patients.parquet", index=False)
    val_patients.to_parquet(out_dir / "val_patients.parquet", index=False)
    test_patients.to_parquet(out_dir / "test_patients.parquet", index=False)

    splits = {
        "seed": seed,
        "test_hospital_frac": test_hospital_frac,
        "val_patient_frac": val_patient_frac,
        "n_hospitals_total": int(len(hospitals)),
        "n_hospitals_test": int(len(test_hospitals)),
        "test_hospitals": sorted(map(int, test_hospitals)),
        "train_hospitals": sorted(map(int, train_hospitals)),
        "n_patients": {
            "train": int(train_patients.shape[0]),
            "val": int(val_patients.shape[0]),
            "test": int(test_patients.shape[0]),
        },
    }

    with open(out_dir / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    print("Split summary:")
    print(json.dumps(splits["n_patients"], indent=2))
    print(f"Test hospitals: {len(test_hospitals)} / {len(hospitals)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create hospital-held-out data splits.")
    parser.add_argument("--labels", type=str, required=True, help="Path to labels.parquet")
    parser.add_argument("--out_dir", type=str, default="data/processed")
    parser.add_argument("--test_hospital_frac", type=float, default=0.2)
    parser.add_argument("--val_patient_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    make_splits(
        labels_path=Path(args.labels),
        out_dir=Path(args.out_dir),
        test_hospital_frac=args.test_hospital_frac,
        val_patient_frac=args.val_patient_frac,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def bin_value(x: float, edges: list[float]) -> int:
    # returns bin index in [0..len(edges)-2]
    # edges are sorted; np.digitize returns 1..len(edges)-1
    idx = int(np.digitize([x], edges, right=False)[0] - 1)
    return max(0, min(idx, len(edges) - 2))


def tokenize(events_path: Path, tokenizer_dir: Path, out_path: Path) -> None:
    events = pd.read_parquet(events_path)

    required = {"patient_id", "hospital_id", "time_hours", "variable", "value"}
    if not required.issubset(events.columns):
        raise ValueError(f"events.parquet must contain {required}. Found: {set(events.columns)}")

    variable_vocab = load_json(tokenizer_dir / "variable_vocab.json")
    value_bins = load_json(tokenizer_dir / "value_bins.json")

    events["variable"] = events["variable"].astype(str).str.strip()
    events = events[events["variable"].isin(variable_vocab.keys())].copy()

    # map variable -> id
    events["variable_id"] = events["variable"].map(variable_vocab).astype(int)

    # bin continuous values per variable
    def _bin_row(row) -> int:
        v = row["variable"]
        edges = value_bins[v]["edges"]
        return bin_value(float(row["value"]), edges)

    events["value_bin"] = events.apply(_bin_row, axis=1).astype(int)

    # compute dt (time gap) per patient
    events = events.sort_values(["patient_id", "time_hours"]).reset_index(drop=True)
    events["dt_hours"] = events.groupby("patient_id")["time_hours"].diff().fillna(0.0).clip(lower=0.0)

    # keep only what the model needs
    tokens = events[
        ["patient_id", "hospital_id", "time_hours", "dt_hours", "variable_id", "value_bin"]
    ].copy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokens.to_parquet(out_path, index=False)

    print(f"Saved tokenized events: {out_path} (rows={len(tokens):,})")
    print(f"Patients: {tokens['patient_id'].nunique():,}")
    print(f"Variables: {len(variable_vocab)}")


def main() -> None:
    p = argparse.ArgumentParser(description="Tokenize ICU events using fitted tokenizer artifacts.")
    p.add_argument("--events", type=str, required=True)
    p.add_argument("--tokenizer", type=str, required=True)
    p.add_argument("--out_path", type=str, required=True)
    args = p.parse_args()

    tokenize(Path(args.events), Path(args.tokenizer), Path(args.out_path))


if __name__ == "__main__":
    main()
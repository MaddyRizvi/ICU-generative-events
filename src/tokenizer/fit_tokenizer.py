from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


DEFAULT_VARS = ["HR", "MAP", "SpO2", "lactate", "creatinine"]


def fit_tokenizer(events_path: Path, out_dir: Path, variables: List[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_parquet(events_path)

    # Expect columns from your Step 7 builder:
    # patient_id, time_hours, event_type, variable, value, hospital_id
    required = {"patient_id", "time_hours", "variable", "value"}
    if not required.issubset(events.columns):
        raise ValueError(f"events.parquet must contain {required}. Found: {set(events.columns)}")

    # Normalize variable names
    events["variable"] = events["variable"].astype(str).str.strip()

    # If user passed variables, restrict; else use DEFAULT_VARS that exist
    if variables:
        keep = set([v.strip() for v in variables])
    else:
        keep = set([v for v in DEFAULT_VARS if v in set(events["variable"].unique())])

    if not keep:
        # fallback: use top variables by frequency
        keep = set(events["variable"].value_counts().head(10).index.tolist())

    events = events[events["variable"].isin(keep)].copy()

    # Build variable vocab
    # Reserve 0 for PAD if you want later; here start at 1
    var_list = sorted(keep)
    variable_vocab: Dict[str, int] = {v: i + 1 for i, v in enumerate(var_list)}

    # Build value bins per variable (quantile bins are robust)
    value_bins: Dict[str, Dict] = {}
    for v in var_list:
        vals = events.loc[events["variable"] == v, "value"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if len(vals) < 100:
            # fallback to min/max bins if too few
            edges = np.linspace(vals.min(), vals.max(), num=11).tolist() if len(vals) else [0.0, 1.0]
        else:
            # 10 bins by quantiles
            qs = np.quantile(vals, np.linspace(0.0, 1.0, 11))
            # ensure strictly increasing edges
            qs = np.unique(qs)
            if len(qs) < 3:
                qs = np.linspace(vals.min(), vals.max(), num=11)
            edges = qs.tolist()
        value_bins[v] = {"edges": edges}

    (out_dir / "variable_vocab.json").write_text(json.dumps(variable_vocab, indent=2))
    (out_dir / "value_bins.json").write_text(json.dumps(value_bins, indent=2))

    print(f"Saved tokenizer to: {out_dir}")
    print(f"Variables: {var_list}")


def main() -> None:
    p = argparse.ArgumentParser(description="Fit tokenizer for ICU events (variable vocab + value bins).")
    p.add_argument("--events", type=str, required=True, help="Path to events.parquet")
    p.add_argument("--out_dir", type=str, required=True, help="Output dir for tokenizer artifacts")
    p.add_argument(
        "--variables",
        type=str,
        default="",
        help="Comma-separated variables to include (e.g., HR,MAP,SpO2,lactate,creatinine). Leave empty to auto-pick.",
    )
    args = p.parse_args()

    vars_list = [v.strip() for v in args.variables.split(",") if v.strip()] if args.variables else []
    fit_tokenizer(Path(args.events), Path(args.out_dir), vars_list)


if __name__ == "__main__":
    main()
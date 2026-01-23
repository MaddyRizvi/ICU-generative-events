from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    # pandas reads .csv.gz directly
    return pd.read_csv(path, low_memory=False)


def _find(raw_dir: Path, name: str) -> Path:
    # name like "patient" -> patient.csv.gz or patient.csv
    name = name.lower()
    for p in raw_dir.iterdir():
        if not p.is_file():
            continue
        n = p.name.lower()
        if n == f"{name}.csv.gz" or n == f"{name}.csv":
            return p
    raise FileNotFoundError(f"Missing {name}.csv.gz in {raw_dir}. Found: {[p.name for p in raw_dir.iterdir()]}")


def build_events(raw_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------
    # 1) PATIENT (labels + hospital_id)
    # -----------------------
    patient_path = _find(raw_dir, "patient")
    patient = _read_csv(patient_path)

    required_patient = ["patientunitstayid", "hospitalid", "unitadmittime24", "unitdischargeoffset"]
    for c in required_patient:
        if c not in patient.columns:
            raise ValueError(f"{patient_path.name} missing '{c}'. Columns: {patient.columns.tolist()}")

    labels = patient[["patientunitstayid", "hospitalid", "unitdischargeoffset"]].copy()
    labels["los_hours"] = labels["unitdischargeoffset"] / 60.0

    # Mortality: use unitdischargestatus if present
    if "unitdischargestatus" in patient.columns:
        labels["mortality"] = patient["unitdischargestatus"].astype(str).str.lower().eq("expired").astype(int)
    else:
        labels["mortality"] = 0

    labels = labels.rename(
        columns={
            "patientunitstayid": "patient_id",
            "hospitalid": "hospital_id",
        }
    )[["patient_id", "hospital_id", "los_hours", "mortality"]]

    # -----------------------
    # 2) VITALS (vitalPeriodic preferred, fallback to vitalAperiodic)
    # -----------------------
    vitals_stem = None
    for stem in ["vitalperiodic", "vitalaperiodic", "vitalPeriodic", "vitalAperiodic"]:
        try:
            vitals_path = _find(raw_dir, stem)
            vitals_stem = stem
            break
        except FileNotFoundError:
            continue
    if vitals_stem is None:
        raise FileNotFoundError("Could not find vitalperiodic.csv.gz or vitalaperiodic.csv.gz")

    vitals = _read_csv(vitals_path)

    # your columns (from you):
    # patientunitstayid, observationoffset, sao2, heartrate, systemicmean, ...
    for c in ["patientunitstayid", "observationoffset"]:
        if c not in vitals.columns:
            raise ValueError(f"{vitals_path.name} missing '{c}'. Columns: {vitals.columns.tolist()}")

    vitals["time_hours"] = vitals["observationoffset"] / 60.0

    # We’ll use: heartrate, systemicmean (MAP proxy), sao2
    vital_vars = []
    if "heartrate" in vitals.columns:
        vital_vars.append(("heartrate", "HR"))
    if "systemicmean" in vitals.columns:
        vital_vars.append(("systemicmean", "MAP"))
    if "sao2" in vitals.columns:
        vital_vars.append(("sao2", "SpO2"))

    if not vital_vars:
        raise ValueError(
            f"No expected vital columns found in {vitals_path.name}. "
            f"Looked for heartrate/systemicmean/sao2. Columns: {vitals.columns.tolist()}"
        )

    vital_events = []
    for col, varname in vital_vars:
        df = vitals[["patientunitstayid", "time_hours", col]].dropna()
        df = df.rename(columns={"patientunitstayid": "patient_id", col: "value"})
        df["event_type"] = "vital"
        df["variable"] = varname
        vital_events.append(df)

    vital_events = pd.concat(vital_events, ignore_index=True)

    # -----------------------
    # 3) LABS
    # -----------------------
    lab_path = _find(raw_dir, "lab")
    lab = _read_csv(lab_path)

    needed = ["patientunitstayid", "labresultoffset", "labname", "labresult"]
    for c in needed:
        if c not in lab.columns:
            raise ValueError(f"{lab_path.name} missing '{c}'. Columns: {lab.columns.tolist()}")

    lab["time_hours"] = lab["labresultoffset"] / 60.0
    lab["labname_norm"] = lab["labname"].astype(str).str.strip().str.lower()

    keep = {"lactate", "creatinine"}
    lab = lab[lab["labname_norm"].isin(keep)].copy()

    lab_events = lab.rename(
        columns={
            "patientunitstayid": "patient_id",
            "labname_norm": "variable",
            "labresult": "value",
        }
    )[["patient_id", "time_hours", "variable", "value"]]
    lab_events["event_type"] = "lab"

    # -----------------------
    # 4) COMBINE + ADD HOSPITAL ID
    # -----------------------
    events = pd.concat([vital_events, lab_events], ignore_index=True)
    events = events.merge(labels[["patient_id", "hospital_id"]], on="patient_id", how="left")
    events = events.sort_values(["patient_id", "time_hours"]).reset_index(drop=True)

    # -----------------------
    # 5) SAVE
    # -----------------------
    events_path = out_dir / "events.parquet"
    labels_path = out_dir / "labels.parquet"

    events.to_parquet(events_path, index=False)
    labels.to_parquet(labels_path, index=False)

    print(f"Saved events:  {events_path} (rows={len(events):,})")
    print(f"Saved labels: {labels_path} (patients={len(labels):,})")
    print(f"Vitals source: {vitals_path.name} | vars used: {[v for _, v in vital_vars]}")
    print(f"Lab vars kept: {sorted(list(keep))}")


def main() -> None:
    p = argparse.ArgumentParser(description="Build eICU event stream from csv/csv.gz files (lowercase schema).")
    p.add_argument("--raw_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="data/processed")
    args = p.parse_args()
    build_events(Path(args.raw_dir), Path(args.out_dir))


if __name__ == "__main__":
    main()

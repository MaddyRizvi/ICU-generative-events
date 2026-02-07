# ICU-generative-events
**Generative pretraining for irregular ICU clinical event sequences using eICU-CRD**

---

## Abstract
ICU data are difficult to model because measurements are recorded at irregular times, values are continuous, and data collection depends on clinical decisions that vary across hospitals. As a result, predictive models trained on data from a single hospital often perform poorly when applied elsewhere. This project presents an end-to-end pipeline for learning patient representations from ICU event data using the eICU Collaborative Research Database. ICU stays are converted into sequences of timestamped vital signs and laboratory measurements. A Transformer model is pretrained using a masked event objective and then fine-tuned for in-hospital mortality prediction. Evaluation is performed using hospital-held-out splits to study generalisation across clinical settings.

---

## Overview
This repository implements a complete clinical machine learning workflow:

1. Convert raw ICU tables into irregular clinical event sequences
2. Learn general patient representations using self-supervised generative pretraining
3. Fine-tune the pretrained model for mortality prediction
4. Evaluate performance under cross-hospital generalisation

---

## Pipeline

### 1. Event Stream Construction
Raw eICU tables (patient demographics, vital signs, and laboratory results) are converted into longitudinal event streams. Each ICU stay is represented as a sequence of timestamped events with continuous values and associated hospital identifiers.

### 2. Hospital-Held-Out Data Splits
Patients are split into training, validation, and test sets by hospital. Entire hospitals are held out during testing to assess model robustness to domain shift.

### 3. Tokenisation and Discretisation
Clinical variables are mapped to categorical identifiers, continuous values are discretised into variable-specific bins, and time gaps between events are encoded using bucketed representations.

### 4. Self-Supervised Pretraining
A Transformer encoder is pretrained using masked event modeling. Random subsets of events are masked, and the model learns to reconstruct the masked variables and values from surrounding context.

### 5. Downstream Fine-Tuning
The pretrained encoder is fine-tuned with a lightweight classification head for in-hospital mortality prediction.

### 6. Evaluation Across Hospitals
Performance is evaluated on patients from unseen hospitals to quantify cross-site generalisation.

---

## Key Idea
Each ICU stay is modeled as a sequence of clinical events, where each event contains:
- `patient_id`
- `time_hours` (since ICU admission)
- `variable` (e.g., HR, MAP, SpO2, lactate, creatinine)
- `value` (continuous measurement)
- `hospital_id` (used for generalisation analysis)

After tokenisation, events are represented using:
- `variable_id` (categorical)
- `value_bin` (discretised continuous value)
- `dt_hours` (time gap) → bucketed for modeling

Pretraining uses **Masked Event Modeling** (BERT-style):
- Randomly mask event content
- Train a Transformer encoder to reconstruct masked `variable_id` and `value_bin`

---

## Data: eICU Collaborative Research Database (CSV.GZ)

This repository is designed to work with eICU-CRD tables provided as CSV.GZ files.

**Important:** Patient-level eICU data are not included in this repository. You must download the data separately and place the files locally.

### Expected Raw Directory Structure
```
data/raw/eicu_demo/
├── patient.csv.gz
├── lab.csv.gz
└── vitalPeriodic.csv.gz (or vitalAperiodic.csv.gz)
```

---

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Build Events from CSV
```bash
python -m src.data.build_events_from_csv --raw_dir data/raw/eicu_demo --out_dir data/processed
```

### 2. Create Train/Val/Test Splits
```bash
python -m src.data.make_splits --labels data/processed/labels.parquet --out_dir data/processed
```

### 3. Fit Tokenizer
```bash
python -m src.tokenizer.fit_tokenizer --events data/processed/events.parquet --out_dir data/artifacts
```

### 4. Tokenize Events
```bash
python -m src.tokenizer.tokenize_events --events data/processed/events.parquet --tokenizer data/artifacts --out_path data/processed/tokens.parquet
```

### 5. Pretrain Model
```bash
python -m src.train.pretrain --tokens data/processed/tokens.parquet --labels data/processed/labels.parquet --out outputs/checkpoints/pretrain.pt
```

### 6. Fine-Tune for Mortality Prediction
```bash
python -m src.train.finetune_mortality --tokens data/processed/tokens.parquet --labels data/processed/labels.parquet --pretrained outputs/checkpoints/pretrain.pt --out outputs/checkpoints/mortality.pt
```

---

## Project Structure

```
src/
├── data/              # Data processing scripts
├── models/            # Transformer architecture
├── tokenizer/         # Tokenization utilities
└── train/             # Training scripts (pretrain & finetune)
```

---

## License

MIT License

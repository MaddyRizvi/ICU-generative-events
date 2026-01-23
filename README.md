# ICU-generative-events
**Generative pretraining for irregular ICU clinical event sequences (eICU-CRD) with hospital-held-out generalisation.**

This repository implements an end-to-end clinical ML pipeline inspired by recent “foundation model” approaches for ICU data:
1) convert ICU tables into **irregular event streams** (vitals/labs),
2) learn general patient representations via **self-supervised masked event modeling**,
3) fine-tune the pretrained backbone for **mortality prediction**, and
4) evaluate **cross-hospital generalisation** using hospital-held-out splits.

> Motivation: ICU data are not like text. Measurements occur at irregular times, values are continuous, and missingness is decision-driven (what is recorded depends on clinical choices). Models trained on one hospital often fail to generalise to others. This repo provides a reproducible baseline for generative pretraining on event sequences and evaluating domain shift.

---

## Key idea
We represent each ICU stay as a **sequence of clinical events**, each event containing:
- `patient_id`
- `time_hours` (since unit admission)
- `variable` (e.g., HR, MAP, SpO2, lactate, creatinine)
- `value` (continuous measurement)
- `hospital_id` (domain identifier for generalisation experiments)

We then tokenize events into:
- `variable_id` (categorical)
- `value_bin` (discretised continuous value)
- `dt_hours` (time gap) → bucketed for modeling

Pretraining uses **Masked Event Modeling** (BERT-style):
- randomly mask event content,
- train a Transformer encoder to reconstruct masked `variable_id` and `value_bin`.

---

## Data: eICU Collaborative Research Database (CSV.GZ)
This repo is designed to work with **eICU-CRD** tables distributed as `*.csv.gz` files.

**Important:** Patient-level eICU data are not included in this repository.  
You must download the data separately and place the files locally.

Expected raw directory:
# ICU-generative-events
**Generative pretraining for irregular ICU clinical event sequences (eICU-CRD) with hospital-held-out generalisation.**

This repository implements an end-to-end clinical ML pipeline inspired by recent “foundation model” approaches for ICU data:
1) convert ICU tables into **irregular event streams** (vitals/labs),
2) learn general patient representations via **self-supervised masked event modeling**,
3) fine-tune the pretrained backbone for **mortality prediction**, and
4) evaluate **cross-hospital generalisation** using hospital-held-out splits.

> Motivation: ICU data are not like text. Measurements occur at irregular times, values are continuous, and missingness is decision-driven (what is recorded depends on clinical choices). Models trained on one hospital often fail to generalise to others. This repo provides a reproducible baseline for generative pretraining on event sequences and evaluating domain shift.

---

## Key idea
We represent each ICU stay as a **sequence of clinical events**, each event containing:
- `patient_id`
- `time_hours` (since unit admission)
- `variable` (e.g., HR, MAP, SpO2, lactate, creatinine)
- `value` (continuous measurement)
- `hospital_id` (domain identifier for generalisation experiments)

We then tokenize events into:
- `variable_id` (categorical)
- `value_bin` (discretised continuous value)
- `dt_hours` (time gap) → bucketed for modeling

Pretraining uses **Masked Event Modeling** (BERT-style):
- randomly mask event content,
- train a Transformer encoder to reconstruct masked `variable_id` and `value_bin`.

---

## Data: eICU Collaborative Research Database (CSV.GZ)
This repo is designed to work with **eICU-CRD** tables distributed as `*.csv.gz` files.

**Important:** Patient-level eICU data are not included in this repository.  
You must download the data separately and place the files locally.

Expected raw directory:

data/raw/eicu_demo/
patient.csv.gz
lab.csv.gz
vitalPeriodic.csv.gz (or vitalAperiodic.csv.gz)

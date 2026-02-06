# ICU-generative-events  
**Generative pretraining for irregular ICU clinical event sequences using eICU-CRD**

---

## Abstract
ICU data are difficult to model because measurements are recorded at irregular times, values are continuous, and data collection depends on clinical decisions that vary across hospitals. As a result, predictive models trained on data from a single hospital often perform poorly when applied elsewhere.  
This project presents an end-to-end pipeline for learning patient representations from ICU event data using the eICU Collaborative Research Database. ICU stays are converted into sequences of timestamped vital signs and laboratory measurements. A Transformer model is pretrained using a masked event objective and then fine-tuned for in-hospital mortality prediction. Evaluation is performed using hospital-held-out splits to study generalisation across clinical settings. The repository provides a reproducible baseline for representation learning on multi-centre ICU event data.

---

## Overview
This repository implements a complete clinical machine learning workflow:

1. Convert raw ICU tables into irregular clinical event sequences  
2. Learn general patient representations using self-supervised generative pretraining  
3. Fine-tune the pretrained model for mortality prediction  
4. Evaluate performance under cross-hospital generalisation  

---

## Pipeline

### 1. Event stream construction  
Raw eICU tables (patient demographics, vital signs, and laboratory results) are converted into longitudinal event streams. Each ICU stay is represented as a sequence of timestamped events with continuous values and associated hospital identifiers.

### 2. Hospital-held-out data splits  
Patients are split into training, validation, and test sets by hospital. Entire hospitals are held out during testing to assess model robustness to domain shift.

### 3. Tokenisation and discretisation  
Clinical variables are mapped to categorical identifiers, continuous values are discretised into variable-specific bins, and time gaps between events are encoded using bucketed representations.

### 4. Self-supervised pretraining  
A Transformer encoder is pretrained using masked event modeling. Random subsets of events are masked, and the model learns to reconstruct the masked variables and values from surrounding context.

### 5. Downstream fine-tuning  
The pretrained encoder is fine-tuned with a lightweight classification head for in-hospital mortality prediction.

### 6. Evaluation across hospitals  
Performance is evaluated on patients from unseen hospitals to quantify cross-site generalisation.

---

## Key idea
Each ICU stay is modeled as a sequence of clinical events, where each event contains:
- patient_id  
- time_hours (since ICU admission)  
- variable (e.g., HR, MAP, SpO2, lactate, creatinine)  
- value (continuous measurement)  
- hospital_id (used for generalisation analysis)

After tokenisation, events are represented using:
- variable_id  
- value_bin  
- dt_hours (time gap between events)

---

## Data: eICU Collaborative Research Database (CSV.GZ)

This repository is designed to work with eICU-CRD tables provided as CSV.GZ files.

Patient-level data are not included in this repository.  
Users must obtain access to eICU-CRD separately and run preprocessing locally.

Expected raw directory structure:

data/raw/eicu_demo/  
- patient.csv.gz  
- lab.csv.gz  
- vitalPeriodic.csv.gz (or vitalAperiodic.csv.gz)

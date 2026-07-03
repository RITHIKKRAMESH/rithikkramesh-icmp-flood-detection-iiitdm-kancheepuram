# DDoS Attack Detection and Classification using Ensemble Machine Learning

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3%2B-green)](https://lightgbm.readthedocs.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red)](https://xgboost.readthedocs.io/)

This repository implements a rigorous, end-to-end Machine Learning pipeline for **DDoS Flood Attack Detection**, designed in accordance with academic project and evaluation guidelines. The project focuses on cross-domain generalization, temporal-aware data integrity, information leakage prevention, and real-time inference.

---

## 📖 Project Objective
Cybersecurity detection models trained on a single laboratory dataset often suffer from overfitting and perform poorly in real-world environments. The objective of this project is to:
1. Train robust ensemble classifiers on an internal SDN-simulated dataset (**Dataset A3**).
2. Validate the generalization capabilities on a completely unseen, physical network-captured dataset (**Dataset B3**) without retraining.
3. Align network features to a highly generalizable **10 Layer-3/Layer-4 packet header schema** to enable robust operation across different environments.
4. Develop a **Real-Time Detection Adapter** to bridge the machine learning model with live network interfaces.

---

## 📂 Project Directory Structure

```text
├── dataset/                        # Raw network datasets (Dataset_A3.csv, Dataset_B3.csv)
├── preprocessing/                  # Data cleaning and binary protocol mapping scripts
│   └── cleaned_dataset/            # Output location for clean CSV files
├── feature_engineering/            # Feature selection and label encoding modules
│   └── output/                     # Formatted training matrices (Features_A, Labels_A)
├── training/                       # Model training and data balancing scripts
│   ├── training_data/              # Chronological splits (70/10/20) & SMOTE balanced sets
│   ├── split_dataset.py            # Preserves temporal integrity during splits
│   ├── apply_smote.py              # Balances minority benign classes using SMOTE
│   ├── train_random_forest.py      # Baseline RF model
│   ├── train_xgboost.py            # Advanced XGBoost classifier
│   ├── train_lightgbm.py           # Advanced LightGBM classifier
│   └── train_catboost.py           # Advanced CatBoost classifier
├── models/                         # Saved weights (.pkl) for best and individual classifiers
│   └── save_best_model.py          # Script to stage LightGBM as the production model
├── feature_selection/              # Feature importance ranking script based on LightGBM
│   └── output/                     # Feature importance CSV and plotted horizontal bar chart
├── dataset_B/                      # Cross-domain verification scripts
│   └── evaluate_datasetB.py        # Pipeline to test models against the unseen B3 dataset
├── multiclass_pipeline/            # Self-contained folder archiving the multiclass pipeline
├── reports/                        # Saved ablation and generalization text reports
├── realtime_adapter.py             # Live sniffing and packet simulation alert adapter
└── README.md                       # Comprehensive project documentation
```

---

## ⚙️ Feature Schema Alignment
To resolve domain mismatches between the SDN simulation (Dataset A3) and the physical network capture (Dataset B3), the feature pipeline extracts and aligns **10 core packet header features**:

1. **`proto`**: Protocol ID (1: ICMP, 6: TCP, 17: UDP)
2. **`total_length`**: Total packet length in bytes
3. **`src_port`**: Source port number (0 if ICMP)
4. **`dst_port`**: Destination port number (0 if ICMP)
5. **`fin`**: TCP FIN flag indicator
6. **`syn`**: TCP SYN flag indicator
7. **`rst`**: TCP RST flag indicator
8. **`psh`**: TCP PSH flag indicator
9. **`ack`**: TCP ACK flag indicator
10. **`urg`**: TCP URG flag indicator

---

## 🛡️ Leakage Prevention & Data splits
* **Temporal Splits**: To preserve realistic deployment scenarios, random shuffling before splitting is strictly prohibited. The datasets are chronologically ordered: **70% Training**, **10% Validation**, and **20% Testing**. Shuffling is only performed on the isolated training split.
* **Leakage Control**: SMOTE oversampling and feature engineering fits are computed **exclusively on the training subset** and subsequently applied to validation/test slices to prevent data contamination.
* **Class Balancing**: Implemented via **SMOTE (k_neighbors=5)** on the training set, balancing Benign (3,033) vs Attack (136,420) classes to a stable **136,420 vs 136,420** distribution.

---

## 📊 Experimental Results & Evaluations

### 1. Cross-Domain Generalization on Unseen Dataset B3 (Binary)
All classifiers were evaluated on the independent **Dataset B3** (210,000 Benign / 210,000 Attack samples) without any retraining. **LightGBM is the best-performing model**:

| Model | Accuracy | Benign Recall | Attack Recall | F1-Score | Result |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** (Baseline) | 71.89% | 81.02% | 62.76% | 0.6907 | Stably generalizable |
| **XGBoost** | 64.80% | 29.61% | 100.00% | 0.7397 | High attack recall |
| **LightGBM** | **73.61%** | **47.22%** | **100.00%** | **0.7912** | **Best Model (Winner)** |
| **CatBoost** | 48.63% | 30.59% | 66.67% | 0.5648 | Heavy concept shift |

### 2. Feature Importance Ranking (LightGBM)
Derived from the best-performing model, the top features ranked by splits importance are:
1. `dst_port` (Importance: 7,734)
2. `total_length` (Importance: 535)
3. `src_port` (Importance: 286)
4. `proto` (Importance: 202)
5. `syn` (Importance: 201)

---

## ⏱️ Real-Time Detection Adapter
The **`realtime_adapter.py`** script serves as the bridge between network traffic and the trained LightGBM model. 

```text
    [ Real Network Traffic ]
               |
               v
    [ Packet Sniffer / Capture Layer ] --(Scapy sniff() / Simulation Engine)
               |
               v
    [ Feature Extractor Module ] -------(Extracts proto, length, TCP flags, ports)
               |
               v
    [ Data Transform Layer ] -----------(Aligns to 10-feature model schema)
               |
               v
    [ Inference Engine ] ---------------(LightGBM Best Binary Model)
               |
               +------------------------+
               v                        v
      [ Prediction: 0 ]        [ Prediction: 1 ]
               |                        |
               v                        v
       [ Normal Traffic ]       [ DDOS ATTACK ALERT! ]
       (Console Logging)        (Red Alert Printed)
```

It supports two capture modes:
1. **Live Capture Mode (Choice 1)**: Leverages Scapy to sniff raw packets on your network card, dynamically extracting features and outputting alerts.
2. **Simulation Mode (Choice 2)**: Falls back to simulating realistic background TCP/UDP traffic and random bursts of SYN/UDP/ICMP flood attacks if Npcap or administrator privileges are missing.

---

## 🚀 Execution & Replication Guide

To run the entire pipeline sequentially, execute the following commands in order:

```bash
# 1. Preprocess raw training dataset (Dataset_A3.csv)
python preprocessing/preprocess_dataset.py

# 2. Extract features and align schema
python feature_engineering/feature_engineering.py

# 3. Create temporal splits (70/10/20)
python training/split_dataset.py

# 4. Balance training classes using SMOTE
python training/apply_smote.py

# 5. Train classifiers
python training/train_random_forest.py
python training/train_xgboost.py
python training/train_lightgbm.py
python training/train_catboost.py

# 6. Save LightGBM as best model
python models/save_best_model.py

# 7. Map unseen Dataset B3 and evaluate generalization performance
python dataset/feature_mapping.py
python dataset_B/evaluate_datasetB.py

# 8. Run ablation study and generalization reports
python evaluation/run_ablation_and_generalization.py

# 9. Extract feature importances
python feature_selection/feature_importance.py

# 10. Start the Real-Time Detection Adapter
python realtime_adapter.py
```

# Ablation Study Report

This report documents the systematic removal and modification of components within the ICMP Flood Detection ML pipeline to understand their individual contributions to model performance and generalization.

---

## Experiment 1: Comparing Packet-Level vs. Flow-Level Features

### 1. What was changed?
We restricted the model's feature set from the full 46 features down to a subset of **12 domain-invariant, packet-level features** (`Protocol Type`, `ICMP`, `TCP`, `UDP`, and the individual bitwise TCP flags: `fin`, `syn`, `rst`, `psh`, `ack`, `urg`, `ece`, `cwr`). We completely removed flow-level stats (e.g. `Header_Length`, `flow_duration`, `Tot size`, `AVG`, `Rate`).

### 2. Why was it changed?
This was changed to evaluate the domain generalization gap. Since Dataset A1 consists of flow-level statistics and Dataset B1 consists of packet-level SDN data, we wanted to see if restricting the features to basic packet headers (which exist in both domains) would allow the models to generalize to Dataset B1.

### 3. What was the observed impact?
* **Internal Test Split (A1)**: Accuracy remained extremely high (**99.85%**), indicating that packet-header flags alone contain sufficient signal to separate classes within the same dataset.
* **External Dataset (B1) Evaluation**: Multiclass accuracy rose slightly from **50.00% to 51.87%**. Crucially, the **CatBoost** model's recall on benign traffic improved from **0.00% to 33.33%** (correctly identifying 70,000 benign packets), though it introduced some confusion with UDP attacks.

### 4. What does this reveal about the model?
This reveals that while tree-based classifiers are highly dependent on flow statistics (like rate and size) to distinguish benign traffic, they can learn some classification rules using header flags alone. However, packet-level header flags in a cross-dataset setting are insufficient to completely resolve domain shift because standard benign traffic and DDoS attacks both utilize standard flags, causing overlapping features.

---

## Experiment 2: Removing the Balancing Strategy (SMOTE)

### 1. What was changed?
We bypassed the SMOTE oversampling step and trained the baseline Random Forest and XGBoost classifiers directly on the imbalanced training split of Dataset A1 (where `benign` traffic represents only **5.6%** of the samples).

### 2. Why was it changed?
This was changed to measure the impact of class imbalance on the classifiers and to justify the selection of SMOTE as the balancing technique.

### 3. What was the observed impact?
* **Benign Recall**: Without SMOTE, the recall for the `benign` class dropped significantly (below **85%**), and the model exhibited a high rate of False Positives (falsely flagging normal traffic as DDoS).
* **Overall Accuracy**: Overall accuracy remained high (around 99.4%) because the DDoS classes dominate the dataset, hiding the poor performance on benign traffic.

### 4. What does this reveal about the model?
This reveals that tree-based classifiers are highly sensitive to class imbalance. Without synthetic oversampling (SMOTE), the models optimize for the majority classes (DDoS categories) and fail to learn a robust decision boundary for benign traffic, which is unacceptable for real-world deployment where false alarms must be minimized.

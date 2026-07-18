# Ablation Study Report

This report documents the systematic removal and modification of components within the ICMP Flood Detection ML pipeline to understand their individual contributions to model performance and generalization.

---

## Experiment 1: Comparing Full Feature set vs. Core Packet Headers

### 1. What was changed?
We restricted the model's feature set from the full 10 features down to a subset of **4 core packet-level header features** (`proto`, `total_length`, `src_port`, `dst_port`). We completely removed all TCP flag features (`fin`, `syn`, `rst`, `psh`, `ack`, `urg`).

### 2. Why was it changed?
This was changed to evaluate the contribution of TCP flags to cross-domain generalization. Since some networks do not report flag information or mask flags, we wanted to see if basic headers alone are sufficient.

### 3. What was the observed impact?
* **Internal Validation Split (A3)**: Accuracy remained extremely high across all 8 models (ranging from **99.64%** to **99.95%**), indicating that basic header fields are sufficient to classify DDoS traffic within a single domain.
* **External Dataset (B3) Evaluation**: The accuracy under ablation varies significantly. Models like **XGBoost** and **Gradient Boosting** achieved **69.05%** accuracy, whereas linear models (Logistic Regression) and LightGBM hit **66.67%**. CatBoost, AdaBoost, and Decision Tree dropped to **50.00%** (zero-like discrimination).

### 4. What does this reveal about the model?
This reveals that while tree-based classifiers and neural networks can achieve near-perfect classification on in-domain datasets using basic headers alone, they require additional features like TCP flags to maintain high discrimination capability across domains. Removing flag indicators leads to a severe degradation in out-of-domain generalization.

---

## Experiment 2: Removing the Balancing Strategy (SMOTE)

### 1. What was changed?
We bypassed the SMOTE oversampling step and trained the classifiers directly on the imbalanced training split of Dataset A3 (where benign traffic represents a tiny minority of samples).

### 2. Why was it changed?
This was changed to measure the impact of class imbalance on the classifiers and to justify the selection of SMOTE as the balancing technique.

### 3. What was the observed impact?
* **Benign Recall**: Without SMOTE, the recall for the benign class dropped significantly, and models frequently flagged benign traffic as attacks.
* **Overall Accuracy**: Overall accuracy remained artificially high because the DDoS classes dominate the dataset, hiding the poor performance on benign traffic.

### 4. What does this reveal about the model?
This reveals that classifiers are highly sensitive to class imbalance. Without synthetic oversampling (SMOTE), the models optimize for the majority classes (DDoS categories) and fail to learn a robust decision boundary for benign traffic.

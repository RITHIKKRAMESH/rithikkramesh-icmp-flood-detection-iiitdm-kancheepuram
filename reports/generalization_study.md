# Generalization Study Report (Dataset B3)

This report evaluates the generalization capability of the trained models when exposed to an independent, unseen dataset (`Dataset_B3.csv`) without retraining.

---

## 1. Source and Schema Comparison

* **Dataset A3 (Training Domain)**: A flow-level network statistics dataset. Contains SDN-simulated ICMP flood and DDoS traffic features mapped to a highly generalizable 10-feature Layer-3/Layer-4 schema.
* **Dataset B3 (External Evaluation Domain)**: A physical network packet capture dataset. Aligned using identical preprocessing rules to the 10-feature schema to measure out-of-domain robustness.

---

## 2. Quantitative Evaluation (Cross-Domain)

The final trained models (trained on Dataset A3) were evaluated directly on the mapped Dataset B3 features:

| Model | Accuracy | Macro F1-Score | Benign Recall | Attack Recall |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost** | 64.80% | 0.5983 | 29.61% | 100.00% |
| **LightGBM** | **73.61%** | **0.7163** | **47.22%** | **100.00%** |
| **CatBoost** | 48.63% | 0.4690 | 30.59% | 66.67% |
| **Decision Tree** | 60.94% | 0.5391 | 21.89% | 100.00% |
| **AdaBoost** | 66.51% | 0.6651 | 66.35% | 66.67% |
| **Logistic Regression** | 66.67% | 0.6250 | 100.00% | 33.33% |
| **Gradient Boosting** | 60.94% | 0.5391 | 21.89% | 100.00% |
| **MLP Neural Network** | 47.67% | 0.4657 | 62.00% | 33.33% |

---

## 3. Analysis of Performance Degradation

A significant performance degradation is observed in out-of-domain evaluations, particularly in benign traffic recall:
1. **Flow vs. Packet Shift**: Standard tree-based models trained on Dataset A3 heavily rely on packet size (`total_length`) and ports to make decisions.
2. **Feature Scale Mismatch**: In the physical network capture (B3), benign traffic packets are small, often matching the length signature of flood attacks in Dataset A3, leading to false classifications.
3. **Robustness of Linear/Neural Models**: Models like Logistic Regression and MLP Neural Network are less prone to hard rule boundaries than decision trees, leading to better benign recall, albeit at the cost of attack detection rates.

---

## 4. Control Group: Retraining directly on B3

To verify that the features of B3 are learnable and that the degradation is purely a domain-shift problem, a set of models was trained directly on B3's training split and evaluated on its test split, achieving over **85%** accuracy on average, confirming that the performance drop is due to the cross-dataset shift.


# Generalization Study Report (Dataset B1)

This report evaluates the generalization capability of the trained models when exposed to an independent, unseen dataset (`Dataset_B1.csv`) without retraining.

---

## 1. Source and Schema Comparison

* **Dataset A1 (Training Domain)**: A flow-level network statistics dataset. Contains 456,596 rows and 46 feature columns. It summarizes packet behaviors over time windows (e.g. rate of packets, total bytes, standard deviation of packet lengths).
* **Dataset B1 (External Evaluation Domain)**: An SDN controller packet-in dataset. Contains 420,000 rows and 26 features. Each row represents packet-level features (e.g., individual packet length, switch port, bitwise TCP flag integer).

---

## 2. Quantitative Evaluation (Cross-Domain)

The final trained models (trained on Dataset A1) were evaluated directly on the mapped Dataset B1 features:

| Model | Accuracy | Macro F1-Score | Benign Recall | Attack Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | 16.67% | 0.0714 | 0.00% | 33.33% |
| **XGBoost** | 50.00% | 0.5000 | 0.00% | 100.00% |
| **LightGBM** | 50.00% | 0.5000 | 0.00% | 100.00% |
| **CatBoost** | 50.00% | 0.4333 | 33.33% | 66.67% |

---

## 3. Analysis of Performance Degradation

A significant performance degradation is observed, particularly in **benign traffic recall**:
1. **Flow vs. Packet Shift**: The models trained on Dataset A1 heavily rely on flow-level statistical metrics (like `Header_Length` and `flow_duration`) to classify benign traffic. In the training set, benign flows have an average header length of **1,054,273** bytes.
2. **Feature Scale Mismatch**: In the external packet-level Dataset B1, all records represent individual packets, resulting in a constant `Header_Length` of **5** (20 bytes). 
3. **Classification Failure**: Because B1's benign packet headers are tiny, the models classify them as DDoS attacks.

---

## 4. Control Group: Retraining directly on B1

To verify that the features of B1 are learnable and that the degradation is purely a domain-shift problem, a set of models was trained directly on B1's training split and evaluated on its test split:

* **Random Forest / XGBoost / LightGBM / CatBoost Accuracy**: **88.74%**
* **Macro F1-Score**: **0.8250**
* **Benign Recall**: **100.00%**

This confirms that B1 is highly learnable. The performance drop in the cross-domain evaluation is entirely a **representation domain shift** (flow-level vs. packet-level), highlighting the importance of evaluating models on realistic, heterogeneous network traffic.

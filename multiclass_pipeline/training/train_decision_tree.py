import os
import time
import joblib
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# ==========================================================
# DATA PATHS (Resolved relative to this file)
# ==========================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
X_TRAIN = os.path.join(SCRIPT_DIR, "training_data", "X_train_balanced.csv")
Y_TRAIN = os.path.join(SCRIPT_DIR, "training_data", "y_train_balanced.csv")

X_VAL = os.path.join(SCRIPT_DIR, "training_data", "X_val.csv")
Y_VAL = os.path.join(SCRIPT_DIR, "training_data", "y_val.csv")

MODEL_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "models")

# ==========================================================

print("=" * 70)
print("TRAINING MULTICLASS DECISION TREE MODEL")
print("=" * 70)

X_train = pd.read_csv(X_TRAIN)
y_train = pd.read_csv(Y_TRAIN).iloc[:, 0]

X_val = pd.read_csv(X_VAL)
y_val = pd.read_csv(Y_VAL).iloc[:, 0]

print("\nTraining Samples :", X_train.shape)
print("Validation Samples :", X_val.shape)

model = DecisionTreeClassifier(
    max_depth=12,
    random_state=42
)

print("\nTraining Started...")

start = time.time()

model.fit(X_train, y_train)

end = time.time()

print("\nTraining Completed")

print(f"Training Time : {round(end-start,2)} Seconds")

prediction = model.predict(X_val)

accuracy = accuracy_score(y_val, prediction)

print("\nValidation Accuracy : {:.2f}%".format(accuracy * 100))

os.makedirs(MODEL_DIR, exist_ok=True)

save_path = os.path.join(MODEL_DIR, "decision_tree.pkl")
joblib.dump(model, save_path)

print("\nModel Saved Successfully")
print(save_path)

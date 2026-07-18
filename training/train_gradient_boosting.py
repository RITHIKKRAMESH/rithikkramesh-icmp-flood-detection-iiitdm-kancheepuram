import os
import time
import joblib
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# ==========================================================
# DATA PATHS
# ==========================================================

X_TRAIN = "training/training_data/X_train_balanced.csv"
Y_TRAIN = "training/training_data/y_train_balanced.csv"

X_VAL = "training/training_data/X_val.csv"
Y_VAL = "training/training_data/y_val.csv"

# ==========================================================

print("=" * 70)
print("TRAINING GRADIENT BOOSTING MODEL")
print("=" * 70)

X_train = pd.read_csv(X_TRAIN)
y_train = pd.read_csv(Y_TRAIN).iloc[:, 0]

X_val = pd.read_csv(X_VAL)
y_val = pd.read_csv(Y_VAL).iloc[:, 0]

print("\nTraining Samples :", X_train.shape)
print("Validation Samples :", X_val.shape)

model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=3,
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

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/gradient_boosting.pkl")

print("\nModel Saved Successfully")
print("models/gradient_boosting.pkl")

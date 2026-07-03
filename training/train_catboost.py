import os
import time
import joblib
import pandas as pd

from catboost import CatBoostClassifier
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
print("TRAINING CATBOOST MODEL")
print("=" * 70)

# ==========================================================
# LOAD DATA
# ==========================================================

X_train = pd.read_csv(X_TRAIN)
y_train = pd.read_csv(Y_TRAIN).iloc[:, 0]

X_val = pd.read_csv(X_VAL)
y_val = pd.read_csv(Y_VAL).iloc[:, 0]

print("\nTraining Samples :", X_train.shape)
print("Validation Samples :", X_val.shape)

# ==========================================================
# CREATE MODEL
# ==========================================================

model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.1,
    depth=6,
    loss_function="MultiClass",
    random_seed=42,
    verbose=50  # Print progress every 50 iterations
)

# ==========================================================
# TRAIN MODEL
# ==========================================================

print("\nTraining Started...")

start = time.time()

model.fit(
    X_train,
    y_train,
    eval_set=(X_val, y_val),
    early_stopping_rounds=30
)

end = time.time()

print("\nTraining Completed")
print(f"Training Time : {round(end - start, 2)} Seconds")

# ==========================================================
# VALIDATION
# ==========================================================

prediction = model.predict(X_val)
# CatBoost predict returns 2D array [[class]], squeeze it
prediction = prediction.squeeze()

accuracy = accuracy_score(
    y_val,
    prediction
)

print("\nValidation Accuracy : {:.2f}%".format(
    accuracy * 100
))

# ==========================================================
# SAVE MODEL
# ==========================================================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    model,
    "models/catboost.pkl"
)

print("\nModel Saved Successfully")
print("models/catboost.pkl")

print("\n" + "=" * 70)
print("CATBOOST TRAINING COMPLETED")
print("=" * 70)

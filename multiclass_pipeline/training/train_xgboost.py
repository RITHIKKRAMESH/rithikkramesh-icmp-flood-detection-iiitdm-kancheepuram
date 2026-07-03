import os
import time
import joblib
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# ==========================================================
# DATA PATHS
# ==========================================================

X_TRAIN = "training/training_data/X_train_balanced.csv"
Y_TRAIN = "training/training_data/y_train_balanced.csv"

X_VAL = "training/training_data/X_val.csv"
Y_VAL = "training/training_data/y_val.csv"

# ==========================================================

print("=" * 70)
print("TRAINING XGBOOST MODEL")
print("=" * 70)

X_train = pd.read_csv(X_TRAIN)
y_train = pd.read_csv(Y_TRAIN).iloc[:, 0]

X_val = pd.read_csv(X_VAL)
y_val = pd.read_csv(Y_VAL).iloc[:, 0]

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_val = le.transform(y_val)

print("\nTraining Samples :", X_train.shape)
print("Validation Samples :", X_val.shape)
print("Number of Classes :", len(le.classes_))

model = XGBClassifier(

    n_estimators=300,

    max_depth=8,

    learning_rate=0.1,

    objective="multi:softmax",

    num_class=len(le.classes_),

    random_state=42,

    eval_metric="mlogloss",

    tree_method="hist"

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

joblib.dump(model, "models/xgboost.pkl")

print("\nModel Saved Successfully")

print("models/xgboost.pkl")
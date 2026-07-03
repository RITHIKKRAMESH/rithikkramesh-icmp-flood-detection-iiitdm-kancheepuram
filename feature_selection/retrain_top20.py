import joblib
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

print("=" * 70)
print("RETRAINING USING TOP 20 FEATURES")
print("=" * 70)

# --------------------------------------------------

top20 = pd.read_csv(

    "feature_selection/output/top20_features.csv"

)

selected = top20["Feature"].tolist()

# --------------------------------------------------

X_train = pd.read_csv(

    "training/training_data/X_train_balanced.csv"

)

X_test = pd.read_csv(

    "training/training_data/X_test.csv"

)

y_train = pd.read_csv(

    "training/training_data/y_train_balanced.csv"

).iloc[:,0]

y_test = pd.read_csv(

    "training/training_data/y_test.csv"

).iloc[:,0]

# --------------------------------------------------

X_train = X_train[selected]

X_test = X_test[selected]

# --------------------------------------------------

model = XGBClassifier(

    n_estimators=300,

    learning_rate=0.1,

    max_depth=8,

    objective="multi:softmax",

    num_class=7,

    eval_metric="mlogloss",

    random_state=42,

    tree_method="hist"

)

print("\nTraining Started...")

model.fit(

    X_train,

    y_train

)

print("Training Completed")

# --------------------------------------------------

prediction = model.predict(

    X_test

)

print("\nAccuracy :",accuracy_score(y_test,prediction))

print("Macro Precision :",precision_score(y_test,prediction,average="macro"))

print("Macro Recall :",recall_score(y_test,prediction,average="macro"))

print("Macro F1 :",f1_score(y_test,prediction,average="macro"))

print("Weighted F1 :",f1_score(y_test,prediction,average="weighted"))

# --------------------------------------------------

joblib.dump(

    model,

    "models/best_model_top20.pkl"

)

print("\nTop20 Model Saved Successfully")
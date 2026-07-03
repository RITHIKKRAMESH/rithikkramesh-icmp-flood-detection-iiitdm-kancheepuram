import joblib
import pandas as pd

# -------------------------------------------------

MODEL_PATH = "models/best_model.pkl"

X_VAL = "training/training_data/X_val.csv"
Y_VAL = "training/training_data/y_val.csv"

ENCODER = "models/label_encoder.pkl"

# -------------------------------------------------

model = joblib.load(MODEL_PATH)

encoder = joblib.load(ENCODER)

X = pd.read_csv(X_VAL)

y = pd.read_csv(Y_VAL).iloc[:,0]

# -------------------------------------------------

prediction = model.predict(X)

# -------------------------------------------------

actual = encoder.inverse_transform(y)

predicted = encoder.inverse_transform(prediction)

result = pd.DataFrame({

    "Actual": actual,

    "Predicted": predicted

})

result["Correct"] = result["Actual"] == result["Predicted"]

print(result.head(30))

print()

print("Correct Predictions :", result["Correct"].sum())

print("Wrong Predictions :", (~result["Correct"]).sum())

print("Validation Accuracy :", result["Correct"].mean()*100)

# -------------------------------------------------

result.to_csv(

    "reports/validation_predictions.csv",

    index=False

)

print("\nPrediction file saved.")
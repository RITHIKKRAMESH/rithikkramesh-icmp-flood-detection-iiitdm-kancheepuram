import shutil

SOURCE_MODEL = "models/xgboost.pkl"
DESTINATION_MODEL = "models/best_model.pkl"

shutil.copy(
    SOURCE_MODEL,
    DESTINATION_MODEL
)

print("="*60)
print("BEST MODEL SAVED")
print("="*60)

print("\nBest Model : XGBoost")

print("\nSaved As : models/best_model.pkl")
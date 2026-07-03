import shutil

SOURCE_MODEL = "models/lightgbm.pkl"
DESTINATION_MODEL = "models/best_model.pkl"

shutil.copy(
    SOURCE_MODEL,
    DESTINATION_MODEL
)

print("="*60)
print("BEST MODEL SAVED")
print("="*60)

print("\nBest Model : LightGBM")

print("\nSaved As : models/best_model.pkl")
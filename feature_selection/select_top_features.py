import pandas as pd

print("=" * 70)
print("TOP 20 FEATURE SELECTION")
print("=" * 70)

importance = pd.read_csv(

    "feature_selection/output/feature_importance.csv"

)

top20 = importance.head(20)

print(top20)

top20.to_csv(

    "feature_selection/output/top20_features.csv",

    index=False

)

print("\nTop 20 Features Saved Successfully")
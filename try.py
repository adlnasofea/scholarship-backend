import joblib
feature_columns = joblib.load("models/feature_columns.pkl")
print("Feature column count:", len(feature_columns))
print("Feature columns:")
for col in feature_columns:
    print("-", col)

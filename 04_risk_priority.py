
import pandas as pd
import joblib

# Load the trained model
saved_data = joblib.load("models/genetic_disorder_model.pkl")

model = saved_data["model"]
preprocessor = saved_data["preprocessor"]
label_encoder = saved_data["label_encoder"]
feature_columns = saved_data["feature_columns"]

# Load cleaned dataset for sample predictions
df = pd.read_csv("processed_data/cleaned_genetic_data.csv")

# Remove target and personal identifier columns
drop_columns = [
    "Genetic Disorder",
    "Disorder Subclass",
    "Patient Id",
    "Patient First Name",
    "Family Name"
]

X = df.drop(
    columns=[col for col in drop_columns if col in df.columns],
    errors="ignore"
)

# Match the training feature columns
X = X.reindex(columns=feature_columns)

# Predict probabilities
X_processed = preprocessor.transform(X)
probabilities = model.predict_proba(X_processed)

predicted_indices = probabilities.argmax(axis=1)
predicted_labels = label_encoder.inverse_transform(predicted_indices)
max_probabilities = probabilities.max(axis=1)

# Assign priority based on model confidence
def assign_priority(probability):
    if probability >= 0.70:
        return "High"
    elif probability >= 0.40:
        return "Medium"
    else:
        return "Low"

results = pd.DataFrame({
    "Patient ID": range(1, len(df) + 1),
    "Predicted Category": predicted_labels,
    "Probability": max_probabilities.round(3),
    "Priority Level": [
        assign_priority(p) for p in max_probabilities
    ]
})

print("\n===== GENETIC DISORDER RISK PRIORITY =====")
print("\nTotal Records:", len(results))
print("\nPriority Counts:")
print(results["Priority Level"].value_counts())

print("\nSample Predictions:")
print(results.head(10).to_string(index=False))

# Save results
results.to_csv("processed_data/risk_priority_results.csv", index=False)

print("\nResults saved to:")
print("processed_data/risk_priority_results.csv")

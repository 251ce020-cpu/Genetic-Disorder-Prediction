
import pandas as pd
import requests

# Read one sample record
df = pd.read_csv("processed_data/cleaned_genetic_data.csv")

drop_columns = [
    "Genetic Disorder",
    "Disorder Subclass",
    "Patient Id",
    "Patient First Name",
    "Family Name"
]

features = df.drop(
    columns=[c for c in drop_columns if c in df.columns],
    errors="ignore"
)

# Use the first record as sample input
sample = features.iloc[0].where(
    pd.notna(features.iloc[0]), None
).to_dict()

response = requests.post(
    "http://127.0.0.1:5000/predict",
    json={"features": sample}
)

print("Status Code:", response.status_code)
print("Response:")
print(response.text)

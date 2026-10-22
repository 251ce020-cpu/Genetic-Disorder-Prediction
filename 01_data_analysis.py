
import pandas as pd

# Load the correct dataset files
train = pd.read_csv("data/train_genetic_disorders.csv")
test = pd.read_csv("data/test_genetic_disorders.csv")

print("TRAIN DATASET")
print("Shape:", train.shape)

print("\nColumn Names:")
print(train.columns.tolist())

print("\nFirst 5 Records:")
print(train.head())

print("\nMissing Values:")
print(train.isnull().sum())

print("\nDuplicate Rows:", train.duplicated().sum())

print("\nTEST DATASET")
print("Shape:", test.shape)
print("Columns:", test.columns.tolist())

print("\n===== DATA QUALITY REPORT =====")

print("\nTotal Missing Values:")
print(train.isnull().sum().sum())

print("\nColumns with Missing Values:")
print(train.isnull().sum()[train.isnull().sum() > 0])

print("\nDuplicate Records:")
print(train.duplicated().sum())

print("\nTarget Column Counts:")

for col in ["Genetic Disorder", "Disorder Subclass"]:
    if col in train.columns:
        print(f"\n{col}:")
        print(train[col].value_counts(dropna=False))

# STEP 8: DATA CLEANING

print("\n===== DATA CLEANING =====")

# Create a copy of the training dataset
clean_train = train.copy()

# Remove duplicate rows
before = len(clean_train)
clean_train = clean_train.drop_duplicates()
after = len(clean_train)

print("Duplicate rows removed:", before - after)

# Identify numerical and categorical columns
numeric_cols = clean_train.select_dtypes(
    include=["number"]
).columns

categorical_cols = clean_train.select_dtypes(
    include=["object", "category", "string"]
).columns

# Fill missing numerical values with median
for col in numeric_cols:
    clean_train[col] = clean_train[col].fillna(
        clean_train[col].median()
    )

# Fill missing categorical values with Unknown
for col in categorical_cols:
    clean_train[col] = clean_train[col].fillna(
        "Unknown"
    )

print("\nMissing values after cleaning:")
print(clean_train.isnull().sum().sum())

print("\nOriginal training shape:", train.shape)
print("Cleaned training shape:", clean_train.shape)

print("\nData cleaning completed!")

# STEP 9: CHECK TARGET COLUMNS

print("\n===== TARGET COLUMN ANALYSIS =====")

for col in ["Genetic Disorder", "Disorder Subclass"]:
    if col in clean_train.columns:
        print(f"\nColumn: {col}")
        print("Unique categories:")
        print(clean_train[col].value_counts(dropna=False))

        print("\nNumber of categories:")
        print(clean_train[col].nunique(dropna=False))
    else:
        print(f"\n{col} not found in dataset.")

# STEP 10: SEPARATE TARGET AND FEATURES

print("\n===== TARGET AND FEATURE CHECK =====")

# Check target columns
target_columns = [
    "Genetic Disorder",
    "Disorder Subclass"
]

for col in target_columns:
    if col in clean_train.columns:
        print(f"\n{col}")
        print("Missing values:", clean_train[col].isnull().sum())
        print("Number of classes:", clean_train[col].nunique())
        print("Class names:")
        print(clean_train[col].dropna().unique())

# Display possible patient ID columns
print("\nPossible ID columns:")
print([
    col for col in clean_train.columns
    if "id" in col.lower()
])

# Display dataset columns
print("\nAll columns:")
print(clean_train.columns.tolist())

# STEP 11: FEATURE ANALYSIS

print("\n===== FEATURE ANALYSIS =====")

# Show columns and their data types
print("\nColumn names and data types:")
print(clean_train.dtypes)

# Count numerical and categorical columns
numeric_columns = clean_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = clean_train.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

print("\nNumerical columns:")
print(numeric_columns)

print("\nCategorical columns:")
print(categorical_columns)

print("\nTotal numerical columns:", len(numeric_columns))
print("Total categorical columns:", len(categorical_columns))

# STEP 12: CHECK TARGET DATA

print("\n===== TARGET DATA CHECK =====")

for col in ["Genetic Disorder", "Disorder Subclass"]:
    if col in clean_train.columns:
        print(f"\nTarget: {col}")

        print("\nCategory counts:")
        print(clean_train[col].value_counts(dropna=False))

        print("\nPercentage distribution:")
        print(
            clean_train[col]
            .value_counts(normalize=True, dropna=False)
            .mul(100)
            .round(2)
        )
print("\n===== TARGET AND FEATURE ANALYSIS =====")

target_columns = ["Genetic Disorder", "Disorder Subclass"]

for col in target_columns:
    if col in clean_train.columns:
        print(f"\nTarget Column: {col}")
        print("Categories:")
        print(clean_train[col].value_counts(dropna=False))

print("\n===== ALL AVAILABLE COLUMNS =====")
for col in clean_train.columns:
    print(col)
    print("\n===== FEATURE SELECTION =====")

# Remove columns that are mainly identifiers or personal names
possible_unwanted = [
    "Patient Id",
    "Patient First Name",
    "Family Name"
]

unwanted_columns = [
    col for col in possible_unwanted
    if col in clean_train.columns
]

feature_data = clean_train.drop(
    columns=unwanted_columns
)

print("\nRemoved columns:")
print(unwanted_columns)

print("\nRemaining columns:")
print(feature_data.columns.tolist())

print("\nFeature dataset shape:")
print(feature_data.shape)

print("\nFeature selection completed!")
print("\n===== CATEGORICAL ENCODING CHECK =====")

# Identify categorical columns
categorical_columns = feature_data.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

# Identify numerical columns
numerical_columns = feature_data.select_dtypes(
    include=["number"]
).columns.tolist()

print("\nCategorical columns:")
print(categorical_columns)

print("\nNumerical columns:")
print(numerical_columns)

print("\nCategorical encoding check completed!")
import os

# Create output folder
os.makedirs("processed_data", exist_ok=True)

# Save cleaned dataset
clean_train.to_csv(
    "processed_data/cleaned_genetic_data.csv",
    index=False
)

print("\n===== DATASET SAVED =====")
print("File: processed_data/cleaned_genetic_data.csv")
print("Total rows:", len(clean_train))
print("Total columns:", len(clean_train.columns))
print("Dataset saved successfully!")
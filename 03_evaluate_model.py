
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report
)
from xgboost import XGBClassifier

# Load cleaned dataset
df = pd.read_csv("processed_data/cleaned_genetic_data.csv")

TARGET = "Genetic Disorder"

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found.")

# Remove records without target labels
df = df.dropna(subset=[TARGET]).copy()

# This step evaluates one chosen category against all other categories.
# Choose a real category from your dataset's target counts.
print("Available Genetic Disorder categories:")
print(df[TARGET].value_counts())

POSITIVE_CLASS = input(
    "\nEnter the exact category to prioritize: "
).strip()

if POSITIVE_CLASS not in df[TARGET].astype(str).unique():
    raise ValueError("Category not found. Run again and enter an exact category.")

y = (df[TARGET].astype(str) == POSITIVE_CLASS).astype(int)

drop_columns = [
    TARGET, "Disorder Subclass", "Patient Id",
    "Patient First Name", "Family Name"
]
X = df.drop(columns=[c for c in drop_columns if c in df.columns])

# Remove constant columns
X = X.loc[:, X.nunique(dropna=False) > 1]

# Split before preprocessing to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Use the same preprocessing pipeline as model training
numeric_columns = X_train.select_dtypes(include=["number"]).columns.tolist()
categorical_columns = X_train.select_dtypes(
    include=["object", "category", "string", "bool"]
).columns.tolist()

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

preprocessor = ColumnTransformer([
    ("numeric", SimpleImputer(strategy="median"), numeric_columns),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_columns)
])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train_processed, y_train)

# Predicted probability for the selected category
probabilities = model.predict_proba(X_test_processed)[:, 1]

print("\n===== THRESHOLD COMPARISON =====")
print("Positive category:", POSITIVE_CLASS)
print("Threshold | Precision | Recall")

for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:
    predictions = (probabilities >= threshold).astype(int)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    print(
        f"{threshold:.2f}      | "
        f"{precision:.3f}     | {recall:.3f}"
    )

# Cross-validation on training data
if y_train.value_counts().min() >= 2:
    folds = min(5, int(y_train.value_counts().min()))
    cv = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=42
    )

    # Preprocess each fold independently
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42
        ))
    ])

    # Use raw training features for fold-wise preprocessing
    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=cv, scoring="accuracy"
    )
    print("\nCross-validation accuracy scores:", np.round(cv_scores, 3))
    print("Mean CV accuracy:", round(cv_scores.mean(), 3))
else:
    print("\nCross-validation skipped: too few examples in a class.")

print("\nTest-set report at threshold 0.50:")
predictions_50 = (probabilities >= 0.50).astype(int)
print(classification_report(
    y_test, predictions_50, zero_division=0
))

print(
    "\nReminder: choose a threshold based on measured precision/recall "
    "and the assignment's stated objective. This is an educational "
    "dataset model, not a clinical diagnosis."
)

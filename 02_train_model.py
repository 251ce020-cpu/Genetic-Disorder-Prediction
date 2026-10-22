
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("processed_data/cleaned_genetic_data.csv")

# Select target column
TARGET = "Genetic Disorder"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found. "
        f"Available columns: {df.columns.tolist()}"
    )

# Remove records without target labels
df = df.dropna(subset=[TARGET]).copy()

# Separate features and target
drop_columns = [
    TARGET,
    "Disorder Subclass",
    "Patient Id",
    "Patient First Name",
    "Family Name"
]

X = df.drop(
    columns=[c for c in drop_columns if c in df.columns]
)
y = df[TARGET].astype(str)

# Remove columns that contain only one unique value
constant_columns = [
    col for col in X.columns if X[col].nunique(dropna=False) <= 1
]
X = X.drop(columns=constant_columns)

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

if len(label_encoder.classes_) < 2:
    raise ValueError("Need at least two target categories to train.")

# Stratify only if every category has at least two records
counts = pd.Series(y_encoded).value_counts()
stratify_labels = y_encoded if counts.min() >= 2 else None

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=stratify_labels
)

# Identify columns by data type
numeric_columns = X_train.select_dtypes(include=["number"]).columns.tolist()
categorical_columns = X_train.select_dtypes(
    include=["object", "category", "string", "bool"]
).columns.tolist()

# Preprocessing pipeline
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

# Choose model based on number of target categories
if len(label_encoder.classes_) == 2:
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42
    )
else:
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=len(label_encoder.classes_),
        eval_metric="mlogloss",
        random_state=42
    )

# Fit preprocessing only on training data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

model.fit(X_train_processed, y_train)

# Evaluate model
predictions = model.predict(X_test_processed)

print("\n===== MODEL EVALUATION =====")
print("Target column:", TARGET)
print("Target categories:", list(label_encoder.classes_))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print(
    "Macro Precision:",
    round(precision_score(y_test, predictions, average="macro", zero_division=0), 4)
)
print(
    "Macro Recall:",
    round(recall_score(y_test, predictions, average="macro", zero_division=0), 4)
)
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=list(range(len(label_encoder.classes_))),
        target_names=[str(c) for c in label_encoder.classes_],
        zero_division=0
    )
)

# Save trained model and preprocessing tools
os.makedirs("models", exist_ok=True)

joblib.dump(
    {
        "preprocessor": preprocessor,
        "model": model,
        "label_encoder": label_encoder,
        "feature_columns": X.columns.tolist(),
        "target_column": TARGET
    },
    "models/genetic_disorder_model.pkl"
)

print("\nModel saved to models/genetic_disorder_model.pkl")


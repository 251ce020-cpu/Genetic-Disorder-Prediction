
from flask import Flask, request, jsonify
from pathlib import Path
import pandas as pd
import joblib

app = Flask(__name__)

# Load the existing model bundle
MODEL_PATH = Path(__file__).resolve().parent / "genetic_disorder_bundle.pkl"
bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
label_encoder = bundle["label_encoder"]
feature_columns = bundle["feature_columns"]
raw_columns = bundle["raw_columns"]
categorical_columns = bundle["categorical_columns"]
numeric_medians = bundle["numeric_medians"]


def assign_priority(probability):
    if probability >= 0.75:
        return "High Priority"
    elif probability >= 0.50:
        return "Medium Priority"
    return "Low Priority"


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Genetic Disorder Risk Prioritization API",
        "status": "running",
        "endpoint": "/predict",
        "method": "POST"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict) or not isinstance(
        data.get("features"), dict
    ):
        return jsonify({
            "error": "Send JSON containing a 'features' object."
        }), 400

    try:
        # Convert the supplied patient features to a DataFrame
        input_df = pd.DataFrame([data["features"]])

        # Match the raw input columns used by the existing dashboard
        X = input_df.reindex(columns=raw_columns).copy()

        # Apply the same preprocessing as dashboard.py
        for col in X.columns:
            if col in categorical_columns:
                X[col] = X[col].fillna("Unknown").astype(str)
            else:
                X[col] = pd.to_numeric(X[col], errors="coerce")
                X[col] = X[col].fillna(
                    numeric_medians.get(col, 0)
                )

        # Encode categories and match the trained model's columns
        X = pd.get_dummies(X)
        X = X.reindex(columns=feature_columns, fill_value=0)

        # Predict category and confidence
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)

        predicted_indices = predictions.astype(int)
        categories = label_encoder.inverse_transform(
            predicted_indices
        )

        results = []

        for i, category in enumerate(categories):
            confidence = float(probabilities[i].max())

            results.append({
                "predicted_category": str(category),
                "model_confidence": round(confidence, 4),
                "priority_label": assign_priority(confidence)
            })

        return jsonify({
            "predictions": results,
            "count": len(results),
            "note": (
                "Educational model output only. "
                "Not a medical diagnosis or validated clinical "
                "risk assessment."
            )
        }), 200

    except Exception as e:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed. Check the input features "
                     "and model compatibility.",
            "details": str(e)
        }), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

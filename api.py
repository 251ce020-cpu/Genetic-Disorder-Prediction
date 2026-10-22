
from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load saved model components
saved_data = joblib.load("models/genetic_disorder_model.pkl")

model = saved_data["model"]
preprocessor = saved_data["preprocessor"]
label_encoder = saved_data["label_encoder"]
feature_columns = saved_data["feature_columns"]


def assign_priority(probability):
    if probability >= 0.70:
        return "High"
    elif probability >= 0.40:
        return "Medium"
    return "Low"


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Genetic Disorder Risk Prioritization API",
        "status": "running",
        "endpoint": "/predict",
        "method": "POST"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict) or not isinstance(data.get("features"), dict):
        return jsonify({
            "error": "Send JSON containing a 'features' object."
        }), 400

    try:
        # Convert input into a one-row DataFrame
        input_df = pd.DataFrame([data["features"]])

        # Check required features
        missing = [
            col for col in feature_columns
            if col not in input_df.columns
        ]

        if missing:
            return jsonify({
                "error": "Missing required features",
                "missing_features": missing
            }), 400

        input_df = input_df[feature_columns]

        # Preprocess and predict
        processed = preprocessor.transform(input_df)
        probabilities = model.predict_proba(processed)[0]

        predicted_index = int(probabilities.argmax())
        predicted_category = str(
            label_encoder.inverse_transform([predicted_index])[0]
        )

        confidence = float(probabilities[predicted_index])

        return jsonify({
            "predicted_category": predicted_category,
            "model_confidence": round(confidence, 4),
            "priority_label": assign_priority(confidence),
            "note": (
                "Educational model output only; not a medical diagnosis "
                "or validated clinical risk assessment."
            )
        })

    except Exception as e:
        return jsonify({
            "error": "Could not process input.",
            "details": str(e)
        }), 400


if __name__ == "__main__":
    app.run(debug=False, port=5000)
 
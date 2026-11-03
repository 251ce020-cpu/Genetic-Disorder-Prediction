
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Genetic Disorder Risk Prioritization",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Genetic Disorder Risk Prioritization Platform")
st.caption(
    "Experimental machine-learning prioritization tool — "
    "not a clinical diagnosis."
)

@st.cache_resource
def load_bundle():
    return joblib.load("genetic_disorder_bundle.pkl")

try:
    bundle = load_bundle()
    model = bundle["model"]
    label_encoder = bundle["label_encoder"]
    feature_columns = bundle["feature_columns"]
    raw_columns = bundle["raw_columns"]
    categorical_columns = bundle["categorical_columns"]
    numeric_medians = bundle["numeric_medians"]
except Exception as e:
    st.error(f"Could not load model bundle: {e}")
    st.stop()

st.subheader("Upload Patient Dataset")
uploaded_file = st.file_uploader(
    "Upload a CSV file containing patient features",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        st.write("Dataset preview")
        st.dataframe(data.head(), use_container_width=True)

        if st.button("Predict Disorder Categories", type="primary"):
            patient_ids = (
                data["Patient Id"].astype(str).tolist()
                if "Patient Id" in data.columns
                else [str(i + 1) for i in range(len(data))]
            )

            # Keep the original input columns for patient IDs
            X = data.reindex(columns=raw_columns).copy()

            # Prepare missing and numeric values
            for col in X.columns:
                if col in categorical_columns:
                    X[col] = X[col].fillna("Unknown").astype(str)
                else:
                    X[col] = pd.to_numeric(
                        X[col], errors="coerce"
                    )
                    X[col] = X[col].fillna(
                        numeric_medians.get(col, 0)
                    )

            # Encode categorical columns like during training
            X = pd.get_dummies(X)
            X = X.reindex(
                columns=feature_columns, fill_value=0
            )

            predictions = model.predict(X)
            probabilities = model.predict_proba(X)
            confidence = probabilities.max(axis=1)

            categories = label_encoder.inverse_transform(
                predictions.astype(int)
            )

            results = pd.DataFrame({
                "Patient ID": patient_ids,
                "Predicted Category": categories,
                "Probability": confidence
            })

            results["Priority Level"] = results["Probability"].apply(
                lambda p: (
                    "High Priority" if p >= 0.75
                    else "Medium Priority" if p >= 0.50
                    else "Low Priority"
                )
            )

            st.session_state["prediction_results"] = results

    except Exception as e:
        st.error(f"Prediction failed: {e}")

if "prediction_results" in st.session_state:
    results = st.session_state["prediction_results"]

    st.divider()
    st.subheader("📊 Prediction Dashboard")

    total = len(results)
    high = int((results["Priority Level"] == "High Priority").sum())
    medium = int((results["Priority Level"] == "Medium Priority").sum())
    low = int((results["Priority Level"] == "Low Priority").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", total)
    c2.metric("High Priority", high)
    c3.metric("Medium Priority", medium)
    c4.metric("Low Priority", low)

    st.subheader("Prediction Results")
    st.dataframe(results, use_container_width=True)

    st.subheader("Priority Level Distribution")
    st.bar_chart(results["Priority Level"].value_counts())

    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Predictions CSV",
        data=csv,
        file_name="genetic_disorder_predictions.csv",
        mime="text/csv"
    )

    st.info(
        "Priority levels represent model prediction confidence only. "
        "They do not indicate actual medical risk or diagnosis."
    )

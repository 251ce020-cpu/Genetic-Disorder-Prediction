
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Genetic Disorder Risk Prioritization",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Genetic Disorder Risk Prioritization Platform")
st.write(
    "XGBoost-based educational prediction dashboard"
)

results_path = Path("processed_data/risk_priority_results.csv")

if not results_path.exists():
    st.error(
        "Prediction results file not found. "
        "First run: python 04_risk_priority.py"
    )
    st.stop()

df = pd.read_csv(results_path)

# Check required columns
required_columns = [
    "Patient ID",
    "Predicted Category",
    "Probability",
    "Priority Level"
]

if not all(col in df.columns for col in required_columns):
    st.error("The results CSV is missing required columns.")
    st.stop()

# Summary cards
total_records = len(df)
high_count = (df["Priority Level"] == "High").sum()
medium_count = (df["Priority Level"] == "Medium").sum()
low_count = (df["Priority Level"] == "Low").sum()

st.subheader("Dashboard Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Records", total_records)
col2.metric("High Priority", int(high_count))
col3.metric("Medium Priority", int(medium_count))
col4.metric("Low Priority", int(low_count))

st.divider()

# Filters
st.subheader("Patient Prediction Results")

filter_options = ["All", "High", "Medium", "Low"]

selected_priority = st.selectbox(
    "Filter by Priority",
    filter_options
)

if selected_priority == "All":
    filtered_df = df.copy()
else:
    filtered_df = df[
        df["Priority Level"] == selected_priority
    ].copy()

st.write("Records displayed:", len(filtered_df))

st.dataframe(
    filtered_df,
    use_container_width=True,
    hide_index=True
)

# Download option
csv_data = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Results as CSV",
    data=csv_data,
    file_name="genetic_disorder_priority_results.csv",
    mime="text/csv"
)

# Priority distribution
st.subheader("Priority Distribution")

priority_counts = (
    df["Priority Level"]
    .value_counts()
    .reindex(["High", "Medium", "Low"], fill_value=0)
)

st.bar_chart(priority_counts)

st.caption(
    "Educational prototype only. Priority labels reflect model "
    "confidence under demonstration thresholds, not a validated "
    "medical risk assessment or diagnosis."
)

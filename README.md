
# Genetic Disorder Risk Prioritization Platform

## Project Overview
This project uses Machine Learning and XGBoost to predict genetic disorder categories and prioritize predictions based on model confidence.

## Objectives
- Analyze and preprocess genetic disorder data.
- Perform Exploratory Data Analysis (EDA).
- Train an XGBoost classification model.
- Evaluate model performance.
- Provide predictions through a Flask API.
- Display prediction results using a Streamlit dashboard.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Flask
- Streamlit
- Matplotlib and Seaborn

## Project Features
- Data cleaning and preprocessing
- Data visualization
- Machine Learning classification
- Prediction probabilities
- Experimental High, Medium, and Low priority categories
- Flask prediction API
- Streamlit dashboard
- Downloadable prediction results

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Run the Flask API

```bash
python app.py
```

Open http://127.0.0.1:5001 in your browser.

## Run the Streamlit Dashboard

Open another terminal and run:

```bash
python -m streamlit run dashboard.py
```

The dashboard normally opens at http://localhost:8501.

## Model File
The application requires `genetic_disorder_bundle.pkl` in the project folder.

## Important Disclaimer
This project is for educational and experimental purposes only. Prediction confidence and priority labels are not medical risk estimates and must not be used for clinical diagnosis or treatment decisions.

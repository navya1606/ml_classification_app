import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="ML Classifier App", page_icon="🧠", layout="centered")
st.title("🧠 ML Classifier Web App")
st.write("Predicts using SVM, Decision Tree, Random Forest, and Voting Ensemble.")

# Load saved models
best_svm = joblib.load("best_svm_model.pkl")
tree_model = joblib.load("decision_tree_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

# Sidebar for inputs
st.sidebar.header("Enter feature values")
features = []
for i in range(20):  # 20 features from your dataset
    val = st.sidebar.number_input(f"Feature {i}", value=0.0)
    features.append(val)

X_input = np.array(features).reshape(1, -1)
X_scaled = scaler.transform(X_input)

if st.button("Predict"):
    pred_svm = best_svm.predict(X_scaled)[0]
    pred_tree = tree_model.predict(X_input)[0]
    pred_rf = rf_model.predict(X_input)[0]

    # Voting ensemble
    preds = [pred_svm, pred_tree, pred_rf]
    ensemble_pred = max(set(preds), key=preds.count)

    st.subheader("Predictions")
    st.write(f"**SVM:** {pred_svm}")
    st.write(f"**Decision Tree:** {pred_tree}")
    st.write(f"**Random Forest:** {pred_rf}")
    st.success(f"**Voting Ensemble Prediction:** {ensemble_pred}")

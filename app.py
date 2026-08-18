
import streamlit as stl
import pandas as pd
from pathlib import Path
import numpy as np
import pickle, json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                              recall_score, f1_score, matthews_corrcoef,
                              confusion_matrix, classification_report)

stl.set_page_config(page_title="ML Assignment 2 - Classification Dashboard", layout="wide")

stl.title("🔬 Classification Model Comparison Dashboard")
stl.markdown("**Dataset:** Breast Cancer Wisconsin (Diagnostic) — Binary Classification "
            "(0 = Malignant, 1 = Benign)")

# ---------- Load artifacts ----------
@stl.cache_resource
def load_artifacts():
    base = Path(__file__).resolve().parent
    model_dir = base / "model"
    with open(model_dir / "scaler.pkl", "rb") as file_obj:
        scaler = pickle.load(file_obj)
    with open(model_dir / "feature_names.json") as file_obj:
        feature_names = json.load(file_obj)
    with open(model_dir / "target_names.json") as file_obj:
        target_names = json.load(file_obj)

    model_files = {
        "Logistic Regression": "model/logistic_regression.pkl",
        "Decision Tree": "model/decision_tree.pkl",
        "kNN": "model/knn.pkl",
        "Naive Bayes": "model/naive_bayes.pkl",
        "Random Forest (Ensemble)": "model/random_forest_ensemble.pkl",
    }
    models = {}
    for name, relpath in model_files.items():
        p = model_dir / Path(relpath).name
        with open(p, "rb") as file_obj:
            models[name] = pickle.load(file_obj)
    return scaler, feature_names, target_names, models

scaler, feature_names, target_names, models = load_artifacts()

# ---------- Sidebar ----------
stl.sidebar.header("⚙️ Controls")
uploaded_file = st.sidebar.file_uploader("Upload test data (CSV)", type=["csv"])
selected_model_name = st.sidebar.selectbox("Select Model", list(models.keys()))

stl.sidebar.markdown("---")
stl.sidebar.markdown(
    "CSV must contain the 30 breast-cancer feature columns "
    "(same as `test_data.csv` in the repo), plus an optional `target` column "
    "for evaluation."
)

if uploaded_file is None:
    stl.info("👈 Upload a test CSV file from the sidebar to get started. "
            "You can use the `test_data.csv` provided in the GitHub repository.")
    stl.stop()

# ---------- Load data ----------
df = pd.read_csv(uploaded_file)
stl.subheader("📄 Uploaded Data Preview")
stl.dataframe(df.head())

has_target = "target" in df.columns
if has_target:
    X_input = df[feature_names]
    y_true = df["target"]
else:
    X_input = df[feature_names]
    y_true = None

X_scaled = scaler.transform(X_input)

# ---------- Predictions for selected model ----------
model = models[selected_model_name]
y_pred = model.predict(X_scaled)
y_proba = model.predict_proba(X_scaled)[:, 1]

stl.subheader(f"🔮 Predictions — {selected_model_name}")
pred_df = df.copy()
pred_df["Predicted"] = y_pred
pred_df["Predicted_Label"] = [target_names[p] for p in y_pred]
stl.dataframe(pred_df.head(20))

# ---------- Metrics ----------
stl.subheader("📊 Evaluation Metrics")
if has_target:
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    cols = stl.columns(6)
    for col, (k, v) in zip(cols, metrics.items()):
        col.metric(k, f"{v:.4f}")

    # Confusion matrix
    stl.subheader("🧩 Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    stl.pyplot(fig)

    # Classification report
    stl.subheader("📋 Classification Report")
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    stl.dataframe(pd.DataFrame(report).T.round(4))
else:
    stl.warning("No `target` column found in uploaded CSV — showing predictions only "
               "(metrics require ground-truth labels).")

# ---------- All-models comparison table ----------
stl.subheader("🏆 All Models Comparison (on this uploaded data)")
if has_target:
    comp_rows = []
    for name, mdl in models.items():
        yp = mdl.predict(X_scaled)
        ypr = mdl.predict_proba(X_scaled)[:, 1]
        comp_rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_true, yp),
            "AUC": roc_auc_score(y_true, ypr),
            "Precision": precision_score(y_true, yp),
            "Recall": recall_score(y_true, yp),
            "F1": f1_score(y_true, yp),
            "MCC": matthews_corrcoef(y_true, yp),
        })
    comp_df = pd.DataFrame(comp_rows).set_index("Model").round(4)
    stl.dataframe(comp_df.style.highlight_max(axis=0, color="lightgreen"))
else:
    stl.info("Upload a CSV with a `target` column to see the full model comparison table.")

stl.markdown("---")
stl.caption("BITS Pilani WILP — M.Tech (AIML/DSE) — Machine Learning — Assignment 2")

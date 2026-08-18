"""
Training script for ML Assignment 2
Dataset: Breast Cancer Wisconsin (Diagnostic) - sklearn built-in
30 features, 569 instances, binary classification (malignant/benign)
"""
import pandas as pd
import numpy as np
import pickle, os, json
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                              recall_score, f1_score, matthews_corrcoef)

os.makedirs("model", exist_ok=True)

# ---------- 1.download and Load dataset from UCI- ----------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")  # 0 = malignant, 1 = benign

print(f"Dataset shape: {X.shape}, classes: {np.unique(y)}")
print(f"Dataset head: {X.head}, classes: {np.unique(y)}")

# ---------- 2. Train/test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save test data (features + true label)
test_data = X_test.copy()
test_data["target"] = y_test.values
test_data.to_csv("test_data.csv", index=False)
print("Saved test_data.csv:", test_data.shape)

# ---------- 3. Scale features ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# ---------- 4. Define models ----------
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "kNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=200, random_state=42),
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }
    results[name] = metrics
    print(name, metrics)

    fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    with open(f"model/{fname}.pkl", "wb") as f:
        pickle.dump(model, f)

# ---------- 5. Save results ----------
results_df = pd.DataFrame(results).T
results_df.index.name = "ML Model Name"
results_df.round(4).to_csv("model/results.csv")
print("\n=== Comparison Table ===")
print(results_df.round(4))

with open("model/feature_names.json", "w") as f:
    json.dump(list(X.columns), f)

with open("model/target_names.json", "w") as f:
    json.dump(list(data.target_names), f)

print("\nAll models trained and saved in model/ directory.")

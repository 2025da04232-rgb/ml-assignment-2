# ML Assignment 2 — Classification Model Comparison Dashboard

## a. Problem Statement

The goal of this assignment is to build, evaluate, and deploy multiple classification
models on a real-world dataset, and expose the results through an interactive
Streamlit web application. The task is a **binary classification problem**: predicting
whether a breast tumor is **malignant** or **benign** based on measurements computed
from a digitized image of a fine needle aspirate (FNA) of a breast mass.

## b. Dataset Description

- **Name:** Breast Cancer Wisconsin (Diagnostic) Dataset
- **Source:** UCI Machine Learning Repository (also available via `sklearn.datasets.load_breast_cancer`)
- **Instances:** 569 (≥ 500 ✅)
- **Features:** 30 numeric features (≥ 12 ✅) — e.g., `mean radius`, `mean texture`,
  `mean perimeter`, `mean area`, `mean smoothness`, `mean compactness`,
  `mean concavity`, `mean symmetry`, `worst radius`, `worst texture`, etc.
- **Target:** Binary — `0 = malignant`, `1 = benign`
- **Class balance:** 212 malignant / 357 benign

The full feature set and 20% held-out test split (114 rows) are provided in
`test_data.csv` in this repository — this is the file to upload in the Streamlit app.

## c. GitHub Repository Link

>https://github.com/2025da04232-rgb/ml-assignment-2

Repository structure:
```
ML-Assignment-2/
│-- app.py                
│-- train_models.py       
│-- requirements.txt
│-- README.md
│-- test_data.csv        
│-- model/                
    │-- logistic_regression.pkl
    │-- decision_tree.pkl
    │-- knn.pkl
    │-- naive_bayes.pkl
    │-- random_forest_ensemble.pkl
    │-- scaler.pkl
    │-- results.csv
    │-- feature_names.json
    │-- target_names.json
```

## d. Models Used

All 5 classifiers were trained on the same 80/20 stratified train-test split of the
Breast Cancer dataset, with features standardized using `StandardScaler`.

### Comparison Table

| ML Model Name             | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|----------------------------|---------:|-------:|----------:|-------:|-------:|-------:|
| Logistic Regression        | 0.9825   | 0.9954 | 0.9861    | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree               | 0.9123   | 0.9157 | 0.9559    | 0.9028 | 0.9286 | 0.8174 |
| kNN                         | 0.9561   | 0.9788 | 0.9589    | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes                 | 0.9298   | 0.9868 | 0.9444    | 0.9444 | 0.9444 | 0.8492 |
| Random Forest (Ensemble)    | 0.9561   | 0.9932 | 0.9589    | 0.9722 | 0.9655 | 0.9054 |

### Observations

| ML Model Name             | Observation about model performance |
|----------------------------|--------------------------------------|
| Logistic Regression        | Best overall performer on this dataset. Since the features (after scaling) are largely linearly separable for this diagnostic dataset, a linear decision boundary works extremely well, achieving the highest accuracy, precision, recall, F1, and MCC. |
| Decision Tree               | Weakest performer. A single unconstrained tree overfits the training data and does not generalize as well, leading to the lowest accuracy, AUC, and MCC among all models. |
| kNN                         | Strong performer after feature scaling (distance-based, so scaling was essential). Performs comparably to Random Forest, correctly capturing local neighborhood structure between malignant/benign cases. |
| Naive Bayes                 | Reasonable performance despite its strong (and technically violated) feature-independence assumption; the high AUC (0.987) shows good ranking/probability calibration even though hard-label accuracy is slightly lower. |
| Random Forest (Ensemble)    | Very strong performer — ensembling many decision trees corrects the overfitting problem seen in the single Decision Tree, producing high, stable metrics across the board and the 2nd-highest AUC. |
| **Overall Winner for our selected dataset?** | **Logistic Regression** — it achieved the highest scores on 5 of 6 metrics (Accuracy, Precision, Recall, F1, MCC) and the 2nd-highest AUC, making it the most reliable model for this dataset. Random Forest is a close, more robust alternative for noisier or non-linear real-world data. |

## Streamlit App Features

The deployed app (`app.py`) supports:
- **CSV upload** of test data (use `test_data.csv` from this repo)
- **Model selection dropdown** (choose among the 5 trained models)
- **Live evaluation metrics** (Accuracy, AUC, Precision, Recall, F1, MCC)
- **Confusion matrix** heatmap and full **classification report**
- **Side-by-side comparison table** of all 5 models on the uploaded data

## Live Streamlit App Link
https://ml-assignment-2-fvxx8qiemvztx2wr3pc35z.streamlit.app/


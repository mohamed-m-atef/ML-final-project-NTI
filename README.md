# 🩺 Diabetes Prediction using Machine Learning

## 📌 Project Overview

This repository contains a complete workflow for predicting diabetes using Machine Learning. The work includes:

* Preliminary dataset analysis
* Data preprocessing and balancing
* Multiple model training and evaluation
* Hyperparameter tuning (GridSearch example)
* Practical recommendations and next steps

> This README summarizes the analysis results (from the uploaded files), the approaches used, reproducible code snippets, and suggested procedures to continue improving the project.

---

## 🔎 1) Preliminary Analysis Results (from the uploaded files)

* **Dataset shape:** `(100000, 16)` — 100k rows × 16 columns.
* **Categorical columns (auto label-encoded):** `['gender', 'location', 'smoking_history']`.
* **Detected binary target:** `diabetes`

  * Distribution: `0 → 91,500`, `1 → 8,500`
  * **Imbalance ratio:** ≈ **10.76 : 1** (majority class >> minority class)
* **Outliers (IQR method):** IQR-based detection flagged nearly **100%** of rows as having at least one outlier. Important note: this is expected when binary/one-hot columns are included in the IQR procedure (because Q1 = Q3 = 0, so any `1` becomes an outlier). See the recommended handling below.
* **Class balancing (SMOTE):** Applied SMOTE only on the training split. Example result on training split: `{0: 73,200, 1: 73,200}` (SMOTE produced synthetic minority samples and increased training data size).
* **GridSearch / full training note:** Attempting a full GridSearch over the entire 100k dataset in the analysis environment hit runtime limits (kernel/time). The core EDA and preprocessing were completed successfully; however, long hyperparameter searches should be executed on an environment with greater time/memory: local machine, Google Colab with GPU/TPU, or a cloud server.

---

## 📝 2) Quick Explanations & Immediate Recommendations

### Why did the IQR method flag \~100% outliers?

IQR is not suitable for binary (0/1) or extremely low-variance columns. For a 0/1 column, Q1 = Q3 = 0, so `1` falls outside the IQR bounds and is flagged as an outlier. **Recommendation:** remove binary / categorical columns from the IQR check, or use robust methods that are aware of column types (e.g., treat binaries separately or use robust scaling).

### Imbalance handling

* The dataset is highly imbalanced (\~11:1). SMOTE (Synthetic Minority Over-sampling Technique) was applied on the training set to balance classes.
* **Cautions of SMOTE:** it generates synthetic samples in feature space and can produce unrealistic examples if many categorical features exist or features are not scaled properly.
* **Alternatives:** `class_weight='balanced'` in certain estimators, `RandomUnderSampler`, `SMOTEENN`, `ADASYN`, or using ensemble techniques that handle imbalance.

### Avoid data leakage

* Perform oversampling (SMOTE) **only on training data** or inside cross-validation folds. Do **not** apply SMOTE before the train/test split. Best practice: place SMOTE inside an `imblearn` pipeline so resampling is part of CV folds.

---

## 🧩 3) Models Overview & How They Work (with key hyperparameters)

### A) Boosting (e.g., `GradientBoostingClassifier`) + GridSearch

**Idea:** Build an ensemble of weak learners (usually shallow decision trees). Each new tree focuses on correcting errors made by previous trees — reduces bias and yields strong performance.
**Use when:** you want better results than a single decision tree (medium/large datasets).
**Key hyperparameters:** `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `min_samples_leaf`, `max_features`.
**GridSearch:** systematically tests parameter combinations across folds (e.g., `StratifiedKFold`) with scoring suitable for imbalance (e.g., `f1`, `roc_auc`, or `average_precision`).
**Practical tip:** start with a small grid (e.g., `n_estimators: [50,100]`, `learning_rate: [0.1,0.05]`, `max_depth: [3,5]`) and expand if promising. For large hyperparameter spaces, prefer `RandomizedSearchCV`.

---

### B) Bagging (Bootstrap Aggregating)

**Idea:** Train `N` base estimators on different bootstrap samples and aggregate predictions (vote/average).
**Advantage:** reduces variance and is helpful when the base estimator is high variance (e.g., decision trees).
**Key hyperparameters:** `n_estimators`, `max_samples`, `max_features`, and choice of `base_estimator`.

---

### C) Decision Tree

**Idea:** Tree-based splits on features to reach class decisions.
**Advantages:** easy to interpret, fast, no scaling needed.
**Disadvantages:** prone to overfitting; tune `max_depth`, `min_samples_leaf`, `min_samples_split`.

---

### D) SVM (Support Vector Machine)

**Idea:** Finds a hyperplane that maximizes the margin between classes; can use kernels (`linear`, `rbf`) for non-linear separation.
**Notes:** requires feature scaling; `rbf` kernel is expensive on large datasets. For large problems use `LinearSVC` with `class_weight='balanced'`.

---

### E) Logistic Regression

**Idea:** Linear model producing probability estimates via the logistic (sigmoid) function.
**Advantages:** strong baseline, interpretable coefficients, supports regularization (`C`, `penalty`) and `class_weight` to handle imbalance.

---

### F) Naive Bayes (e.g., `GaussianNB`)

**Idea:** Probabilistic classifier based on Bayes theorem with a naive independence assumption between features.
**Advantages:** extremely fast and scalable; performs surprisingly well in many problems.
**Disadvantages:** independence assumption can limit accuracy when features are correlated.

---

## 🧪 4) Reproducible Pipeline + GridSearch Example (recommended)

> **Note:** Use `imblearn.pipeline.Pipeline` to place SMOTE inside cross-validation and avoid leakage.



## ⚠️ 5) Handling Outliers 

- **Outliers:** Outliers were detected during the analysis.  
  Since their number is relatively large, removing them would cause significant data loss.  
  Therefore, we decided to keep them in the dataset to preserve information.  


---

## 📈 6) Evaluation Recommendations for Imbalanced Data

* **Do not rely on Accuracy alone.** Use: `Precision`, `Recall`, `F1`, `ROC-AUC`, and `PR-AUC` (Precision-Recall AUC).
* If the priority is to minimize `false negatives` (e.g., missing actual diabetes cases), **prioritize Recall**.
* Adjust decision thresholds after training (use ROC / Precision-Recall curves) to trade-off precision vs recall.
* Use `StratifiedKFold` for cross-validation to preserve class ratios in folds.

---

## 🔭 7) Proposed Practical Roadmap (next steps)

1. **Data cleaning & validation:** exclude binary columns from naive IQR checks, apply `RobustScaler` for numerics, correct impossible values.
2. **Quick modeling on a stratified sample (10k–20k rows)** to iterate fast: `LogisticRegression`, `DecisionTree`, `SVM (linear)`, `GaussianNB`, `Bagging`, `Quick GradientBoosting`.
3. **Hyperparameter tuning:** Grid/Randomized search on the top 1–2 candidate models (e.g., `GradientBoosting`, `XGBoost`/`LightGBM` if available).
4. **Explainability:** compute feature importance and SHAP explanations for the final model.
5. **Delivery:** create a polished notebook (or script) + `README.md` and add a results table to the repository README.

**Important environment note:** a full GridSearch over the entire 100k dataset requires more time and memory than allowed in some interactive kernels — prefer running full tuning on a local machine or cloud environment.

---

## ✅ 8) Quick Conclusion

* Completed a preliminary EDA and preprocessing: label-encoding, correlation analysis, outlier detection caveats.
* Identified a strong class imbalance and applied SMOTE on training splits.
* Implemented initial model runs and prepared a robust pipeline example for proper hyperparameter tuning.
* Due to execution limits in the current environment, larger hyperparameter searches should be performed on a machine with extended runtime / memory.

---

## 📂 Files & How to Run (short guide)

1. `diabetes_dataset.csv` — dataset file (provided).
2. `notebook.ipynb` — analysis notebook (contains EDA, preprocessing, modeling).
3. `README.md` — this file.

**Run locally / Colab**

1. Create a Python environment: `python -m venv venv && source venv/bin/activate` (or use conda).
2. Install dependencies: `pip install -r requirements.txt` (or `pip install scikit-learn pandas numpy imbalanced-learn matplotlib seaborn shap`).
3. Open the notebook or run the scripts. For long GridSearch jobs, prefer Google Colab Pro or a cloud VM.

---

## 👥 Authors

* Ahmed Raed
* Mohammed Mahmoud Atef Saleh

---

## 📦 License

MIT License — feel free to reuse and adapt for academic / portfolio purposes.

---

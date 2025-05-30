# Mortgage Default Prediction with Explainable AI (XGBoost + SHAP)

This repository contains the full code, data, and model evaluation used for my MSc Dissertation at Coventry University. The project aims to predict mortgage loan defaults using machine learning and deep learning models while ensuring interpretability using SHAP (SHapley Additive Explanations).

## 📂 Files Included

- `New_df.csv`: Preprocessed dataset with selected features (Q1–Q3 2024 Fannie Mae loan data)
- `loan_default_prediction.ipynb`: Colab notebook for data preprocessing, model training, and SHAP explainability
- `README.md`: Project overview and structure
- `appendix/`: Optional screenshots or supplementary visualizations

## 🧠 Project Overview

- Built ML/DL models including **XGBoost**, **ANN**, and **CNN** to classify mortgage loan default.
- Dataset: ~40,000 loan records, 110 original features → reduced to 12 key predictors.
- Techniques: Feature engineering, SMOTE for class imbalance, model evaluation (F1-score, ROC-AUC), XAI using SHAP.
- Best model: **XGBoost** – F1-score: 0.93 | ROC-AUC: 0.98

## 🔍 Model Interpretability

- SHAP was used to interpret model decisions and feature importance.
- Key influential features: `LOAN_AGE`, `CURRENT_UPB`, `OCLTV`, `ORIG_TERM`, etc.

## 🚀 How to Use

1. Open the notebook in [Google Colab](https://colab.research.google.com/) or download it to your local Jupyter environment.
2. Make sure required libraries are installed (`scikit-learn`, `xgboost`, `shap`, `keras`, etc.)
3. Run the notebook end-to-end to train models and generate explainability plots.

## 💡 Contact

Created by Vivek Kumar
Dissertation Project – MSc Data Science and Computational Intelligence  
Coventry University | 2024–2025  

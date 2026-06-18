# House Price Prediction: Advanced Regression Analysis

An end-to-end machine learning pipeline built to predict residential home prices using advanced regression techniques. This project handles heavy structural missing data, transforms categorical vectors, and utilizes an optimized ensemble model to achieve competitive accuracy on unseen validation sets.

##  Performance Milestone
* **Algorithm:** Random Forest Regressor
* **Kaggle Public Score (RMSLE):** 0.16335
* **Global Leaderboard Placement:** Rank 3,217

---

##  Project Overview
Predicting house prices requires capturing complex, non-linear relationships across a broad spectrum of structural, environmental, and spatial attributes. This repository outlines a reproducible workflow starting from raw, unverified data matrices to a fully production-ready predictive pipeline.

### Key Pipeline Stages:
1. **Data Preprocessing & Cleaning:** Managed structural missingness by dynamically isolating high-null feature parameters (dropping features with >80% missing data) and executing localized median/mode statistical imputations.
2. **Feature Engineering:** Implemented strict Categorical One-Hot Encoding transformations across categorical columns to map linguistic parameters into highly structured numerical vectors.
3. **Model Selection:** Evaluated baseline estimators and ultimately selected an ensemble **Random Forest Regressor** to robustly combat high dimensionality and target variance without overfitting.

---

##  Tech Stack & Dependencies
* **Language:** Python 3.10+
* **Data Processing:** Pandas, NumPy
* **Machine Learning Framework:** Scikit-Learn
* **Data Visualization:** Seaborn, Matplotlib

---

## 💻 Getting Started

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_GITHUB_USERNAME/House-Price-Regression-Analysis.git](https://github.com/sohaibmalik2233/house-prices-advanced-regression)
cd House-Price-Regression-Analysis
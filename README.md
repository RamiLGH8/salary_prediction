# 💼 Salary Prediction using Linear Regression

This project builds a machine learning model to predict salaries for machine learning jobs based on features like job title, education level, and experience level. The goal is to understand the factors influencing salary and build a simple regression model from scratch (and using `scikit-learn`).

---

## 📊 Dataset

- The dataset contains salary information from **machine learning-related jobs** from **2020 to 2025**.
- Total samples: ~145,000+
- Columns used in this project:
  - `job_title` (416 categories)
  - `experience_level` (`EN`, `MI`, `SE`, `EX`)
  - `education_level` (`High School`, `Bachelor`, `Master`, `PhD`)
  - `salary_in_usd` (Target)

---

## 🛠️ Features Engineering

- **Job Title**: Encoded using **mean target encoding** (average salary per job title)
- **Experience Level**: Ordinal encoding (`EN`=0, `MI`=1, `SE`=2, `EX`=3)
- **Education Level**: Ordinal encoding (`High School`=0, `Bachelor`=1, `Master`=2, `PhD`=3)
- **Years of Experience**: Scaled using **standardization (Z-score)**

---

## 🧠 Models Used

### ✅ Custom Linear Regression (from scratch)
- Gradient Descent optimizer
- Early stopping based on cost improvement
- Cost function: Mean Squared Error (MSE)

### ✅ Scikit-learn Linear Regression
- For validation and comparison

---

## 🔍 Evaluation Metrics

- **MAE** (Mean Absolute Error)
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **R² Score** (Coefficient of Determination)
- **Custom Accuracy**: Percentage of predictions within ±$5,000

---


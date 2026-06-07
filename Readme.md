# 🤖 Machine Learning Income Prediction Dashboard

An interactive machine learning project that compares multiple classification models to predict whether an individual earns more than $50K/year using the UCI Adult dataset.

The project includes a full ML pipeline and a Streamlit-based interactive dashboard with model evaluation, visualization, and automated PDF reporting.

---

## 🚀 Project Features

- End-to-end ML pipeline (data preprocessing → training → evaluation)
- Comparison of multiple models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
- Interactive Streamlit dashboard
- Confusion matrix visualization
- Feature importance analysis
- Classification metrics (Precision, Recall, F1-score)
- Automated PDF report generation

---

## 📊 Dashboard Preview

The Streamlit app allows users to:
- Compare model performance visually
- Select models dynamically
- Analyze confusion matrix results
- Understand feature importance
- Generate downloadable PDF reports

---

## 🧠 Key Insight

Although all models perform similarly, Gradient Boosting provides the best overall accuracy. However, evaluation metrics such as recall and precision are essential to understand performance on imbalanced data.

---

## 🛠️ Tech Stack

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- ReportLab

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
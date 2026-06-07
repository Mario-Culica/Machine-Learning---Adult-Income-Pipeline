import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import datetime
from model import load_and_train

# PAGE CONFIG

st.set_page_config(
    page_title="ML Dashboard",
    page_icon="🤖",
    layout="wide"
)
st.title("📊 Machine Learning Dashboard - Adult Income Prediction")
st.caption("Comparison of ML models using UCI Adult dataset")

# LOAD DATA

st.write("Training models... ⏳")
results, models, X_test, y_test = load_and_train()
st.success("Models trained successfully!")
st.divider()

# MODEL COMPARISON
st.header("📊 Model Comparison")
col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots()
    ax.bar(results.keys(), results.values())
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Performance")
    plt.xticks(rotation=30)
    st.pyplot(fig)

with col2:
    best_model_name = max(results, key=results.get)
    st.subheader("🏆 Best Model")
    st.write(f"**{best_model_name}**")
    st.write(f"Accuracy: **{results[best_model_name]:.4f}**")

st.divider()

# MODEL SELECTOR

st.header("🧠 Model Analysis")
model_name = st.selectbox("Choose a model", list(models.keys()))
model = models[model_name]
y_pred = model.predict(X_test)

# CONFUSION MATRIX

st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig2, ax2 = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2)
st.pyplot(fig2)

# CLASSIFICATION REPORT

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))
st.divider()

# FEATURE IMPORTANCE

st.header("📈 Feature Importance")

if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    fig3, ax3 = plt.subplots()
    ax3.barh(range(len(importances)), importances)
    ax3.set_title("Feature Importance")
    st.pyplot(fig3)
else:
    st.info("This model does not support feature importance.")
st.divider()

# INSIGHTS

st.header("💡 Business Insight")
st.info(f"""Best performing model: **{best_model_name}**
Although accuracy is similar across models, Gradient Boosting tends to handle
complex relationships better in structured tabular data like this dataset.
Confusion matrix should be analyzed to understand class imbalance effects.""")


# PDF REPORT

st.header("📄 Export Report")
def generate_pdf(best_model, results):
    filename = "ml_report.pdf"
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()
    content = []
    content.append(Paragraph("ML Model Comparison Report", styles["Title"]))
    content.append(Spacer(1, 12))
    content.append(Paragraph(f"Generated: {datetime.datetime.now()}", styles["Normal"]))
    content.append(Spacer(1, 12))
    content.append(Paragraph("Model Results:", styles["Heading2"]))
    for model, acc in results.items():
        content.append(Paragraph(f"{model}: {acc:.4f}", styles["Normal"]))
    content.append(Spacer(1, 12))
    content.append(Paragraph(f"Best Model: {best_model}", styles["Heading2"]))
    doc.build(content)
    return filename

if st.button("📄 Generate PDF Report"):
    file = generate_pdf(best_model_name, results)
    st.success(f"PDF generated: {file}")

st.divider()

# FOOTER

st.write("Built using Python, Scikit-learn & Streamlit")
st.write("Developed by Mario as a machine learning and data visualization project.")
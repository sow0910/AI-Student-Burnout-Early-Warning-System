import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import tempfile

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="AI-Powered Student Early Warning System",
    layout="wide"
)

st.title("🎓 AI-Powered Student Burnout & Dropout Early Warning System")
st.markdown("Early detection system for identifying at-risk students using Machine Learning.")

# -------------------------------------------------
# Load Data
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("student_features_v2.csv")

df = load_data()

# -------------------------------------------------
# Load Models
# -------------------------------------------------
@st.cache_resource
def load_models():
    burnout_model = joblib.load("burnout_pipeline.pkl")
    dropout_model = joblib.load("dropout_pipeline.pkl")
    return burnout_model, dropout_model

# Try loading models, else prompt for upload
try:
    burnout_model, dropout_model = load_models()
    models_loaded = True
except Exception:
    models_loaded = False

if not models_loaded:
    st.error("❌ Model files not found. Please upload 'burnout_pipeline.pkl' and 'dropout_pipeline.pkl' to use the dashboard features.")
    st.info("Upload your model files below to enable predictions:")

    burnout_file = st.file_uploader("Upload burnout_pipeline.pkl", type="pkl")
    dropout_file = st.file_uploader("Upload dropout_pipeline.pkl", type="pkl")

    if burnout_file and dropout_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_burnout:
            tmp_burnout.write(burnout_file.read())
            burnout_model = joblib.load(tmp_burnout.name)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_dropout:
            tmp_dropout.write(dropout_file.read())
            dropout_model = joblib.load(tmp_dropout.name)
        st.success("✅ Models loaded! You can now use the dashboard features.")
        models_loaded = True
    else:
        st.stop()

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.header("Select Student")

student_id = st.sidebar.selectbox("Student ID", df["student_id"].unique())

student_data = df[df["student_id"] == student_id]

# -------------------------------------------------
# Prepare Features
# -------------------------------------------------
features = student_data.drop(columns=[
    "student_id",
    "name",
    "burnout_risk",
    "burnout_risk_label",
    "dropout_flag",
    "dropout_probability",
    "risk_score",
    "recommended_action"
], errors="ignore")

# -------------------------------------------------
# Predictions
# -------------------------------------------------
burnout_pred = burnout_model.predict(features)[0]
burnout_prob = burnout_model.predict_proba(features)[0][1]

dropout_pred = dropout_model.predict(features)[0]
dropout_prob = dropout_model.predict_proba(features)[0][1]

# -------------------------------------------------
# Risk Score Logic
# -------------------------------------------------
risk_score = round((burnout_prob * 0.6 + dropout_prob * 0.4) * 100, 2)

if risk_score > 75:
    risk_level = "🔴 High Risk"
    recommendation = "Immediate counseling and academic intervention required."
elif risk_score > 50:
    risk_level = "🟠 Moderate Risk"
    recommendation = "Monitor student closely and provide mentoring support."
else:
    risk_level = "🟢 Low Risk"
    recommendation = "Student is stable. Continue regular monitoring."

# -------------------------------------------------
# Display Results
# -------------------------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Burnout Probability", f"{round(burnout_prob*100,2)}%")
col2.metric("Dropout Probability", f"{round(dropout_prob*100,2)}%")
col3.metric("Overall Risk Score", f"{risk_score}%")

st.markdown("---")

st.subheader("Risk Level")
st.success(risk_level)

st.subheader("Recommended Action")
st.info(recommendation)

# -------------------------------------------------
# Visualization
# -------------------------------------------------
st.subheader("Risk Visualization")

col_left, col_center, col_right = st.columns([1,2,1])

with col_center:
    fig, ax = plt.subplots(figsize=(4,3))  # smaller size
    
    bars = ax.bar(
        ["Burnout", "Dropout"],
        [burnout_prob*100, dropout_prob*100]
    )

    ax.set_ylim(0, 100)
    ax.set_ylabel("Probability (%)")

    # Add percentage labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + 2,
            f"{round(height,1)}%",
            ha='center'
        )

    st.pyplot(fig, use_container_width=False)
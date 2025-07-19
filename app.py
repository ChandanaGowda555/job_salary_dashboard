import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="AI Job Salary Dashboard", layout="wide")

# ------------------------------
# Load and preprocess dataset
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("ai_job_dataset.csv")  # Use relative path for GitHub/Streamlit Cloud
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

    exp_map = {'Entry': 0, 'Mid': 1, 'Senior': 2, 'Executive': 3}
    df = df[df['experience_level'].isin(exp_map.keys())]
    df['experience_encoded'] = df['experience_level'].map(exp_map)

    df = df.dropna(subset=['experience_encoded', 'remote_ratio', 'benefits_score', 'salary_usd'])

    return df, exp_map

df, exp_map = load_data()

# ------------------------------
# Sidebar Filters
# ------------------------------
st.sidebar.title("🔍 Filter Job Data")

if not df.empty:
    job_title = st.sidebar.selectbox("Job Title", sorted(df['job_title'].dropna().unique()))
    location = st.sidebar.selectbox("Company Location", sorted(df['company_location'].dropna().unique()))
    experience = st.sidebar.selectbox("Experience Level", sorted(df['experience_level'].dropna().unique()))

    filtered_df = df[
        (df['job_title'] == job_title) &
        (df['company_location'] == location) &
        (df['experience_level'] == experience)
    ]
else:
    st.sidebar.warning("⚠️ Dataset is empty or failed to load.")
    st.stop()

# ------------------------------
# Dashboard Main
# ------------------------------
st.title("🌐 Global AI Job Market & Salary Trends (2025)")
st.markdown("Gain insights into salary ranges, remote work flexibility, and benefits trends in the AI job market.")

# Salary Distribution Plot
st.subheader("💰 Salary Distribution (USD)")
fig1 = px.histogram(filtered_df, x='salary_usd', nbins=20, title="Salary Distribution")
st.plotly_chart(fig1, use_container_width=True)

# Remote Work
st.subheader("🏠 Remote Work Ratio")
fig2 = px.histogram(filtered_df, x='remote_ratio', title="Remote Ratio Distribution")
st.plotly_chart(fig2, use_container_width=True)

# Benefits Score
st.subheader("🎁 Benefits Score Distribution")
fig3 = px.histogram(filtered_df, x='benefits_score', nbins=20, title="Benefits Score")
st.plotly_chart(fig3, use_container_width=True)

# ------------------------------
# Train Salary Prediction Model
# ------------------------------
features = ['experience_encoded', 'remote_ratio', 'benefits_score']
target = df['salary_usd']

st.markdown("---")
st.subheader("🔧 Model Training Status")

if df[features].empty or df.shape[0] < 10:
    st.error("🚫 Not enough data to train the model. Please check the dataset or filter selections.")
    st.stop()
else:
    try:
        X_train, X_test, y_train, y_test = train_test_split(df[features], target, test_size=0.2, random_state=42)
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        mae = mean_absolute_error(y_test, model.predict(X_test))
        st.success("✅ Model trained successfully!")
        st.caption(f"Model Mean Absolute Error (MAE): ${mae:,.0f}")
    except Exception as e:
        st.error(f"🚨 Model training failed: {e}")
        st.stop()

# ------------------------------
# Salary Prediction UI
# ------------------------------
st.title("📊 AI Job Salary Estimator")
st.markdown("Enter your details below to estimate your expected salary:")

col1, col2, col3 = st.columns(3)

with col1:
    exp_level = st.selectbox("Experience Level", list(exp_map.keys()))
with col2:
    remote_ratio = st.slider("Remote Work Ratio (%)", min_value=0, max_value=100, step=10, value=50)
with col3:
    benefits_score = st.slider("Benefits Score (0.00 - 1.00)", min_value=0.0, max_value=1.0, step=0.05, value=0.5)

input_df = pd.DataFrame([{
    'experience_encoded': exp_map[exp_level],
    'remote_ratio': remote_ratio,
    'benefits_score': benefits_score
}])

if st.button("💼 Predict Salary"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"🧾 Estimated Salary: **${prediction:,.2f} USD**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown("🚀 Built with ❤️ using Streamlit | 📊 Dataset: AI Job Market 2025")

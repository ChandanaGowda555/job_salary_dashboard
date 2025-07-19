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
    # MODIFICATION 1: Use relative path for the dataset.
    # Assumes ai_job_dataset.csv is in the same directory as this script.
    # If it's in a 'data' subfolder, use "data/ai_job_dataset.csv"
    try:
        df = pd.read_csv("ai_job_dataset.csv")
    except FileNotFoundError:
        st.error("Error: 'ai_job_dataset.csv' not found. Please ensure it's in the same directory as the app.")
        st.stop() # Stop the app if data can't be loaded

    # MODIFICATION 2: Uncomment and ensure column standardization if needed.
    # This line ensures consistency in column names, e.g., 'Job Title' -> 'job_title'
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

    exp_map = {'entry': 0, 'mid': 1, 'senior': 2, 'executive': 3} # Map lowercase experience levels
    # Ensure 'experience_level' column is also lowercased before mapping
    df['experience_level'] = df['experience_level'].str.lower()
    df = df[df['experience_level'].isin(exp_map.keys())]
    df['experience_encoded'] = df['experience_level'].map(exp_map)

    # Ensure required columns exist before dropping NaNs
    required_cols = ['experience_encoded', 'remote_ratio', 'benefits_score', 'salary_usd', 'job_title', 'company_location', 'experience_level']
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Error: Required column '{col}' not found in the dataset. Please check your CSV file.")
            st.stop()

    df = df.dropna(subset=['experience_encoded', 'remote_ratio', 'benefits_score', 'salary_usd', 'job_title', 'company_location', 'experience_level'])

    if df.empty:
        st.error("Error: Dataset is empty after preprocessing. Check your CSV content or preprocessing steps.")
        st.stop()

    return df, exp_map

df, exp_map = load_data()

# ------------------------------
# Sidebar Filters
# ------------------------------
st.sidebar.title("🔍 Filter Job Data")

# Ensure unique values for selectboxes are available
if not df.empty:
    job_titles = sorted(df['job_title'].dropna().unique())
    locations = sorted(df['company_location'].dropna().unique())
    experiences = sorted(df['experience_level'].dropna().unique(), key=lambda x: exp_map.get(x, 99)) # Sort by encoded value

    job_title = st.sidebar.selectbox("Job Title", job_titles)
    location = st.sidebar.selectbox("Company Location", locations)
    experience = st.sidebar.selectbox("Experience Level", experiences)

    filtered_df = df[
        (df['job_title'] == job_title) &
        (df['company_location'] == location) &
        (df['experience_level'] == experience)
    ]
else:
    st.sidebar.warning("⚠️ Dataset is empty or failed to load. Cannot apply filters.")
    st.stop()


# ------------------------------
# Dashboard Main
# ------------------------------
st.title("🌐 Global AI Job Market & Salary Trends (2025)")
st.markdown("Gain insights into salary ranges, remote work flexibility, and benefits trends in the AI job market.")

# MODIFICATION 3: Check for empty filtered_df before plotting
if filtered_df.empty:
    st.warning("⚠️ No data matches the selected filters. Please adjust your selections to see the dashboard insights.")
else:
    # Salary Distribution Plot
    st.subheader("💰 Salary Distribution (USD)")
    fig1 = px.histogram(filtered_df, x='salary_usd', nbins=20, title="Salary Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    # Remote Work
    st.subheader("🏠 Remote Work Ratio")
    # Ensure remote_ratio is treated as a categorical or binned value for this plot
    # If it's 0, 50, 100, a bar chart might be more appropriate, or bin it.
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
target_col = 'salary_usd'

st.markdown("---")
st.subheader("🔧 Model Training Status")

# MODIFICATION 4: Cache the model training
@st.cache_resource
def train_salary_model(data, features, target_column):
    if data[features].empty or data.shape[0] < 10:
        st.error("🚫 Not enough data to train the model. Please check the dataset or filter selections.")
        return None # Return None if training is not possible

    try:
        # Using the full 'df' for model training for better generalization
        X_train, X_test, y_train, y_test = train_test_split(data[features], data[target_column], test_size=0.2, random_state=42)
        model = RandomForestRegressor(random_state=42) # MODIFICATION 5: Add random_state for reproducibility
        model.fit(X_train, y_train)
        mae = mean_absolute_error(y_test, model.predict(X_test))
        st.success("✅ Model trained successfully!")
        st.caption(f"Model Mean Absolute Error (MAE): ${mae:,.0f}")
        return model
    except Exception as e:
        st.error(f"🚨 Model training failed: {e}")
        return None # Return None if training failed

model = train_salary_model(df, features, target_col) # Train model using the full dataset

# ------------------------------
# Salary Prediction UI
# ------------------------------
st.title("📊 AI Job Salary Estimator")
st.markdown("Enter your details below to estimate your expected salary:")

if model is None:
    st.warning("⚠️ Salary prediction model is not available as training failed or there was insufficient data.")
else:
    col1, col2, col3 = st.columns(3)

    with col1:
        # Use keys from exp_map for display, which are already lowercased
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

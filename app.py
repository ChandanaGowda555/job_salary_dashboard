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
    try:
        df = pd.read_csv("ai_job_dataset.csv")
    except FileNotFoundError:
        st.error("Error: 'ai_job_dataset.csv' not found. Please ensure it's in the same directory as the app.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        st.stop()

    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

    expected_cols = ['job_title', 'company_location', 'experience_level',
                     'remote_ratio', 'benefits_score', 'salary_usd']

    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Error: Missing essential columns in the dataset: {', '.join(missing_cols)}. Please check your CSV headers.")
        st.stop()

    exp_map = {'en': 0, 'mi': 1, 'se': 2, 'ex': 3}

    if 'experience_level' in df.columns:
        df['experience_level'] = df['experience_level'].astype(str).str.lower()
        
        initial_rows = df.shape[0]
        df = df[df['experience_level'].isin(exp_map.keys())].copy()
        
        if df.empty:
            st.error("Dataset became empty after filtering for valid 'experience_level' values. This indicates a mismatch between expected and actual values.")
            st.stop()

        df['experience_encoded'] = df['experience_level'].map(exp_map)
    else:
        st.error("Error: 'experience_level' column not found, even after standardizing names.")
        st.stop()

    required_for_app = ['experience_encoded', 'remote_ratio', 'benefits_score', 'salary_usd',
                        'job_title', 'company_location', 'experience_level']

    initial_rows_before_dropna = df.shape[0]
    df_cleaned = df.dropna(subset=required_for_app)
    rows_dropped_by_dropna = initial_rows_before_dropna - df_cleaned.shape[0]

    if df_cleaned.empty:
        st.error(f"Dataset became empty after dropping rows with missing values in {required_for_app}. This suggests too many essential values are missing.")
        st.stop()
    
    df = df_cleaned

    return df, exp_map

df, exp_map = load_data()

# Define a map for display names (encoded value -> full name)
display_exp_map = {0: 'Entry', 1: 'Mid', 2: 'Senior', 3: 'Executive'}

# ------------------------------
# Sidebar Filters
# ------------------------------
st.sidebar.title("🔍 Filter Job Data")

if not df.empty:
    job_titles = sorted(df['job_title'].dropna().unique())
    locations = sorted(df['company_location'].dropna().unique())
    
    # Get unique abbreviations from the dataframe, sort them by their encoded value
    available_exp_abbr_sorted = sorted(df['experience_level'].dropna().unique(), key=lambda abbr: exp_map[abbr])

    # Create a list of tuples (full_name, abbreviation) for the selectbox
    sidebar_exp_options = [
        (display_exp_map[exp_map[abbr]], abbr)
        for abbr in available_exp_abbr_sorted
    ]
    
    job_title = st.sidebar.selectbox("Job Title", job_titles)
    location = st.sidebar.selectbox("Company Location", locations)
    
    # Use the display name for the selectbox
    selected_exp_display_sidebar = st.sidebar.selectbox(
        "Experience Level", 
        [option[0] for option in sidebar_exp_options] 
    )
    # Find the corresponding abbreviation for filtering
    experience = next((abbr for display_name, abbr in sidebar_exp_options if display_name == selected_exp_display_sidebar), None)


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

if filtered_df.empty:
    st.warning("⚠️ No data matches the selected filters. Please adjust your selections to see the dashboard insights.")
else:
    st.subheader("💰 Salary Distribution (USD)")
    fig1 = px.histogram(filtered_df, x='salary_usd', nbins=20, title="Salary Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("🏠 Remote Work Ratio")
    fig2 = px.histogram(filtered_df, x='remote_ratio', title="Remote Ratio Distribution")
    st.plotly_chart(fig2, use_container_width=True)

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

@st.cache_resource
def train_salary_model(data, features, target_column):
    if data[features].empty or data.shape[0] < 10:
        st.error("🚫 Not enough data to train the model. Please check the dataset or filter selections.")
        return None

    try:
        X_train, X_test, y_train, y_test = train_test_split(data[features], data[target_column], test_size=0.2, random_state=42)
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        mae = mean_absolute_error(y_test, model.predict(X_test))
        st.success("✅ Model trained successfully!")
        st.caption(f"Model Mean Absolute Error (MAE): ${mae:,.0f}")
        return model
    except Exception as e:
        st.error(f"🚨 Model training failed: {e}")
        return None

model = train_salary_model(df, features, target_col)

# ------------------------------
# Salary Prediction UI
# ------------------------------
st.title("📊 AI Job Salary Estimator")
st.markdown("Enter your details below to estimate your expected salary:")

if model is None:
    st.warning("⚠️ Salary prediction model is not available as training failed or there was insufficient data.")
else:
    col1, col2, col3 = st.columns(3)

    # Use the same sorted options as sidebar for consistency in prediction UI
    prediction_exp_display_options = [option[0] for option in sidebar_exp_options]

    with col1:
        exp_level_selected_display = st.selectbox("Experience Level", prediction_exp_display_options)
        # Find the corresponding abbreviated value for encoding
        selected_abbr_for_prediction = next((abbr for display_name, abbr in sidebar_exp_options if display_name == exp_level_selected_display), None)
        exp_level_encoded = exp_map[selected_abbr_for_prediction] if selected_abbr_for_prediction else None

    with col2:
        remote_ratio = st.slider("Remote Work Ratio (%)", min_value=0, max_value=100, step=10, value=50)
    with col3:
        benefits_score = st.slider("Benefits Score (0.00 - 1.00)", min_value=0.0, max_value=1.0, step=0.05, value=0.5)

    input_df = pd.DataFrame([{
        'experience_encoded': exp_level_encoded,
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

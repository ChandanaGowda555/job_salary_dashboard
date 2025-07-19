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
        # Assumes ai_job_dataset.csv is in the same directory as this script.
        # If it's in a 'data' subfolder, use "data/ai_job_dataset.csv"
        df = pd.read_csv("ai_job_dataset.csv")
        # st.success(f"Successfully loaded 'ai_job_dataset.csv'. Initial rows: {df.shape[0]}") # Debugging removed for cleaner output
    except FileNotFoundError:
        st.error("Error: 'ai_job_dataset.csv' not found. Please ensure it's in the same directory as the app.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        st.stop()

    # Standardize column names early
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
    # st.write(f"Processed columns: {df.columns.tolist()}") # Debugging removed for cleaner output

    # Define the expected columns
    expected_cols = ['job_title', 'company_location', 'experience_level',
                     'remote_ratio', 'benefits_score', 'salary_usd']

    # Check for presence of essential columns BEFORE proceeding
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Error: Missing essential columns in the dataset: {', '.join(missing_cols)}. Please check your CSV headers.")
        st.stop()

    # MODIFICATION: Corrected experience level mapping to use abbreviations
    exp_map = {'en': 0, 'mi': 1, 'se': 2, 'ex': 3} # Mapped to 'Entry', 'Mid', 'Senior', 'Executive' respectively

    # Ensure 'experience_level' column is consistently lowercased before mapping
    if 'experience_level' in df.columns:
        df['experience_level'] = df['experience_level'].astype(str).str.lower()
        # st.write(f"Unique experience levels after lowercasing: {df['experience_level'].unique().tolist()}") # Debugging removed

        initial_rows = df.shape[0]
        df = df[df['experience_level'].isin(exp_map.keys())].copy() # Use .copy() to avoid SettingWithCopyWarning
        # st.write(f"Rows after filtering for valid experience levels: {df.shape[0]} (dropped {initial_rows - df.shape[0]} rows)") # Debugging removed

        if df.empty:
            st.error("Dataset became empty after filtering for valid 'experience_level' values. This indicates a mismatch between expected and actual values.")
            st.stop()

        df['experience_encoded'] = df['experience_level'].map(exp_map)
    else:
        st.error("Error: 'experience_level' column not found, even after standardizing names.")
        st.stop()

    # Columns needed for the application's core functionality (prediction and plotting)
    required_for_app = ['experience_encoded', 'remote_ratio', 'benefits_score', 'salary_usd',
                        'job_title', 'company_location', 'experience_level'] # experience_level needed for sidebar

    initial_rows_before_dropna = df.shape[0]
    df_cleaned = df.dropna(subset=required_for_app)
    rows_dropped_by_dropna = initial_rows_before_dropna - df_cleaned.shape[0]

    # st.write(f"Rows dropped due to NaNs in essential columns: {rows_dropped_by_dropna}") # Debugging removed
    if df_cleaned.empty:
        st.error(f"Dataset became empty after dropping rows with missing values in {required_for_app}. This suggests too many essential values are missing.")
        st.stop()
    
    df = df_cleaned # Use the cleaned dataframe

    # st.success(f"Preprocessing complete. Final dataset size: {df.shape[0]} rows.") # Debugging removed

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
    
    # Correct the mapping for display purposes if you want to show full words
    display_exp_map = {0: 'Entry', 1: 'Mid', 2: 'Senior', 3: 'Executive'}
    # Create a sorted list of full experience level names for the selectbox
    experiences_for_display = sorted([display_exp_map[exp_map[abbr]] for abbr in df['experience_level'].dropna().unique()], key=lambda x: display_exp_map[x])
    
    # Get the original abbreviated unique values to map back for filtering
    # This ensures that even if you have only 'se' and 'mi', the options are only 'Senior' and 'Mid'
    available_exp_abbr = sorted(df['experience_level'].dropna().unique(), key=lambda x: exp_map[x])
    experience_level_options = [
        (display_exp_map[exp_map[abbr]], abbr) # (Display Name, Abbreviation)
        for abbr in available_exp_abbr
    ]
    
    job_title = st.sidebar.selectbox("Job Title", job_titles)
    location = st.sidebar.selectbox("Company Location", locations)
    
    # Use the display name for the selectbox, but store the abbreviation for filtering
    selected_exp_display = st.sidebar.selectbox(
        "Experience Level", 
        [item[0] for item in experience_level_options], # Display the full name
        format_func=lambda x: x # Use the full name as is for display
    )
    # Find the corresponding abbreviation for filtering
    experience = next((item[1] for item in experience_level_options if item[0] == selected_exp_display), None)


    filtered_df = df[
        (df['job_title'] == job_title) &
        (df['company_location'] == location) &
        (df['experience_level'] == experience) # Use the abbreviation for filtering
    ]
else:
    st.sidebar.warning("⚠️ Dataset is empty or failed to load. Cannot apply filters.")
    st.stop()


# ------------------------------
# Dashboard Main
# ------------------------------
st.title("🌐 Global AI Job Market & Salary Trends (2025)")
st.markdown("Gain insights into salary ranges, remote work flexibility, and benefits trends in the AI job market.")

# Check for empty filtered_df before plotting
if filtered_df.empty:
    st.warning("⚠️ No data matches the selected filters. Please adjust your selections to see the dashboard insights.")
else:
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

    with col1:
        # Use full names for display in the prediction UI, map back to abbreviations for prediction
        prediction_exp_map = {'Entry': 'en', 'Mid': 'mi', 'Senior': 'se', 'Executive': 'ex'}
        exp_level_display_options = sorted(list(prediction_exp_map.keys()), key=lambda x: exp_map[prediction_exp_map[x]])
        exp_level_selected_display = st.selectbox("Experience Level", exp_level_display_options)
        # Get the corresponding abbreviated value for encoding
        exp_level_encoded = exp_map[prediction_exp_map[exp_level_selected_display]]

    with col2:
        remote_ratio = st.slider("Remote Work Ratio (%)", min_value=0, max_value=100, step=10, value=50)
    with col3:
        benefits_score = st.slider("Benefits Score (0.00 - 1.00)", min_value=0.0, max_value=1.0, step=0.05, value=0.5)

    input_df = pd.DataFrame([{
        'experience_encoded': exp_level_encoded, # Use the correctly encoded value
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

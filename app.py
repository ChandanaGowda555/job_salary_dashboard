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
                     'remote_ratio', 'benefits_score', 'salary_usd',
                     'required_skills', 'company_name']

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

    # Drop NaNs for columns critical to both filtering and model/insights
    required_for_app = ['experience_encoded', 'remote_ratio', 'benefits_score', 'salary_usd',
                        'job_title', 'company_location', 'experience_level',
                        'required_skills', 'company_name'] 

    initial_rows_before_dropna = df.shape[0]
    df_cleaned = df.dropna(subset=required_for_app)
    
    if df_cleaned.empty:
        st.error(f"Dataset became empty after dropping rows with missing values in {required_for_app}. This suggests too many essential values are missing.")
        st.stop()
    
    df = df_cleaned

    return df, exp_map

df, exp_map = load_data()

# Define a map for display names (encoded value -> full name)
display_exp_map = {0: 'Entry', 1: 'Mid', 2: 'Senior', 3: 'Executive'}
# Create a reverse map for sorting by display name
reverse_display_exp_map = {v: k for k, v in display_exp_map.items()}

# ------------------------------
# Sidebar Filters (remain unchanged)
# ------------------------------
st.sidebar.title("🔍 Filter Job Data")

if not df.empty:
    job_titles = sorted(df['job_title'].dropna().unique())
    locations = sorted(df['company_location'].dropna().unique())
    
    available_exp_abbr_sorted = sorted(df['experience_level'].dropna().unique(), key=lambda abbr: exp_map[abbr])

    sidebar_exp_options = [
        (display_exp_map[exp_map[abbr]], abbr)
        for abbr in available_exp_abbr_sorted
    ]
    
    job_title = st.sidebar.selectbox("Job Title", job_titles)
    location = st.sidebar.selectbox("Company Location", locations)
    
    selected_exp_display_sidebar = st.sidebar.selectbox(
        "Experience Level", 
        [option[0] for option in sidebar_exp_options] 
    )
    experience = next((abbr for display_name, abbr in sidebar_exp_options if display_name == selected_exp_display_sidebar), None)


    # filtered_df for average salary and required skills (specific to all three inputs)
    filtered_df = df[
        (df['job_title'] == job_title) &
        (df['company_location'] == location) &
        (df['experience_level'] == experience) 
    ]

    # graph_df for Salary vs Experience Level (filtered only by job title and company location)
    graph_df = df[
        (df['job_title'] == job_title) &
        (df['company_location'] == location)
    ].copy() # Use .copy() to avoid SettingWithCopyWarning
    
    # Add a display column for experience level in graph_df for better plotting labels
    graph_df['experience_level_display'] = graph_df['experience_encoded'].map(display_exp_map)

else:
    st.sidebar.warning("⚠️ Dataset is empty or failed to load. Cannot apply filters.")
    st.stop()


# ------------------------------
# Dashboard Main - Now showing specific insights and a new graph
# ------------------------------
st.title("💡 AI Job Market Insights")
st.markdown(f"**Based on your selection:** \n**Job Title:** `{job_title}`  \n**Location:** `{location}` \n**Experience Level:** `{selected_exp_display_sidebar}`")

if filtered_df.empty:
    st.warning("⚠️ No data matches the selected filters for your specific experience level. Please adjust your selections or check the 'Salary vs. Experience Level' graph below for broader insights.")
else:
    # 1. Average Salary in USD for the three inputs
    st.subheader("💰 Estimated Average Salary (USD)")
    avg_salary = filtered_df['salary_usd'].mean()
    st.markdown(f"The average salary for a **{selected_exp_display_sidebar} {job_title}** in **{location}** is approximately **${avg_salary:,.2f} USD**.")
    
    st.markdown("---")

    # 2. Required Skills for the three inputs
    st.subheader("🛠️ Key Required Skills for this Specific Role")
    all_skills = filtered_df['required_skills'].dropna().tolist()
    if all_skills:
        skills_list = [skill.strip() for sublist in all_skills for skill in sublist.split(',')]
        skill_counts = pd.Series(skills_list).value_counts()
        
        st.write("Candidates for this specific role typically require the following skills:")
        
        num_skills_to_show = 20
        for skill, count in skill_counts.head(num_skills_to_show).items():
            st.markdown(f"- **{skill}** (found in {count} relevant postings)")
        if len(skill_counts) > num_skills_to_show:
            st.info(f"And {len(skill_counts) - num_skills_to_show} more skills...")
    else:
        st.info("No specific skills listed for this combination.")

    st.markdown("---")

# 3. Salary vs. Experience Level Graph (for all experience levels for chosen job & location)
st.subheader(f"📈 Salary Progression for '{job_title}' in '{location}'")

if graph_df.empty:
    st.warning(f"⚠️ No data available to show salary progression for '{job_title}' in '{location}' across different experience levels. Please check your job title and location selections.")
else:
    # Sort the x-axis by the encoded experience level to ensure correct order
    # Use plotly.express.box for better representation of salary distribution at each level
    fig = px.box(
        graph_df,
        x='experience_level_display',
        y='salary_usd',
        title=f"Salary (USD) vs. Experience Level for {job_title} in {location}",
        labels={'salary_usd': 'Salary (USD)', 'experience_level_display': 'Experience Level'},
        color='experience_level_display', # Color by experience level
        category_orders={"experience_level_display": [display_exp_map[i] for i in sorted(display_exp_map.keys())]} # Ensure correct order
    )
    fig.update_traces(marker_line_width=2, marker_line_color='black')
    fig.update_layout(xaxis_title="Experience Level", yaxis_title="Salary (USD)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")


# ------------------------------
# Train Salary Prediction Model (Remains the same)
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
# Salary Prediction UI (Remains the same)
# ------------------------------
st.title("📊 AI Job Salary Estimator")
st.markdown("Enter your details below to estimate your expected salary:")

if model is None:
    st.warning("⚠️ Salary prediction model is not available as training failed or there was insufficient data.")
else:
    col1, col2, col3 = st.columns(3)

    prediction_exp_display_options = [option[0] for option in sidebar_exp_options]

    with col1:
        exp_level_selected_display = st.selectbox("Experience Level", prediction_exp_display_options)
        selected_abbr_for_prediction = next((abbr for display_name, abbr in sidebar_exp_options if display_name == exp_level_selected_display), None)
        exp_level_encoded = exp_map[selected_abbr_for_prediction] if selected_abbr_for_prediction else None

    with col2:
        remote_ratio = st.slider("Remote Work Ratio (%)", min_value=0, max_value=100, step=10, value=50)
        st.info("This indicates how much of your work can be done from home or a remote location. 0% is fully on-site, 50% is hybrid, and 100% is entirely remote.")
    with col3:
        benefits_score = st.slider("Benefits Score (0.00 - 1.00)", min_value=0.0, max_value=1.0, step=0.05, value=0.5)
        st.info("This score (from 0.0 to 1.0) represents the overall quality and comprehensiveness of the job's benefits package. A higher score indicates better benefits.")

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

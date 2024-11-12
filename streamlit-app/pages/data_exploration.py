import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.utils import load_test_data, load_model


# Load test data and model
model_path = '../models/saved_steps.pkl'
_, le_country, le_education = load_model(model_path)

data_path = '../data/survey_results_public_test.csv'
test_data = load_test_data(data_path)

# Page Title
st.title("📊 Data Exploration Dashboard")

if test_data is not None:
    # Decode label encoded columns
    test_data['Country'] = le_country.inverse_transform(test_data['Country'])
    test_data['EdLevel'] = le_education.inverse_transform(test_data['EdLevel'])
    
    # Sidebar for filtering
    st.sidebar.header("🔍 Filters")
    
    selected_country = st.sidebar.selectbox("🌍 Country", options=["All"] + sorted(test_data['Country'].unique()))
    selected_education = st.sidebar.selectbox("🎓 Education Level", options=["All"] + sorted(test_data['EdLevel'].unique()))
    selected_experience = st.sidebar.slider(
        "📅 Years of Experience", 
        int(test_data['YearsCodePro'].min()), 
        int(test_data['YearsCodePro'].max()), 
        (int(test_data['YearsCodePro'].min()), int(test_data['YearsCodePro'].max()))
    )

    # Apply filters
    filtered_data = test_data.copy()
    if selected_country != "All":
        filtered_data = filtered_data[filtered_data['Country'] == selected_country]
    if selected_education != "All":
        filtered_data = filtered_data[filtered_data['EdLevel'] == selected_education]
    filtered_data = filtered_data[
        (filtered_data['YearsCodePro'] >= selected_experience[0]) & 
        (filtered_data['YearsCodePro'] <= selected_experience[1])
    ]

    # Display filtered dataset
    st.write("### Filtered Data", filtered_data.head(10))

    # Salary Distribution
    st.write("### 💰 Salary Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(filtered_data['Salary'], kde=True, color="blue", bins=20, ax=ax)
    ax.set_title("Distribution of Salaries", fontsize=16)
    ax.set_xlabel("Salary", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    st.pyplot(fig)

    # Education Levels Distribution
    st.write("### 🎓 Education Level Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    edu_counts = filtered_data['EdLevel'].value_counts()
    sns.barplot(x=edu_counts.index, y=edu_counts.values, ax=ax, palette="viridis")
    ax.set_title("Distribution of Education Levels", fontsize=16)
    ax.set_xlabel("Education Level", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    plt.xticks(rotation=45)
    st.pyplot(fig)

else:
    st.error("❌ Could not load test data. Please check the `load_test_data` function.")



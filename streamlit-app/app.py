import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.utils import load_model, list_countries, get_data_by_country, get_data_by_education, get_data_by_experience



# Load the model and required data
model_path = '../models/saved_steps.pkl'
regressor_loaded, le_country, le_education = load_model(model_path)

data_path = '../data/survey_results_public.csv'
countries = list_countries(data_path, 400)

education_levels = ['Bachelor’s degree', 'Master’s degree', 'Post grad', 'Less than a Bachelors']

# App Title with Description
st.title("🌍 Salary Prediction Dashboard")
st.markdown(
    """
    Welcome to the **Salary Prediction Dashboard**! Use the form on the left to input your details, and explore insights about salary distributions by country, education level, and experience on the right.
    """
)

# Split the layout into two equal-width columns
col1, col2 = st.columns([1, 2])  # Left: Form, Right: Data Insights

predicted_salary = None  # Initialize predicted salary

# Form Section (Left Column)
with col1:
    st.header("📝 Input Form")
    st.write("Fill out the form below to predict your salary.")

    # Country selection
    country = st.selectbox("🌎 Select Your Country", countries)

    # Experience slider
    experience = st.slider("📅 Years of Professional Experience", 0, 60, 25)
    if experience == 0:
        experience = 0.5
    elif experience > 50:
        experience = 50

    # Education selection
    education = st.selectbox("🎓 Education Level", education_levels)

    # Button to trigger prediction
    if st.button("🚀 Predict Salary"):
        data = pd.DataFrame({
            'Country': [country],
            'EdLevel': [education],
            'YearsCodePro': [experience]
        })
        
        # Encode categorical inputs
        data["EdLevel"] = le_education.transform(data["EdLevel"])
        data["Country"] = le_country.transform(data["Country"])

        # Predict salary
        predicted_salary = regressor_loaded.predict(data)[0]  # Store the predicted salary
        
        # Display the predicted salary
        st.success(f"💰 Predicted Salary: **${predicted_salary:,.2f}**")

# Histograms Section (Right Column)
with col2:
    st.header("📊 Salary Distributions")
    if predicted_salary is not None:
        # Create subplots for histograms
        fig, axes = plt.subplots(3, 1, figsize=(8, 14), tight_layout=True)

        # Education Histogram
        education_data = get_data_by_education(data_path, education)
        sns.histplot(education_data["Salary"], bins=20, color='#1f77b4', ax=axes[0], kde=True, alpha=0.6)
        axes[0].axvline(predicted_salary, color='red', linestyle='--', linewidth=2, label='Predicted Salary')
        axes[0].set_title(f"🎓 Salary Distribution: {education}", fontsize=14)
        axes[0].set_xlabel("Salary", fontsize=12)
        axes[0].set_ylabel("Frequency", fontsize=12)
        axes[0].legend()

        # Experience Histogram
        experience_data = get_data_by_experience(data_path, experience)
        sns.histplot(experience_data["Salary"], bins=20, color='#2ca02c', ax=axes[1], kde=True, alpha=0.6)
        axes[1].axvline(predicted_salary, color='red', linestyle='--', linewidth=2, label='Predicted Salary')
        axes[1].set_title(f"📅 Salary Distribution: {experience} Years Experience", fontsize=14)
        axes[1].set_xlabel("Salary", fontsize=12)
        axes[1].set_ylabel("Frequency", fontsize=12)
        axes[1].legend()

        # Country Histogram
        country_data = get_data_by_country(data_path, country)
        sns.histplot(country_data["Salary"], bins=20, color='#ff7f0e', ax=axes[2], kde=True, alpha=0.6)
        axes[2].axvline(predicted_salary, color='red', linestyle='--', linewidth=2, label='Predicted Salary')
        axes[2].set_title(f"🌎 Salary Distribution: {country}", fontsize=14)
        axes[2].set_xlabel("Salary", fontsize=12)
        axes[2].set_ylabel("Frequency", fontsize=12)
        axes[2].legend()

        # Render the histograms in Streamlit
        st.pyplot(fig)
    else:
        st.info("🔍 No prediction made yet. Use the form to predict a salary.")

# Footer
st.markdown("---")
st.markdown(
    """
    Made with ❤️ using **Streamlit**. Explore salary trends across education, experience, and geography.
    """
)






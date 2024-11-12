import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.utils import load_model, load_test_data


# Load the model and test data
model_path = '../models/saved_steps.pkl'
regressor_loaded, le_country, le_education = load_model(model_path)

data_path = '../data/survey_results_public_test.csv'

# Load the test dataset
test_data = load_test_data(data_path)

# Page Title
st.title("📈 Salary Prediction vs Actual Salary")

# Sidebar Filters
st.sidebar.header("🔍 Filters")

if test_data is not None:
    test_data = test_data[['Country','EdLevel','YearsCodePro','Salary']]
    # Decode country and education columns for filter options
    test_data['Country'] = le_country.inverse_transform(test_data['Country'])
    test_data['EdLevel'] = le_education.inverse_transform(test_data['EdLevel'])

    # Extract unique values for filters
    countries = sorted(test_data['Country'].unique())
    education_levels = sorted(test_data['EdLevel'].unique())
    min_experience, max_experience = int(test_data['YearsCodePro'].min()), int(test_data['YearsCodePro'].max())

    # Filter inputs
    selected_country = st.sidebar.selectbox("🌍 Country", options=["All"] + countries)
    selected_education = st.sidebar.selectbox("🎓 Education Level", options=["All"] + education_levels)
    selected_experience = st.sidebar.slider("📅 Years of Experience", min_experience, max_experience, (min_experience, max_experience))

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

    if not filtered_data.empty:
        # Predict salaries for the filtered data
        transformed_data = filtered_data.copy()
        transformed_data['Country'] = le_country.transform(transformed_data['Country'])
        transformed_data['EdLevel'] = le_education.transform(transformed_data['EdLevel'])
        filtered_data["PredictedSalary"] = regressor_loaded.predict(transformed_data.drop(columns=["Salary"]))

        # Scatter plot creation
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            x="Salary",
            y="PredictedSalary",
            data=filtered_data,
            alpha=0.6,
            ax=ax,
            color="#4C72B0"
        )
        ax.set_title("Actual Salary vs Predicted Salary", fontsize=16)
        ax.set_xlabel("Actual Salary", fontsize=12)
        ax.set_ylabel("Predicted Salary", fontsize=12)
        ax.axline((0, 0), slope=1, color="red", linestyle="--", label="Perfect Prediction")
        ax.legend()

        # Display the scatter plot
        st.pyplot(fig)

        # Display metrics
        mse = ((filtered_data["Salary"] - filtered_data["PredictedSalary"]) ** 2).mean()
        r2 = 1 - (sum((filtered_data["Salary"] - filtered_data["PredictedSalary"]) ** 2) / 
                  sum((filtered_data["Salary"] - filtered_data["Salary"].mean()) ** 2))
        st.write(f"📉 Mean Squared Error: **{mse:,.2f}**")
        st.write(f"🔗 R² Score: **{r2:.2f}**")
    else:
        st.warning("⚠️ No data available for the selected filters.")
else:
    st.error("❌ Could not load test data. Please check the `load_test_data` function.")


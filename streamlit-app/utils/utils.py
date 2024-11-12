import pickle
import pandas as pd
import streamlit as st


@st.cache_resource
def load_model(model_path):
    with open(model_path, 'rb') as file:
        data = pickle.load(file)

    regressor_loaded = data["model"]
    le_country = data["le_country"]
    le_education = data["le_education"]

    return regressor_loaded, le_country, le_education

@st.cache_data
def list_countries(data_path, cutoff=400):
    df = load_and_clean_data(data_path, cutoff)

    return df["Country"].unique().tolist()


def get_data_by_country(data_path, country):
    df = load_and_clean_data(data_path)
    return df[df["Country"] == country]

def get_data_by_experience(data_path, experience):
    df = load_and_clean_data(data_path)
    return df[df["YearsCodePro"] == experience]

def get_data_by_education(data_path, education):
    df = load_and_clean_data(data_path)
    return df[df["EdLevel"] == education]

def load_and_clean_data(data_path, cutoff=400):
    df = pd.read_csv(data_path)
    selected_columns = ["Country", "EdLevel", "YearsCodePro", "Employment", "ConvertedCompYearly"]
    rename = {"ConvertedCompYearly": "Salary"}

    df = df[selected_columns]
    df = df.rename(columns=rename)
    df = df.dropna()

    df = df[df["Employment"] == 'Employed, full-time']

    country_count = df["Country"].value_counts()
    other_countries = country_count[country_count < cutoff].index
    df["Country"] = df["Country"].replace(other_countries, "Other")

    df = df[(df["Salary"] > 10000) & (df["Salary"] < 250000)]

    df['YearsCodePro'] = df['YearsCodePro'].apply(clean_experience)

    df['EdLevel'] = df['EdLevel'].apply(clean_education)

    return df

def clean_education(x):
    if 'Bachelor’s degree' in x:
        return 'Bachelor’s degree'
    if 'Master’s degree' in x:
        return 'Master’s degree'
    if 'Professional degree' in x or 'Other doctoral' in x:
        return 'Post grad'
    return 'Less than a Bachelors'

def clean_experience(x):
    if x ==  'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)

def load_test_data(data_path):
    return pd.read_csv(data_path)


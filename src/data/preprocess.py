import pandas as pd
import re

def normalize_company_name(name):
    if isinstance(name, str):
        # Convert to lowercase
        name = name.lower()
        # Remove non-alphanumeric characters
        name = re.sub(r'[^\w\s]', '', name)
    return name

def clean_data(df):
    df['original_company_name'] = df['original_company_name'].apply(normalize_company_name)
    df['retrieved_company_name'] = df['retrieved_company_name'].apply(normalize_company_name)
    return df

def create_similarity_label(df):
    #df['is_similar'] = (df['original_company_name'] == df['retrieved_company_name']).astype(int)
    df['is_similar'] = 1
    return df
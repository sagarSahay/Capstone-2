import pandas as pd

def clean_data(df):
    df['original_company_name'] = df['original_company_name'].str.lower().str.replace('[^\w\s]', '')
    df['retrieved_company_name'] = df['retrieved_company_name'].str.lower().str.replace('[^\w\s]', '')
    return df

def create_similarity_label(df):
    # Example heuristic: exact match
    df['is_similar'] = (df['original_company_name'] == df['retrieved_company_name']).astype(int)
    return df
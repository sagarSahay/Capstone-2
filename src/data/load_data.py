import pandas as pd

def load_data(file_path: object) -> object:
    return pd.read_excel(file_path)
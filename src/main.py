import pandas as pd
import openpyxl
from sklearn.model_selection import train_test_split

from src.data.load_data import load_data
from src.data.preprocess import clean_data, create_similarity_label
from src.features.build_features import build_features
from src.models.train_model import train_model
from src.models.evaluate_model import evaluate_model
from src.utils.utils import save_model, load_model

def main():
    spain_df = load_data('../data/raw/spain_companies.xlsx')
    germany_df = load_data('../data/raw/germany_companies.xlsx')
    mixed_df = load_data('../data/raw/mixed_companies.xlsx')

    data = pd.concat([spain_df,germany_df,mixed_df])
    data = clean_data(data)

    data = create_similarity_label(data)

    X,y = build_features(data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    accuracy, report = evaluate_model(model, X_test, y_test)
    print(f'Accuracy: {accuracy}')
    print(report)

    # Save the model
    save_model(model, '../models/model1/logistic_regression_model')


if __name__ == '__main__':
    main()
import pandas as pd
import openpyxl
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from src.data.load_data import load_data
from src.data.preprocess import clean_data, create_similarity_label
from src.features.build_features import build_features
from src.models.train_model import train_model, train_random_forest_model
from src.models.evaluate_model import evaluate_model
from src.utils.utils import save_model, load_model


def generate_non_similar_pairs(data):
    # Generate non-similar pairs by shuffling the retrieved company names
    non_similar_df = data.copy()
    non_similar_df['retrieved_company_name'] = shuffle(non_similar_df['retrieved_company_name'].values)
    non_similar_df['is_similar'] = 0

    # Ensure no accidentally created similar pairs
    non_similar_df = non_similar_df[non_similar_df['original_company_name'] != non_similar_df['retrieved_company_name']]

    return non_similar_df
def main():
    # Load and preprocess data
    spain_df = load_data('../data/raw/spain_companies.xlsx')
    germany_df = load_data('../data/raw/germany_companies.xlsx')
    mixed_df = load_data('../data/raw/mixed_companies.xlsx')

    data = pd.concat([spain_df, germany_df, mixed_df])
    data = clean_data(data)
    data = create_similarity_label(data)

    # Generate non-similar pairs
    non_similar_df = generate_non_similar_pairs(data)

    # Combine similar and non-similar pairs
    balanced_df = pd.concat([data, non_similar_df], ignore_index=True).sample(frac=1).reset_index(drop=True)

    # Build features
    X, y = build_features(balanced_df)

    X_df = pd.DataFrame(X)
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    rf_model = train_random_forest_model(X_train, y_train)

    # Evaluate Random Forest model
    rf_accuracy, rf_report = evaluate_model(rf_model, X_test, y_test)
    print(f'Random Forest Accuracy: {rf_accuracy}')
    print(rf_report)

    # Save the Random Forest model
    save_model(rf_model, '../models/random_forest_model.pkl')

    # Print sample output
    y_pred = rf_model.predict(X_test.values)
    sample_output = pd.DataFrame({
        'Original Company Name': data.iloc[X_test.index]['original_company_name'].values,
        'Retrieved Company Name': data.iloc[X_test.index]['retrieved_company_name'].values,
        'Country Code': data.iloc[X_test.index]['country_code'].values,
        'Actual Similarity': y_test,
        'Predicted Similarity': y_pred
    })

    # Show first 10 rows of the sample output
    print("\nSample Output:\n")
    print(sample_output.head(10))

if __name__ == '__main__':
    main()
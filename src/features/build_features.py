import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


def build_features(data):
    # Feature: simple equivalence (1 if names are exactly the same, 0 otherwise)
    data['name_equivalence'] = (data['original_company_name'] == data['retrieved_company_name']).astype(int)

    # Encode country code
    label_encoder = LabelEncoder()
    data['country_code_encoded'] = label_encoder.fit_transform(data['country_code'])

    # TF-IDF features
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
    original_tfidf = vectorizer.fit_transform(data['original_company_name'])
    retrieved_tfidf = vectorizer.transform(data['retrieved_company_name'])

    # Normalize TF-IDF vectors
    from sklearn.preprocessing import normalize
    original_tfidf_normalized = normalize(original_tfidf)
    retrieved_tfidf_normalized = normalize(retrieved_tfidf)

    # Combine all features
    X_tfidf = np.hstack((original_tfidf_normalized.toarray(), retrieved_tfidf_normalized.toarray()))
    X_additional = data[['name_equivalence', 'country_code_encoded']].values
    X = np.hstack((X_tfidf, X_additional))

    y = data['is_similar'].values

    return X, y
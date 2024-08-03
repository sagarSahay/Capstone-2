import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

def build_features(data):
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4))
    original_tfidf = vectorizer.fit_transform(data['original_company_name'])
    retrieved_tfidf = vectorizer.transform(data['retrieved_company_name'])

    X = np.hstack((original_tfidf.toarray(), retrieved_tfidf.toarray()))

    label_encoder = LabelEncoder()
    data['country_code_encoded'] = label_encoder.fit_transform(data['country_code'])

    X = np.hstack((X, data['country_code_encoded'].values.reshape(-1,1)))

    y = data['is_similar']

    return X,y
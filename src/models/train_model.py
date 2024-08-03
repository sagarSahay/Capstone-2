from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

def train_model(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_resampled , y_resampled = smote.fit_resample(X_train, y_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_resampled, y_resampled)

    return model
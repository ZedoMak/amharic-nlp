import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("data/preprocessed_train.csv")

X = df["clean_tweet"]
y = df["sentiment"]

# Split
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Vectorize
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_valid_vec = vectorizer.transform(X_valid)

# Train model
model = LogisticRegression(max_iter=1000)

model.fit(X_train_vec, y_train)

# Predict
preds = model.predict(X_valid_vec)

# Evaluate
print("Accuracy:")
print(accuracy_score(y_valid, preds))

print("\nClassification Report:")
print(classification_report(y_valid, preds))

# Save model
joblib.dump(model, "models/logistic_model.pkl")
joblib.dump(vectorizer, "models/tfidf.pkl")

print("\nModel saved.")
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/preprocessed_train.csv")

X = df["clean_tweet"]
y = df["sentiment"]

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2
)

X_train_vec = vectorizer.fit_transform(X_train)
X_valid_vec = vectorizer.transform(X_valid)

models = {
    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),

    "Naive Bayes":
        MultinomialNB(),

    "Linear SVM":
        LinearSVC(
            class_weight="balanced"
        )
}

for name, model in models.items():

    model.fit(X_train_vec, y_train)

    preds = model.predict(X_valid_vec)

    acc = accuracy_score(
        y_valid,
        preds
    )

    print(f"{name}: {acc:.4f}")
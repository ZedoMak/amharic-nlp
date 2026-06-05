import joblib
from preprocess import clean_text

model = joblib.load("models/logistic_model.pkl")
vectorizer = joblib.load("models/tfidf.pkl")

while True:
    text = input("\nEnter text: ")

    if text.lower() == "quit":
        break

    cleaned = clean_text(text)

    vec = vectorizer.transform([cleaned])

    prediction = model.predict(vec)[0]

    print("Prediction:", prediction)


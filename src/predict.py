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
    probs = model.predict_proba(vec)[0]
    confidence = probs.max()

    prediction = model.predict(vec)[0]

    print("Prediction:", prediction)
    print(f"Confidence: {confidence:.2%}")

    


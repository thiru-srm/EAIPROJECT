"""Train a simple spam detector and save it to spam_model.joblib"""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load data (works with our file, or the bigger UCI "spam.csv" if you add it)
try:
    df = pd.read_csv("data/spam.csv", encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "text"]
    print("Using the big UCI dataset: data/spam.csv")
except FileNotFoundError:
    df = pd.read_csv("data/messages.csv")
    print("Using the small sample dataset: data/messages.csv")

# 2. Split into training and testing data (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# 3. Build the model: text -> numbers (TF-IDF) -> Naive Bayes classifier
model = make_pipeline(
    TfidfVectorizer(lowercase=True, stop_words="english"),
    MultinomialNB(),
)

# 4. Train
model.fit(X_train, y_train)

# 5. Evaluate on unseen messages
pred = model.predict(X_test)
print(f"\nTest accuracy: {accuracy_score(y_test, pred) * 100:.2f}%")
print("\nConfusion matrix (rows = actual, columns = predicted) [ham, spam]:")
print(confusion_matrix(y_test, pred, labels=["ham", "spam"]))
print("\n", classification_report(y_test, pred))

# 6. Retrain on ALL data and save for the app
model.fit(df["text"], df["label"])
joblib.dump(model, "spam_model.joblib")
print("Model saved as spam_model.joblib")

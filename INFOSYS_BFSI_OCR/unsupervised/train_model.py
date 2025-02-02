import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load labeled dataset
data = pd.read_csv(r"C:/Users/Asha Mercy R/OneDrive/Desktop/project/labels.csv")  # Ensure you have a labeled dataset
X = data["Description"]
y = data["Category"]

# Convert text to TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = vectorizer.fit_transform(X)

# Train Model
model = MultinomialNB()
model.fit(X_tfidf, y)

# Save Model and Vectorizer
with open('model.pkl', 'wb') as model_file, open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(model, model_file)
    pickle.dump(vectorizer, vectorizer_file)

print("Model trained and saved successfully!")

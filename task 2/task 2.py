import nltk
import pandas as pd
from datasets import load_dataset
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()


# --- Step 2: Define Preprocessing Function ---
def preprocess_text(text):
    # 1. Lowercase
    text = text.lower()
    # 2. Tokenize (split text into words using NLTK)
    tokens = word_tokenize(text)
    # 3. Clean & Lemmatize (keep only letters/numbers and convert to root form)
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]
    # 4. Join back into a clean string
    return " ".join(cleaned_tokens)


# --- Step 3: Load Dataset ---
ds = load_dataset("MakTek/Customer_support_faqs_dataset")
questions = ds["train"]["question"]
answers = ds["train"]["answer"]

#Apply NLTK Preprocessing to all Dataset Questions ---
cleaned_questions = [preprocess_text(q) for q in questions]

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(cleaned_questions)

print("\n--- FAQ Chatbot Ready! (Type 'exit' to quit) ---")
while True:
    user_query = input("\nHow may we help you? ")

    if user_query.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break

    cleaned_query = preprocess_text(user_query)

    # Vectorize and compare
    query_vector = vectorizer.transform([cleaned_query])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)

    best_match_index = similarity_scores.argmax()
    best_score = similarity_scores.max()

    threshold = 0.2
    if best_score >= threshold:
        print(f"Chatbot: {answers[best_match_index]}")
    else:
        print("Chatbot: I'm sorry, I don't understand that question. Could you rephrase?")
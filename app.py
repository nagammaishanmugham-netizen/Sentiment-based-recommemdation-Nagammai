from flask import Flask, render_template, request
import pickle
import pandas as pd

# -------------------------------------------------
# Create Flask app
# -------------------------------------------------
app = Flask(__name__)

# -------------------------------------------------
# Load trained models
# -------------------------------------------------
sentiment_model = pickle.load(open('pickle_file/final_model.pkl', 'rb'))
count_vectorizer = pickle.load(open('pickle_file/count_vector.pkl', 'rb'))
tfidf_transformer = pickle.load(open('pickle_file/tfidf_transformer.pkl', 'rb'))
user_final_rating = pickle.load(open('pickle_file/user_final_rating.pkl', 'rb'))

# -------------------------------------------------
# Load dataset
# -------------------------------------------------
df = pd.read_csv('sample30.csv')

# -------------------------------------------------
# Recommendation logic
# -------------------------------------------------
def recommend_top_5(username):
    if username not in user_final_rating.index:
        return []

    top_20_products = (
        user_final_rating.loc[username]
        .sort_values(ascending=False)
        .head(20)
        .index
    )

    sentiment_scores = {}

    for product in top_20_products:
        reviews = df[df['name'] == product]['reviews_text']
        if reviews.empty:
            continue

        X_counts = count_vectorizer.transform(reviews)
        X_tfidf = tfidf_transformer.transform(X_counts)
        predictions = sentiment_model.predict(X_tfidf)

        positive_ratio = (predictions == 1).mean()
        sentiment_scores[product] = positive_ratio

    return sorted(
        sentiment_scores,
        key=sentiment_scores.get,
        reverse=True
    )[:5]

# -------------------------------------------------
# Flask route
# -------------------------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    error = None

    if request.method == 'POST':
        username = request.form['username']
        recommendations = recommend_top_5(username)

        if not recommendations:
            error = 'User not found or no recommendations available.'

    return render_template(
        'index.html',
        recommendations=recommendations,
        error=error
    )

# -------------------------------------------------
# Run app (ONLY ONCE, AT THE END)
# -----------------------------------------
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__)

# -----------------------------
# Load models
# -----------------------------
sentiment_model = pickle.load(open('pickle_file/final_model.pkl', 'rb'))
tfidf_pipeline = pickle.load(open('pickle_file/tfidf_pipeline.pkl', 'rb'))
user_final_rating = pickle.load(open('pickle_file/user_final_rating.pkl', 'rb'))

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv('sample30.csv')

# -----------------------------
# Recommendation logic
# -----------------------------
def recommend_top_5(username):
    if username not in user_final_rating.index:
        return []

    user_scores = user_final_rating.loc[username]

    if user_scores.isna().all():
        return []

    top_20_products = (
        user_scores
        .sort_values(ascending=False)
        .head(20)
        .index
        .tolist()
    )

    sentiment_scores = {}

    for product in top_20_products:
        reviews = df.loc[df['name'] == product, 'reviews_text']

        if reviews.empty:
            continue

        try:
            X_tfidf = tfidf_pipeline.transform(reviews.astype(str))
            predictions = sentiment_model.predict(X_tfidf)
            sentiment_scores[product] = (predictions == 1).mean()
        except Exception:
            continue

    if not sentiment_scores:
        return []

    return sorted(
        sentiment_scores,
        key=sentiment_scores.get,
        reverse=True
    )[:5]

# -----------------------------
# Flask route
# -----------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    error = None

    if request.method == 'POST':
        username = request.form.get('username')

        recommendations = recommend_top_5(username)

        if not recommendations:
            error = "No recommendations available for this user."

    return render_template(
        'index.html',
        recommendations=recommendations,
        error=error
    )

popular_products = pickle.load(
    open("pickle_file/popular_products.pkl", "rb")
)


# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

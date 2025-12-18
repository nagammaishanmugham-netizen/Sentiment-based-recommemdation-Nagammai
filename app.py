from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

# -------------------------------------------------
# Create Flask app
# -------------------------------------------------
app = Flask(__name__)

# -------------------------------------------------
# Load trained models (PIPELINE BASED)
# -------------------------------------------------
sentiment_model = pickle.load(open("pickle_file/final_model.pkl", "rb"))
tfidf_pipeline = pickle.load(open("pickle_file/tfidf_pipeline.pkl", "rb"))
user_final_rating = pickle.load(open("pickle_file/user_final_rating.pkl", "rb"))

# -------------------------------------------------
# Load dataset
# -------------------------------------------------
df = pd.read_csv("sample30.csv")

# -------------------------------------------------
# Recommendation logic
# -------------------------------------------------
def recommend_top_5(username):
    try:
        if username not in user_final_rating.index:
            return []

        # Top 20 recommendations from item-based CF
        top_20_products = (
            user_final_rating.loc[username]
            .sort_values(ascending=False)
            .head(20)
            .index
            .tolist()
        )

        sentiment_scores = {}

        for product in top_20_products:
            reviews = df[df["name"] == product]["reviews_text"].dropna()

            if reviews.empty:
                continue

            # TF-IDF PIPELINE (no CountVectorizer separately)
            X_tfidf = tfidf_pipeline.transform(reviews)

            predictions = sentiment_model.predict(X_tfidf)
            sentiment_scores[product] = (predictions == 1).mean()

        # Return top 5 by positive sentiment %
        return sorted(
            sentiment_scores,
            key=sentiment_scores.get,
            reverse=True
        )[:5]

    except Exception as e:
        print("ERROR in recommendation:", e)
        return []

# -------------------------------------------------
# Flask route
# -------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    error = None

    if request.method == "POST":
        username = request.form.get("username")

        recommendations = recommend_top_5(username)

        if not recommendations:
            error = "User not found or no recommendations available."

    return render_template(
        "index.html",
        recommendations=recommendations,
        error=error
    )

# -------------------------------------------------
# Run app (Render-compatible)
# -------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

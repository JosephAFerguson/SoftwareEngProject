from flask import Flask, request, jsonify
from flask_cors import CORS

from services.recommend import recommend_listings

app = Flask(__name__)
CORS(app)  # allow React dev server

@app.route("/api/health")
def health():
    return {"status": "ok"}

@app.route("/api/recommend", methods=["POST"])
def recommend():
    user_preferences = request.json
    recommendations = recommend_listings(user_preferences)
    return jsonify(recommendations)

if __name__ == "__main__":
    app.run(debug=True)

import os
import pickle

import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# Model file must sit next to app.py
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "linear.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Exact column names / order the model was trained with
FEATURES = [
    "Square_Footage", "Num_Bedrooms", "Num_Bathrooms",
    "Year_Built", "Lot_Size", "Garage_Size", "Neighborhood_Quality",
]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        d = request.get_json(force=True)
        values = [
            float(d["square_footage"]), float(d["bedrooms"]), float(d["bathrooms"]),
            float(d["year_built"]), float(d["lot_size"]), float(d["garage_size"]),
            float(d["quality"]),
        ]
        row = pd.DataFrame([values], columns=FEATURES)
        price = float(model.predict(row)[0])
        per_sqft = price / values[0] if values[0] else 0
        return jsonify(ok=True, price=round(price, 2), per_sqft=round(per_sqft, 2))
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)

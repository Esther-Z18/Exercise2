from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)

y = np.array([
    137, 118, 124, 124, 120,
    129, 122, 142, 128, 114,
    132, 130, 130, 112, 132,
    117, 134, 132, 121, 128
])

W = np.array([
    0, 1, 1, 1, 0,
    1, 1, 0, 0, 1,
    1, 0, 0, 1, 0,
    1, 0, 0, 1, 1
])

X_spending = np.array([
    19.8, 23.4, 27.7, 24.6, 21.5,
    25.1, 22.4, 29.3, 20.8, 20.2,
    27.3, 24.5, 22.9, 18.4, 24.2,
    21.0, 25.9, 23.2, 21.6, 22.8
])

X_features = np.column_stack((W, X_spending))

model = LinearRegression().fit(X_features, y)

@app.route("/predict")
def predict():
    try:
        w = float(request.args.get("w", 0))
        x = float(request.args.get("x", 0))
    except ValueError:
        return jsonify({"error"}), 400

    y_pred = model.predict([[w, x]])[0]
    with open("output.txt", "w") as f:
        f.write(f"Input w (treatment): {w}\nInput x (sustainability spending): {x}\nPrediction (engagement score): {y_pred}\n")
    
    return jsonify({"w": w, "x": x, "predicted_engagement_score": y_pred})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

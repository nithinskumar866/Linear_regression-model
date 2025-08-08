# app.py

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
with open("model/lr_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        work_hours = float(request.form["work_hours"])
        break_time = float(request.form["break_time"])
        tasks_completed = float(request.form["tasks_completed"])

        features = np.array([[work_hours, break_time, tasks_completed]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]

        return render_template("index.html", prediction=round(prediction, 2))

    except Exception as e:
        return render_template("index.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True)

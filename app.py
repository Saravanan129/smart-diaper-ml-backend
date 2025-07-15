from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, db
import joblib
import os

# ✅ Initialize Firebase if not already initialized
if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://smartdiaper2-97108-default-rtdb.firebaseio.com"
    })

# ✅ Load ML model
model = joblib.load("uti_risk_model.pkl")

# ✅ Flask App
app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Smart Diaper ML Backend is Running on Render!"

@app.route("/predict", methods=["POST"])
def predict_uti_risk():
    try:
        data = request.json

        features = [
            float(data["moisture"]),
            float(data["gasLevel"]),
            float(data["tempC"]),
            int(data["crying"]),
            int(data["handNearAbdomen"]),
            int(data["urinationFrequency"]),
            float(data["hydrationPercent"]),
        ]

        prediction = int(model.predict([features])[0])
        result = "High" if prediction == 1 else "Low"

        # ✅ Store in Firebase under dailyReports
        date_key = data["date"]
        db.reference(f"/Sensor/dailyReports/{date_key}").update({
            "utiRisk": result
        })

        return jsonify({"utiRisk": result}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
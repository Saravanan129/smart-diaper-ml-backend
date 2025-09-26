from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, db
import joblib
import pandas as pd
import os

# ✅ Initialize Firebase if not already initialized
if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccountKey.json")  # make sure this file is correct
    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://smartdiaper2-97108-default-rtdb.firebaseio.com"
    })

# ✅ Load ML model
model = joblib.load("uti_risk_model.pkl")

# ✅ Flask App
app = Flask(__name__)
CORS(app)  # enable CORS for all requests

@app.route("/")
def home():
    return "✅ Smart Diaper ML Backend is Running on Render!"

@app.route("/predict", methods=["POST"])
def predict_uti_risk():
    try:
        data = request.json
        print("Flask received this:", data)

        # ✅ Convert input to DataFrame with proper feature names
        feature_names = [
            "moisture",
            "gasLevel",
            "tempC",
            "crying",
            "handNearAbdomen",
            "urinationFrequency",
            "hydrationPercent"
        ]
        features_df = pd.DataFrame([data], columns=feature_names)

        # ✅ Predict using ML model
        prediction = int(model.predict(features_df)[0])
        result = "High" if prediction == 1 else "Low"

        # ✅ Store in Firebase under dailyReports
        date_key = data.get("date")
        if date_key:
            db.reference(f"/Sensor/dailyReports/{date_key}").update({
                "utiRisk": result
            })

        return jsonify({"utiRisk": result}), 200

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)




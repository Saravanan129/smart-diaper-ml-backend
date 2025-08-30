from flask import Flask, request, jsonify
from flask_cors import CORS 
import firebase_admin
from firebase_admin import credentials, db
import joblib
import os


if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://smartdiaper2-97108-default-rtdb.firebaseio.com"
    })


model = joblib.load("uti_risk_model.pkl")


app = Flask(__name__)
CORS(app)  

@app.route("/")
def home():
    return 

@app.route("/predict", methods=["POST"])
def predict_uti_risk():
    try:
        data = request.json
        print("Flask received this:", data)

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

        
        date_key = data["date"]
        db.reference(f"/Sensor/dailyReports/{date_key}").update({
            "utiRisk": result
        })

        return jsonify({"utiRisk": result}), 200

    except Exception as e:

        return jsonify({"error": str(e)}), 500

from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, db
import joblib
import os
import json

# ✅ Initialize Firebase if not already initialized
if not firebase_admin._apps:
    try:
        # Get the Firebase config from environment variable
        firebase_config_json = os.environ.get("firebase-config")
        
        if firebase_config_json:
            # Parse the JSON string into a dictionary
            firebase_config = json.loads(firebase_config_json)
            
            # Initialize Firebase with the config
            cred = credentials.Certificate(firebase_config)
            firebase_admin.initialize_app(cred, {
                "databaseURL": "https://smartdiaper2-97108-default-rtdb.firebaseio.com"
            })
            print("✅ Firebase initialized from firebase-config environment variable")
        else:
            print("❌ firebase-config environment variable not found")
            
    except Exception as e:
        print(f"❌ Firebase initialization error: {e}")

# ✅ Load ML model
model = joblib.load("uti_risk_model.pkl")

# ✅ Flask App
app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "✅ Smart Diaper ML Backend is Running on Render!"

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

        # ✅ Store in Firebase under dailyReports
        date_key = data["date"]
        db.reference(f"/Sensor/dailyReports/{date_key}").update({
            "utiRisk": result
        })

        return jsonify({"utiRisk": result}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)








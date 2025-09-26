from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, db
import joblib

# ---------------------------
# Initialize Firebase
# ---------------------------
if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://smartdiaper2-97108-default-rtdb.firebaseio.com"
    })

# ---------------------------
# Load ML model
# ---------------------------
model = joblib.load("uti_risk_model.pkl")

# ---------------------------
# Flask App
# ---------------------------
app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "✅ Smart Diaper ML Backend is Running on Render!"

@app.route("/predict", methods=["POST"])
def predict_uti_risk():
    try:
        data = request.json
        print("Flask received:", data)

        # Convert pH to numeric for the model
        ph_numeric = 1 if str(data.get("ph", "basic")).lower() == "acidic" else 0

        features = [
            float(data["moisture"]),
            float(data["gasLevel"]),
            float(data["tempC"]),
            int(data.get("crying", 0)),
            int(data.get("handNearAbdomen", 0)),
            int(data.get("urinationFrequency", 0)),
            float(data.get("hydrationPercent", 0)),
            ph_numeric
        ]

        prediction = int(model.predict([features])[0])
        result = "High" if prediction == 1 else "Low"

        # Store in Firebase under dailyReports
        date_key = data["date"]
        db.reference(f"/Sensor/dailyReports/{date_key}").update({
            "utiRisk": result
        })

        return jsonify({"utiRisk": result}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)


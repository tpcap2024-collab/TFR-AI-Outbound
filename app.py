from flask import Flask, request

app = Flask(__name__)

@app.route("/")
def home():
    return "TFR AI Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    return {
        "status": "received",
        "data": data
    }

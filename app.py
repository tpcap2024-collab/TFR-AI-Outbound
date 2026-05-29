from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "TFR AI Running"

@app.route("/predict")
def predict():
    return {
        "fill_rate": 78
    }

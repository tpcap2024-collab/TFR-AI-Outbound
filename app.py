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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

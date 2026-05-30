from flask import Flask, request
import requests
import numpy as np
import cv2

app = Flask(__name__)

@app.route("/")
def home():
    return "TFR AI Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json or {}

    img_url = data.get("photo")

    return {
        "status": "ok",
        "photo": img_url
    }

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

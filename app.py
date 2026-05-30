from flask import Flask, request, jsonify
import cv2
import numpy as np
import requests

app = Flask(__name__)

@app.route("/")
def home():
    return "TFR AI Running"

@app.route("/predict", methods=["POST"])
def predict():

    try:

        print("HEADERS:", dict(request.headers))
        print("RAW DATA:", request.data)

        data = request.get_json(silent=True)

        print("JSON:", data)

        if not data:
            return jsonify({
                "status": "error",
                "message": "No JSON received"
            }), 400

        image_url = data.get("image_url")

        if not image_url:
            return jsonify({
                "status": "error",
                "message": "image_url not found",
                "received": data
            }), 400

        response = requests.get(
            image_url,
            timeout=30
        )

        img_array = np.asarray(
            bytearray(response.content),
            dtype=np.uint8
        )

        img = cv2.imdecode(
            img_array,
            cv2.IMREAD_COLOR
        )

        if img is None:
            return jsonify({
                "status": "error",
                "message": "cannot read image"
            }), 400

        h, w = img.shape[:2]

        return jsonify({
            "status": "success",
            "width": int(w),
            "height": int(h)
        })

    except Exception as e:

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=10000
    )

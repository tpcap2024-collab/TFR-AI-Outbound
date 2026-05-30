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

        data = request.get_json()

        image_url = data.get("image_url")

        if not image_url:
            return jsonify({
                "error": "image_url not found"
            }), 400

        response = requests.get(image_url, timeout=30)

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
                "error": "cannot read image"
            }), 400

        # ==========================
        # RESIZE
        # ==========================

        img_resized = cv2.resize(
            img,
            (800, 600)
        )

        h, w = img_resized.shape[:2]

        # ==========================
        # HSV
        # ==========================

        hsv = cv2.cvtColor(
            img_resized,
            cv2.COLOR_BGR2HSV
        )

        hsv = cv2.GaussianBlur(
            hsv,
            (5, 5),
            0
        )

        # ==========================
        # COLOR RANGE
        # ==========================

        lower_egg = np.array([10, 30, 60])
        upper_egg = np.array([45, 255, 255])

        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])

        lower_red2 = np.array([160, 70, 50])
        upper_red2 = np.array([180, 255, 255])

        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])

        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])

        lower_white = np.array([0, 0, 160])
        upper_white = np.array([180, 70, 255])

        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 60])

        # ==========================
        # MASKS
        # ==========================

        mask_egg = cv2.inRange(
            hsv,
            lower_egg,
            upper_egg
        )

        mask_red = cv2.bitwise_or(
            cv2.inRange(hsv, lower_red1, upper_red1),
            cv2.inRange(hsv, lower_red2, upper_red2)
        )

        mask_blue = cv2.inRange(
            hsv,
            lower_blue,
            upper_blue
        )

        mask_green = cv2.inRange(
            hsv,
            lower_green,
            upper_green
        )

        mask_white = cv2.inRange(
            hsv,
            lower_white,
            upper_white
        )

        mask_dark = cv2.inRange(
            hsv,
            lower_dark,
            upper_dark
        )

        return jsonify({
            "status": "success",
            "image_width": w,
            "image_height": h
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=10000
    )

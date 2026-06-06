from flask import Flask, request, jsonify
import requests
import cv2
import numpy as np
import traceback

app = Flask(__name__)

# =========================
# APP SETTINGS
# =========================
APP_ID = "5ebec09a-62dd-4fa9-8f14-830fb104518f"
ACCESS_KEY = "V2-2ZX8p-jmYBx-bH09l-nFTYW-cvV8W-7wNy3-zqOQQ-JvMrp"
TABLE_NAME = "Data TFR"


# =========================
# UPDATE APP SHEET (FIXED)
# =========================
def update_appsheet(date_value, volume):

    url = f"https://api.appsheet.com/api/v2/apps/{APP_ID}/tables/{TABLE_NAME}/Action"

    headers = {
        "ApplicationAccessKey": ACCESS_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "Action": "Edit",
        "Rows": [
            {
                "Date": date_value,   # ✅ KEY = Date
                "TFR AI": volume
            }
        ]
    }

    try:
        r = requests.post(url, json=payload, headers=headers, timeout=30)
        print("APP SHEET UPDATE:", r.status_code, r.text)

    except Exception as e:
        print("UPDATE ERROR:", str(e))


# =========================
# HOME
# =========================
@app.route("/")
def home():
    return "TFR AI Running"


# =========================
# PREDICT
# =========================
@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json(silent=True)
        print("JSON:", data)

        if not data:
            return jsonify({"status": "error"}), 400

        image_url = data.get("link")
        date_value = data.get("datetime")   # ✅ ใช้ Date จริงจาก AppSheet

        print("IMAGE URL:", image_url)
        print("DATE:", date_value)

        if not image_url or not date_value:
            return jsonify({"status": "error", "message": "missing data"}), 400

        # =========================
        # DOWNLOAD IMAGE
        # =========================
        response = requests.get(
            image_url,
            timeout=60,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        img = cv2.imdecode(
            np.frombuffer(response.content, np.uint8),
            cv2.IMREAD_COLOR
        )

        if img is None:
            return jsonify({"status": "error", "message": "decode failed"}), 400

        # =========================
        # PROCESS
        # =========================
        img = cv2.resize(img, (800, 600))

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (5,5), 0)

        mask = cv2.inRange(hsv, (10,30,60), (45,255,255))

        red = cv2.bitwise_or(
            cv2.inRange(hsv, (0,70,50), (10,255,255)),
            cv2.inRange(hsv, (160,70,50), (180,255,255))
        )

        blue = cv2.inRange(hsv, (90,50,50), (130,255,255))
        green = cv2.inRange(hsv, (40,40,40), (80,255,255))
        white = cv2.inRange(hsv, (0,0,160), (180,70,255))

        combined = mask | red | blue | green | white

        h, w = combined.shape
        roi = combined[int(h*0.18):int(h*0.80), int(w*0.05):int(w*0.95)]

        roi[:int(roi.shape[0]*0.12), :] = 0

        roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        roi = cv2.dilate(roi, np.ones((7,7), np.uint8), 1)

        # =========================
        # VOLUME
        # =========================
        volume = int((cv2.countNonZero(roi) / roi.size) * 100)

        if volume >= 85:
            volume = 100

        print("VOLUME:", volume)

        # =========================
        # UPDATE APP SHEET
        # =========================
        update_appsheet(date_value, volume)

        return jsonify({
            "status": "success",
            "volume": volume
        })

    except Exception:
        print(traceback.format_exc())
        return jsonify({"status": "error"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

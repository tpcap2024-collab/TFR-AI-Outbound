from flask import Flask, request, jsonify
import requests
import cv2
import numpy as np
import traceback
import threading

app = Flask(__name__)

APP_ID = "5ebec09a-62dd-4fa9-8f14-830fb104518f"
ACCESS_KEY = "V2-2ZX8p-jmYBx-bH09l-nFTYW-cvV8W-7wNy3-zqOQQ-JvMrp"
TABLE_NAME = "Data TFR"

lock = threading.Lock()
processed_ids = set()

# =========================
# IMAGE DOWNLOAD
# =========================
def download_image(url):
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None

        img = cv2.imdecode(
            np.frombuffer(r.content, np.uint8),
            cv2.IMREAD_COLOR
        )
        return img
    except:
        return None


# =========================
# 🔥 PRODUCTION VOLUME ENGINE
# =========================
def gen_volume(img):

    img = cv2.resize(img, (640, 480))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # =========================
    # CLEAN MASK (ลด noise + ผนัง)
    # =========================
    mask = (
        cv2.inRange(hsv, (10, 35, 60), (40, 255, 255)) |
        cv2.inRange(hsv, (0, 60, 50), (10, 255, 255)) |
        cv2.inRange(hsv, (160, 60, 50), (180, 255, 255)) |
        cv2.inRange(hsv, (90, 50, 50), (130, 255, 255))
    )

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    h, w = mask.shape

    # =========================
    # ROI (ลึก + ครอบรถจริง)
    # =========================
    roi = mask[int(h*0.08):int(h*0.95), int(w*0.02):int(w*0.98)]

    if roi.size == 0:
        return 0

    # =========================
    # FEATURE 1: AREA DENSITY
    # =========================
    area = np.count_nonzero(roi) / roi.size

    # =========================
    # FEATURE 2: TOP OCCUPANCY (สำคัญมาก)
    # =========================
    top = roi[:int(roi.shape[0]*0.35), :]
    top_fill = np.mean(top > 0)

    # =========================
    # FEATURE 3: BOTTOM DENSITY
    # =========================
    bottom = roi[int(roi.shape[0]*0.6):, :]
    bottom_fill = np.mean(bottom > 0)

    # =========================
    # FEATURE 4: EDGE SUPPORT (depth hint)
    # =========================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_roi = edges[int(h*0.08):int(h*0.95), int(w*0.02):int(w*0.98)]
    edge = np.count_nonzero(edge_roi) / edge_roi.size

    # =========================
    # FINAL MODEL (production weighted fusion)
    # =========================
    volume = (
        area * 100 * 0.35 +
        top_fill * 100 * 0.30 +
        bottom_fill * 100 * 0.25 +
        edge * 100 * 0.10
    )

    # =========================
    # 🔥 HIGH-FILL BOOST (fix 100% under-estimate)
    # =========================
    if top_fill > 0.65 and bottom_fill > 0.65:
        volume *= 1.20

    if top_fill > 0.80:
        volume *= 1.10

    # =========================
    # 🔥 LOW-FILL CORRECTION
    # =========================
    if area < 0.10:
        volume *= 0.85

    # =========================
    # FINAL CLAMP
    # =========================
    volume = int(round(volume / 5) * 5)
    volume = max(0, min(100, volume))

    return volume


# =========================
# APPSHEET UPDATE
# =========================
def update_appsheet(row_id, volume_text):

    url = f"https://api.appsheet.com/api/v2/apps/{APP_ID}/tables/{TABLE_NAME}/Action"

    headers = {
        "ApplicationAccessKey": ACCESS_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "Action": "Edit",
        "Rows": [
            {
                "id": row_id,
                "TFR AI": volume_text,
                "status": "Done"
            }
        ]
    }

    try:
        requests.post(url, json=payload, headers=headers, timeout=20)
    except:
        pass


# =========================
# API
# =========================
@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json(silent=True)

        if not data:
            return jsonify({"error": "no json"}), 400

        image_url = data.get("link")
        row_id = data.get("id")

        if not image_url or not row_id:
            return jsonify({"error": "missing data"}), 400

        with lock:
            if row_id in processed_ids:
                return jsonify({"status": "skipped"}), 200
            processed_ids.add(row_id)

        img = download_image(image_url)
        if img is None:
            return jsonify({"error": "image fail"}), 400

        volume = gen_volume(img)

        print("FINAL VOLUME:", volume)

        update_appsheet(row_id, f"{volume}%")

        return jsonify({
            "status": "success",
            "volume": volume
        })

    except:
        print(traceback.format_exc())
        return jsonify({"error": "server error"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, threaded=True)

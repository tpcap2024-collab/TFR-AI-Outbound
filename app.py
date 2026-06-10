from flask import Flask, request, jsonify
import requests
import cv2
import numpy as np
import traceback
import time
import threading

app = Flask(__name__)

APP_ID = "5ebec09a-62dd-4fa9-8f14-830fb104518f"
ACCESS_KEY = "V2-2ZX8p-jmYBx-bH09l-nFTYW-cvV8W-7wNy3-zqOQQ-JvMrp"
TABLE_NAME = "Data TFR"

processed_ids = set()
lock = threading.Lock()


# =========================
# DOWNLOAD IMAGE
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
# 🔥 IMPROVED VOLUME MODEL (LEVEL 1 FIX)
# =========================
def gen_volume(img):

    img = cv2.resize(img, (640, 480))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # =========================
    # 🔥 CLEANER COLOR MASK (ลด false positive)
    # =========================
    mask = (
        cv2.inRange(hsv, (10, 40, 60), (40, 255, 255)) |   # กล่องทั่วไป
        cv2.inRange(hsv, (0, 70, 50), (10, 255, 255)) |    # แดง
        cv2.inRange(hsv, (160, 70, 50), (180, 255, 255)) | # แดงเข้ม
        cv2.inRange(hsv, (90, 60, 60), (130, 255, 255))    # น้ำเงิน
    )

    # =========================
    # 🔥 MORPHOLOGY (สำคัญมาก)
    # =========================
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    h, w = mask.shape

    # =========================
    # 🔥 ROI (ตัดผนัง + พื้น + เพดาน)
    # =========================
    roi = mask[int(h*0.18):int(h*0.88), int(w*0.05):int(w*0.95)]

    if roi.size == 0:
        return 0

    # =========================
    # AREA DENSITY (หลัก)
    # =========================
    area_density = np.count_nonzero(roi) / roi.size
    area_density = np.clip(area_density, 0, 1)

    # =========================
    # VERTICAL SIGNAL (ลด sensitivity)
    # =========================
    v_proj = np.sum(roi, axis=1)
    v_norm = v_proj / (np.max(v_proj) + 1e-6)
    v_score = np.mean(v_norm > 0.12)   # ⬅️ เพิ่ม threshold

    # =========================
    # HORIZONTAL SIGNAL
    # =========================
    h_proj = np.sum(roi, axis=0)
    h_norm = h_proj / (np.max(h_proj) + 1e-6)
    h_score = np.mean(h_norm > 0.12)

    # =========================
    # 🔥 FINAL (ลด overestimate)
    # =========================
    volume = (
        area_density * 100 * 0.75 +
        v_score * 100 * 0.15 +
        h_score * 100 * 0.10
    )

    # =========================
    # 🔥 STABILIZER (กัน 40% → 80% หลอก)
    # =========================
    if area_density < 0.15:
        volume *= 0.75
    elif area_density < 0.35:
        volume *= 0.9

    # =========================
    # ROUND + CLAMP
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
        volume_text = f"{volume}%"

        print("VOLUME:", volume_text)

        update_appsheet(row_id, volume_text)

        return jsonify({
            "status": "success",
            "id": row_id,
            "volume": volume_text
        })

    except:
        print(traceback.format_exc())
        return jsonify({"error": "server error"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, threaded=True)

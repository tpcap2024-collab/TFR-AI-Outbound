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

# =========================
# MEMORY + LOCK (สำคัญมาก)
# =========================
processed_ids = set()
lock = threading.Lock()


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# =========================
# APP SHEET UPDATE
# =========================
def update_appsheet(row_id, volume_text, status="Done"):

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
                "status": status
            }
        ]
    }

    for i in range(3):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=30)
            print(f"[APP SHEET TRY {i+1}] {r.status_code} {r.text}")

            if r.status_code == 200:
                return True

        except Exception as e:
            print("UPDATE ERROR:", str(e))
            time.sleep(1)

    return False


# =========================
# IMAGE DOWNLOAD
# =========================
def download_image(image_url):

    for i in range(3):
        try:
            r = requests.get(image_url, timeout=60)

            if r.status_code == 200:
                img = cv2.imdecode(
                    np.frombuffer(r.content, np.uint8),
                    cv2.IMREAD_COLOR
                )
                if img is not None:
                    return img

        except Exception as e:
            print(f"[DOWNLOAD RETRY {i+1}]", e)

        time.sleep(1)

    return None


# =========================
# MAIN API
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

        # =========================
        # LOCK กันยิงซ้ำ
        # =========================
        with lock:
            if row_id in processed_ids:
                return jsonify({"status": "skipped"}), 200
            processed_ids.add(row_id)

        # =========================
        # delay กัน image upload ยังไม่เสร็จ
        # =========================
        time.sleep(2)

        # =========================
        # IMAGE
        # =========================
        img = download_image(image_url)

        if img is None:
            return jsonify({"error": "image fail"}), 400

        img = cv2.resize(img, (800, 600))

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

        mask = cv2.inRange(hsv, (10, 30, 60), (45, 255, 255))

        red = cv2.bitwise_or(
            cv2.inRange(hsv, (0, 70, 50), (10, 255, 255)),
            cv2.inRange(hsv, (160, 70, 50), (180, 255, 255))
        )

        blue = cv2.inRange(hsv, (90, 50, 50), (130, 255, 255))
        green = cv2.inRange(hsv, (40, 40, 40), (80, 255, 255))
        white = cv2.inRange(hsv, (0, 0, 160), (180, 70, 255))

        combined = mask | red | blue | green | white

        h, w = combined.shape
        roi = combined[int(h*0.18):int(h*0.80), int(w*0.05):int(w*0.95)]

        roi[:int(roi.shape[0]*0.12), :] = 0

        roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        roi = cv2.dilate(roi, np.ones((7, 7), np.uint8), 1)

        fill = cv2.countNonZero(roi)
        total = roi.size

        volume = int((fill / total) * 100)

        if volume >= 85:
            volume = 100

        volume_text = f"{volume}%"

        print("VOLUME:", volume_text)

        # =========================
        # UPDATE
        # =========================
        update_appsheet(row_id, volume_text)

        return jsonify({
            "status": "success",
            "id": row_id,
            "volume": volume_text
        })

    except Exception:
        print(traceback.format_exc())
        return jsonify({"error": "server error"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

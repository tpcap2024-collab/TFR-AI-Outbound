from flask import Flask, request, jsonify, send_file
import requests
import cv2
import numpy as np
import traceback
import threading
import os

app = Flask(__name__)

# =========================
# APPSHEET CONFIG
# =========================
APP_ID = "5ebec09a-62dd-4fa9-8f14-830fb104518f"
ACCESS_KEY = "V2-2ZX8p-jmYBx-bH09l-nFTYW-cvV8W-7wNy3-zqOQQ-JvMrp"
TABLE_NAME = "Data TFR"

# =========================
# DEBUG CONFIG
# =========================
DEBUG_DIR = "/tmp"
os.makedirs(DEBUG_DIR, exist_ok=True)

# =========================
# LOCK
# =========================
processed_ids = set()
lock = threading.Lock()

# =========================
# DOWNLOAD IMAGE
# =========================
def download_image(url):
    try:
        r = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        if r.status_code != 200:
            return None

        img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        return img

    except:
        return None


# =========================
# DEBUG SAVE
# =========================
def save_debug(name, img):
    if img is None:
        return
    cv2.imwrite(os.path.join(DEBUG_DIR, name), img)


# =========================
# OUTBOUND (ง่าย version)
# =========================
def gen_volume(img, debug=True, return_empty=False):

    img = cv2.resize(img, (640, 480))
    h, w = img.shape[:2]

    roi = img[int(h*0.25):int(h*0.80), int(w*0.1):int(w*0.9)]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    green = cv2.inRange(hsv, (35, 50, 50), (90,255,255))
    brown = cv2.inRange(hsv, (5,40,40), (35,255,255))
    blue  = cv2.inRange(hsv, (85,40,40), (130,255,255))
    dark  = cv2.inRange(hsv, (0,0,0), (180,255,60))

    edges = cv2.Canny(gray, 40, 120)

    cargo = green | brown | blue | dark | edges

    filled = cv2.countNonZero(cargo) / cargo.size * 100
    filled = min(100, filled)

    result = (100-filled) if return_empty else filled
    result = int(round(result/5)*5)

    if debug:
        overlay = roi.copy()
        overlay[cargo > 0] = (0,255,0)
        save_debug("debug_overlay.jpg", overlay)

    return result


def gen_pallet(img, debug=True):

    img = cv2.resize(img,(1280,720))
    H,W = img.shape[:2]

    roi = img[
        int(H*0.15):int(H*0.85),
        int(W*0.02):int(W*0.98)
    ]

    rh,rw = roi.shape[:2]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # =========================
    # GREEN MASK (rack)
    # =========================
    green_mask = cv2.inRange(
        hsv,
        (30, 30, 30),
        (90, 255, 255)
    )

    # clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, 1)

    debug_img = roi.copy()

    # =========================
    # STEP 1: SPLIT LEFT-RIGHT
    # =========================
    col_sum = np.sum(green_mask > 0, axis=0)
    col_threshold = np.max(col_sum) * 0.25

    x_blocks = []
    in_block = False

    for i in range(len(col_sum)):
        if col_sum[i] > col_threshold and not in_block:
            start = i
            in_block = True

        elif col_sum[i] <= col_threshold and in_block:
            end = i
            x_blocks.append((start,end))
            in_block = False

    if in_block:
        x_blocks.append((start,len(col_sum)-1))

    # =========================
    # STEP 2: PROCESS EACH COLUMN BLOCK
    # =========================
    pallet_count = 0

    for (x1,x2) in x_blocks:

        width = x2-x1
        if width < rw*0.05:
            continue

        sub_mask = green_mask[:, x1:x2]

        # -------------------------
        # STEP 2.1: split top-bottom (layer)
        # -------------------------
        row_sum = np.sum(sub_mask > 0, axis=1)
        row_threshold = np.max(row_sum) * 0.2

        y_blocks = []
        in_row = False

        for j in range(len(row_sum)):
            if row_sum[j] > row_threshold and not in_row:
                sy = j
                in_row = True

            elif row_sum[j] <= row_threshold and in_row:
                ey = j
                y_blocks.append((sy,ey))
                in_row = False

        if in_row:
            y_blocks.append((sy,len(row_sum)-1))

        # -------------------------
        # STEP 2.2: each sub-block = pallet
        # -------------------------
        for (y1,y2) in y_blocks:

            height = y2 - y1

            if height < rh*0.12:
                continue

            pallet_count += 1

            # draw
            cv2.rectangle(
                debug_img,
                (x1,y1),
                (x2,y2),
                (0,255,0),
                3
            )

            cv2.putText(
                debug_img,
                f"P{pallet_count}",
                (x1+5, y1+25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2
            )

    # =========================
    # RESULT
    # =========================
    cv2.putText(
        debug_img,
        f"PALLET={pallet_count}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,255),
        3
    )

    print("="*50)
    print("FULL PRODUCTION PALLET")
    print("TOTAL =", pallet_count)
    print("="*50)

    # =========================
    # DEBUG
    # =========================
    if debug:
        save_debug("debug_green_mask.jpg", green_mask)
        save_debug("debug_pallet_box.jpg", debug_img)

    return pallet_count



# =========================
# APPSHEET
# =========================
def update_appsheet(row_id, text):
    if not APP_ID or not ACCESS_KEY:
        return

    url = f"https://api.appsheet.com/api/v2/apps/{APP_ID}/tables/{TABLE_NAME}/Action"

    headers = {
        "ApplicationAccessKey": ACCESS_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "Action":"Edit",
        "Rows":[
            {"ID":row_id,"TFR AI":text,"status":"Done"}
        ]
    }

    try:
        requests.post(url,json=payload,headers=headers,timeout=10)
    except:
        pass


# =========================
# DEBUG SERVER
# =========================
@app.route("/debug/<path:filename>")
def debug_file(filename):
    path = os.path.join(DEBUG_DIR, filename)
    if not os.path.exists(path):
        return "Not found",404
    return send_file(path)


@app.route("/debug-list")
def debug_list():
    return jsonify(os.listdir(DEBUG_DIR))


# =========================
# API
# =========================
@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    print("="*40)
    print("REQUEST:", data)
    print("="*40)

    image_url = data.get("link")
    row_id = data.get("id")

    project = str(data.get("project","")).strip().lower()

    if project not in ["inbound", "outbound"]:
        print("⚠️ INVALID PROJECT → FORCE INBOUND")
        project = "inbound"

    print("PROJECT =", project)

    if not image_url or not row_id:
        return {"error":"missing"},400

    img = download_image(image_url)
    if img is None:
        return {"error":"image fail"},400

    if project == "inbound":
        print("RUN PALLET MODEL")
        result = str(gen_pallet(img))

    else:
        print("RUN VOLUME MODEL")
        result = f"{gen_volume(img)}%"

    update_appsheet(row_id, result)

    return {
        "status":"success",
        "project": project,
        "result": result
    }


# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

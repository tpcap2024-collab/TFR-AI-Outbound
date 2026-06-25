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

    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # =========================
    # GREEN MASK
    # =========================
    green_mask = cv2.inRange(hsv,(30,30,30),(90,255,255))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, 1)

    # =========================
    # EDGE (ช่วย structure)
    # =========================
    edges = cv2.Canny(gray, 50, 150)
    combine = cv2.bitwise_or(green_mask, edges)

    # =========================
    # STEP 1: หา raw blocks
    # =========================
    contours,_ = cv2.findContours(
        combine,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    raw_blocks = []

    for c in contours:
        x,y,w,h = cv2.boundingRect(c)

        if w*h < rw*rh*0.01:
            continue

        # กันเสา
        if w < rw*0.04:
            continue

        raw_blocks.append((x,y,x+w,y+h))

    # =========================
    # ✅ STEP 2: MERGE "เฉพาะแนวตั้ง"
    # =========================
    def merge_vertical(blocks, y_dist=60):

        merged = []

        for (x1,y1,x2,y2) in blocks:

            merged_flag = False

            for i,(mx1,my1,mx2,my2) in enumerate(merged):

                # ✅ ต้อง overlap X (ห้าม merge ข้ามช่อง)
                overlap_x = not (x2 < mx1 or x1 > mx2)

                # ✅ ใกล้กันแนว Y
                close_y = abs(y1-my2) < y_dist or abs(y2-my1) < y_dist

                if overlap_x and close_y:
                    nx1 = min(x1,mx1)
                    ny1 = min(y1,my1)
                    nx2 = max(x2,mx2)
                    ny2 = max(y2,my2)

                    merged[i] = (nx1,ny1,nx2,ny2)
                    merged_flag = True
                    break

            if not merged_flag:
                merged.append((x1,y1,x2,y2))

        return merged

    merged_blocks = merge_vertical(raw_blocks)

    debug_img = roi.copy()

    # =========================
    # ✅ STEP 3: FINAL FILTER (ตัดกล่องใหญ่เกิน + noise)
    # =========================
    pallet_count = 0

    for (x1,y1,x2,y2) in merged_blocks:

        w = x2-x1
        h = y2-y1

        # ✅ ตัด block ใหญ่เกิน (กันทั้งภาพ)
        if w > rw * 0.35:
            continue

        # ✅ กัน noise
        if w < rw * 0.10:
            continue

        if h < rh * 0.15:
            continue

        # ✅ ต้องมี green จริง
        sub = green_mask[y1:y2, x1:x2]
        density = np.sum(sub>0)/(sub.size+1e-6)

        if density < 0.05:
            continue

        pallet_count += 1

        # ✅ RED BOX = pallet จริง
        cv2.rectangle(debug_img,(x1,y1),(x2,y2),(0,0,255),3)

        cv2.putText(debug_img,
                    f"P{pallet_count}",
                    (x1+5,y1+30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,0,255),
                    2)

    # =========================
    # RESULT
    # =========================
    cv2.putText(debug_img,
                f"PALLET={pallet_count}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,255),
                3)

    print("="*50)
    print("FINAL PALLET =", pallet_count)
    print("="*50)

    if debug:
        save_debug("debug_green_mask.jpg", green_mask)
        save_debug("debug_edges.jpg", edges)
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

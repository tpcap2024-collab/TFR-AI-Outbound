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
    # COLOR MASKS
    # =========================
    green_mask = cv2.inRange(hsv, (30,30,30), (90,255,255))
    blue_mask  = cv2.inRange(hsv, (90,40,40), (130,255,255))

    cream1 = cv2.inRange(hsv, (5,10,120), (40,120,255))
    cream2 = cv2.inRange(hsv, (0,0,180), (180,50,255))
    cream_mask = cv2.bitwise_or(cream1, cream2)

    # clean
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, 1)

    # =========================
    # EDGE (ช่วย structure)
    # =========================
    edges = cv2.Canny(gray, 50, 150)
    combine = cv2.bitwise_or(green_mask, edges)

    debug_img = roi.copy()

    # =========================
    # STEP 1: SPLIT X
    # =========================
    col_sum = np.sum(combine > 0, axis=0)
    col_th  = np.max(col_sum) * 0.35

    x_blocks = []
    in_block = False

    for i in range(len(col_sum)):

        if col_sum[i] > col_th and not in_block:
            start = i
            in_block = True

        elif col_sum[i] <= col_th and in_block:
            end = i
            width = end - start

            # force split block ใหญ่
            if width > rw * 0.22:
                mid = (start + end)//2
                x_blocks.append((start, mid))
                x_blocks.append((mid, end))
            else:
                x_blocks.append((start, end))

            in_block = False

    if in_block:
        x_blocks.append((start, len(col_sum)-1))

    # =========================
    # FILTER GAP + WIDTH
    # =========================
    filtered = []

    for (x1,x2) in x_blocks:

        width = x2 - x1
        block = green_mask[:, x1:x2]

        density = np.sum(block > 0) / (block.size + 1e-6)

        if density < 0.05:
            continue

        if width < rw * 0.08:
            continue

        filtered.append((x1,x2))

    x_blocks = filtered

    # =========================
    # SPLIT LARGE BLOCK (สำคัญ)
    # =========================
    def split_large(mask, x1, x2):
        sub = mask[:, x1:x2]

        col_sum = np.sum(sub > 0, axis=0)
        th = np.max(col_sum) * 0.30

        splits = []
        in_block = False

        for i in range(len(col_sum)):

            if col_sum[i] > th and not in_block:
                sx = i
                in_block = True

            elif col_sum[i] <= th and in_block:
                ex = i

                if (ex - sx) > rw*0.06:
                    splits.append((x1+sx, x1+ex))

                in_block = False

        if in_block:
            if (len(col_sum)-sx) > rw*0.06:
                splits.append((x1+sx, x2))

        return splits

    final_blocks = []

    for (x1,x2) in x_blocks:
        if (x2-x1) > rw * 0.18:
            final_blocks.extend(split_large(green_mask, x1, x2))
        else:
            final_blocks.append((x1,x2))

    x_blocks = final_blocks

    # =========================
    # CLASSIFY FUNCTION
    # =========================
    def classify_block(x1,x2,y1,y2):

        area = (x2-x1)*(y2-y1)

        g = np.count_nonzero(green_mask[y1:y2, x1:x2]) / (area+1e-6)
        b = np.count_nonzero(blue_mask[y1:y2, x1:x2]) / (area+1e-6)
        c = np.count_nonzero(cream_mask[y1:y2, x1:x2]) / (area+1e-6)

        if g > 0.15:
            return "green"
        elif b > 0.10:
            return "blue"
        elif c > 0.20:
            return "cream"
        else:
            return "unknown"

    # =========================
    # COUNT
    # =========================
    pallet_count = 0

    counts = {
        "green":0,
        "blue":0,
        "cream":0,
        "unknown":0
    }

    for (x1,x2) in x_blocks:

        sub = green_mask[:, x1:x2]

        row_sum = np.sum(sub > 0, axis=1)
        row_th  = np.max(row_sum) * 0.30

        in_row = False

        for j in range(len(row_sum)):

            if row_sum[j] > row_th and not in_row:
                sy = j
                in_row = True

            elif row_sum[j] <= row_th and in_row:

                ey = j
                height = ey - sy

                if height > rh * 0.12:

                    t = classify_block(x1,x2,sy,ey)

                    pallet_count += 1
                    counts[t] += 1

                    cv2.rectangle(debug_img,
                                  (x1, sy),
                                  (x2, ey),
                                  (0,255,0),3)

                    cv2.putText(debug_img,
                                f"{t[0].upper()}{counts[t]}",
                                (x1+5, sy+30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0,255,0),2)

                in_row = False

        if in_row:
            if (len(row_sum)-sy) > rh * 0.12:

                t = classify_block(x1,x2,sy,len(row_sum))

                pallet_count += 1
                counts[t] += 1

                cv2.rectangle(debug_img,
                              (x1, sy),
                              (x2, len(row_sum)),
                              (0,255,0),3)

    # =========================
    # RESULT
    # =========================
    cv2.putText(debug_img,
                f"T={pallet_count} G={counts['green']} B={counts['blue']} C={counts['cream']}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,255),
                3)

    print("="*50)
    print("FINAL RESULT")
    print("TOTAL:", pallet_count)
    print("GREEN:", counts["green"])
    print("BLUE :", counts["blue"])
    print("CREAM:", counts["cream"])
    print("="*50)

    if debug:
        save_debug("debug_green_mask.jpg", green_mask)
        save_debug("debug_edges.jpg", edges)
        save_debug("debug_pallet_box.jpg", debug_img)

    return {
        "total": pallet_count,
        "green": counts["green"],
        "blue": counts["blue"],
        "cream": counts["cream"]
    }



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

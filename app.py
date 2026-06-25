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


# =========================
# INBOUND FIXED (ตัวสำคัญ)
# =========================
def gen_pallet(img, debug=True):

    cells = []
    types = []

    def add_cell(box, t):
        cells.append(box)
        types.append(t)

    def color(t):
        if t == "cream": return (0,255,0)
        if t == "blue":  return (255,180,0)
        if t == "wood":  return (0,180,255)
        return (0,255,255)

    def split_grid(draw, box, cols, rows, t):
        x,y,w,h = box

        for r in range(rows):
            for c in range(cols):
                x1 = x + int(c*w/cols)
                x2 = x + int((c+1)*w/cols)
                y1 = y + int(r*h/rows)
                y2 = y + int((r+1)*h/rows)

                if (x2-x1) < 25 or (y2-y1) < 25:
                    continue

                add_cell((x1,y1,x2-x1,y2-y1), t)

                cv2.rectangle(draw,(x1,y1),(x2,y2),color(t),2)

    # =========================
    # PREPROCESS
    # =========================
    img = cv2.resize(img,(1280,720))
    H, W = img.shape[:2]

    roi = img[
        int(H*0.17):int(H*0.84),
        int(W*0.02):int(W*0.98)
    ]

    rh, rw = roi.shape[:2]

    # =========================
    # LIGHT NORMALIZATION
    # =========================
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(2.0,(8,8))
    l = clahe.apply(l)

    norm = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2BGR)

    hsv  = cv2.cvtColor(norm, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)

    # =========================
    # COLOR MASKS
    # =========================
    cream = cv2.inRange(hsv, (5,10,80), (40,160,255))
    white = cv2.inRange(hsv, (0,0,160), (180,60,255))
    cream_mask = cv2.bitwise_or(cream, white)

    blue_mask = cv2.inRange(hsv, (85,40,40), (130,255,255))
    wood_mask = cv2.inRange(hsv, (5,40,40), (30,200,200))

    # =========================
    # CLEAN
    # =========================
    def clean(mask, k=5):
        return cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT,(k,k)),
            iterations=1
        )

    cream_mask = clean(cream_mask)
    blue_mask  = clean(blue_mask)
    wood_mask  = clean(wood_mask)

    # =========================
    # MERGE ALL
    # =========================
    combined = cv2.bitwise_or(cream_mask, blue_mask)
    combined = cv2.bitwise_or(combined, wood_mask)

    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT,(15,15)),
        iterations=2
    )

    # =========================
    # FIND BLOCKS
    # =========================
    contours,_ = cv2.findContours(
        combined,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    blocks = []

    for c in contours:
        x,y,w,h = cv2.boundingRect(c)

        if w*h < rw*rh*0.015:
            continue

        blocks.append((x,y,w,h))

    # sort ซ้าย → ขวา
    blocks = sorted(blocks, key=lambda b: b[0])

    debug_img = norm.copy()

    # =========================
    # CLASSIFY + GRID
    # =========================
    for box in blocks:

        x,y,w,h = box
        area = float(max(1,w*h))

        cream_ratio = cv2.countNonZero(
            cream_mask[y:y+h, x:x+w]) / area

        blue_ratio = cv2.countNonZero(
            blue_mask[y:y+h, x:x+w]) / area

        wood_ratio = cv2.countNonZero(
            wood_mask[y:y+h, x:x+w]) / area

        # =========================
        # TYPE LOGIC
        # =========================
        if cream_ratio > 0.05:
            t = "cream"

        elif blue_ratio > 0.04:
            t = "blue"

        else:
            t = "wood"

        # =========================
        # GRID DECISION
        # =========================
        aspect = w / float(max(1,h))

        if t == "cream":
            cols = max(2, int(aspect * 2))
            rows = 2

        elif t == "blue":
            cols = 1
            rows = max(2, int(h / 120))

        else:  # wood
            cols = 1
            rows = max(2, int(h / 100))

        split_grid(debug_img, box, cols, rows, t)

        # =========================
        # DRAW BLOCK
        # =========================
        cv2.rectangle(debug_img,(x,y),(x+w,y+h),color(t),3)

        cv2.putText(
            debug_img,
            f"{t}",
            (x+5,y+25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,color(t),2
        )

    # =========================
    # COUNT
    # =========================
    cream_c = sum(1 for t in types if t=="cream")
    blue_c  = sum(1 for t in types if t=="blue")
    wood_c  = sum(1 for t in types if t=="wood")

    total = len(cells)

    cv2.putText(
        debug_img,
        f"C={cream_c} B={blue_c} W={wood_c} TOTAL={total}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,255),
        3
    )

    # =========================
    # LOG
    # =========================
    print("="*50)
    print("PRODUCTION PALLET RESULT")
    print("CREAM:", cream_c)
    print("BLUE :", blue_c)
    print("WOOD :", wood_c)
    print("TOTAL:", total)
    print("="*50)

    # =========================
    # DEBUG
    # =========================
    if debug:
        save_debug("debug_overlay.jpg", debug_img)
        save_debug("debug_blocks.jpg", debug_img)
        save_debug("debug_grid.jpg", debug_img)
        save_debug("debug_cargo.jpg", combined)
        save_debug("debug_cream.jpg", cream_mask)
        save_debug("debug_blue.jpg", blue_mask)
        save_debug("debug_wood.jpg", wood_mask)

    return total


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

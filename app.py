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

    def add(x,y,w,h,t):
        cells.append((x,y,w,h))
        types.append(t)

    def draw(draw, box, cols, rows, t, prefix):
        x,y,w,h = box
        no = 1

        for r in range(rows):
            for c in range(cols):
                x1 = x + int(c*w/cols)
                x2 = x + int((c+1)*w/cols)
                y1 = y + int(r*h/rows)
                y2 = y + int((r+1)*h/rows)

                add(x1,y1,x2-x1,y2-y1,t)

                cv2.rectangle(draw,(x1,y1),(x2,y2),(0,255,255),2)
                cv2.putText(draw,f"{prefix}{no}",
                            (x1+5,y1+25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,(0,255,255),2)
                no+=1

    img = cv2.resize(img,(1280,720))
    H,W = img.shape[:2]

    roi = img[int(H*0.17):int(H*0.84), int(W*0.01):int(W*0.99)]
    rh,rw = roi.shape[:2]

    debug_img = roi.copy()

    y1 = int(rh*0.15)
    y2 = int(rh*0.88)
    h_box = y2-y1

    # แบ่ง 3 block
    wood = (0, y1, int(rw*0.24), h_box)
    blue = (int(rw*0.24), y1, int(rw*0.18), h_box)
    cream= (int(rw*0.42), y1, int(rw*0.58), h_box)

    # =====================
    # COUNT
    # =====================
    draw(debug_img, wood, 1, 3, "wood", "W")
    draw(debug_img, blue, 1, 2, "blue", "B")
    draw(debug_img, cream, 4, 2, "cream", "C")

    # =====================
    # COUNT RESULT
    # =====================
    wood_c = sum(1 for t in types if t=="wood")
    blue_c = sum(1 for t in types if t=="blue")
    cream_c= sum(1 for t in types if t=="cream")

    total = len(cells)

    cv2.putText(
        debug_img,
        f"CREAM={cream_c} BLUE={blue_c} WOOD={wood_c} TOTAL={total}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,255),
        3
    )

    if debug:
        save_debug("debug_blocks.jpg", debug_img)
        save_debug("debug_pallet_box.jpg", debug_img)
        save_debug("debug_grid.jpg", debug_img)

    print("===== INBOUND RESULT =====")
    print("CREAM:", cream_c)
    print("BLUE :", blue_c)
    print("WOOD :", wood_c)
    print("TOTAL:", total)

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

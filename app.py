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

    img = cv2.resize(img, (1280, 720))
    H, W = img.shape[:2]

    roi = img[
        int(H * 0.15):int(H * 0.85),
        int(W * 0.02):int(W * 0.98)
    ]

    rh, rw = roi.shape[:2]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # =========================
    # GREEN MASK (rack structure)
    # =========================
    green_mask = cv2.inRange(
        hsv,
        (30, 35, 35),
        (90, 255, 255)
    )

    # =========================
    # CUT NON-CARGO AREA
    # สำคัญมาก: กันล้อ/คานล่าง/พื้นรถ
    # =========================
    top_cut = int(rh * 0.06)
    bottom_cut = int(rh * 0.78)   # ตัดใต้พื้นรถทิ้ง

    green_mask[:top_cut, :] = 0
    green_mask[bottom_cut:, :] = 0

    # ขอบซ้าย/ขวานิดหน่อยกันเส้นกรอบรถ
    green_mask[:, :int(rw * 0.01)] = 0
    green_mask[:, int(rw * 0.99):] = 0

    # =========================
    # CLEAN MASK
    # =========================
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_mid   = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    green_mask = cv2.morphologyEx(
        green_mask,
        cv2.MORPH_OPEN,
        kernel_small,
        iterations=1
    )

    green_mask = cv2.morphologyEx(
        green_mask,
        cv2.MORPH_CLOSE,
        kernel_mid,
        iterations=1
    )

    # =========================
    # EDGE (ใช้ช่วย split เท่านั้น)
    # =========================
    edges = cv2.Canny(
        cv2.GaussianBlur(gray, (5, 5), 0),
        50,
        150
    )

    edges[:top_cut, :] = 0
    edges[bottom_cut:, :] = 0

    # combine สำหรับ "หาแนวแบ่ง"
    split_map = cv2.bitwise_or(green_mask, edges)

    debug_img = roi.copy()

    # =========================
    # STEP 1: SPLIT X
    # =========================
    col_sum = np.sum(split_map > 0, axis=0).astype(np.float32)

    if np.max(col_sum) <= 0:
        if debug:
            save_debug("debug_green_mask.jpg", green_mask)
            save_debug("debug_edges.jpg", edges)
            save_debug("debug_pallet_box.jpg", debug_img)
        return 0

    col_th = np.max(col_sum) * 0.35

    x_blocks = []
    in_block = False

    for i in range(len(col_sum)):
        if col_sum[i] > col_th and not in_block:
            start = i
            in_block = True

        elif col_sum[i] <= col_th and in_block:
            end = i
            width = end - start

            # split block ใหญ่เกิน
            if width > rw * 0.24:
                mid = (start + end) // 2
                x_blocks.append((start, mid))
                x_blocks.append((mid, end))
            else:
                x_blocks.append((start, end))

            in_block = False

    if in_block:
        x_blocks.append((start, len(col_sum) - 1))

    # =========================
    # FILTER X BLOCK
    # ตัด gap / เสาแคบ / block ที่ไม่มีเขียวจริง
    # =========================
    x_filtered = []

    for (x1, x2) in x_blocks:
        width = x2 - x1

        if width < rw * 0.06:
            continue

        sub = green_mask[:, x1:x2]
        density = np.sum(sub > 0) / (sub.size + 1e-6)

        if density < 0.02:
            continue

        x_filtered.append((x1, x2))

    x_blocks = x_filtered

    # =========================
    # SPLIT LARGE X BLOCK AGAIN
    # กันกรณี merge ติดกันหลาย rack
    # =========================
    def split_large(mask, x1, x2):
        sub = mask[:, x1:x2]
        proj = np.sum(sub > 0, axis=0).astype(np.float32)

        if np.max(proj) <= 0:
            return [(x1, x2)]

        th = np.max(proj) * 0.30
        local = []
        in_local = False

        for i in range(len(proj)):
            if proj[i] > th and not in_local:
                sx = i
                in_local = True

            elif proj[i] <= th and in_local:
                ex = i
                if (ex - sx) > rw * 0.05:
                    local.append((x1 + sx, x1 + ex))
                in_local = False

        if in_local:
            if (len(proj) - sx) > rw * 0.05:
                local.append((x1 + sx, x2))

        return local if local else [(x1, x2)]

    final_x_blocks = []

    for (x1, x2) in x_blocks:
        if (x2 - x1) > rw * 0.18:
            final_x_blocks.extend(split_large(green_mask, x1, x2))
        else:
            final_x_blocks.append((x1, x2))

    x_blocks = final_x_blocks

    # =========================
    # STEP 2: SPLIT Y
    # =========================
    pallet_boxes = []

    for (x1, x2) in x_blocks:

        sub = green_mask[:, x1:x2]
        row_sum = np.sum(sub > 0, axis=1).astype(np.float32)

        if np.max(row_sum) <= 0:
            continue

        row_th = np.max(row_sum) * 0.30
        in_row = False

        for j in range(len(row_sum)):
            if row_sum[j] > row_th and not in_row:
                sy = j
                in_row = True

            elif row_sum[j] <= row_th and in_row:
                ey = j
                h = ey - sy
                w = x2 - x1

                if h > rh * 0.10:
                    pallet_boxes.append((x1, sy, x2, ey))

                in_row = False

        if in_row:
            ey = len(row_sum) - 1
            h = ey - sy
            if h > rh * 0.10:
                pallet_boxes.append((x1, sy, x2, ey))

    # =========================
    # FINAL FILTER
    # ยืนยันว่าเป็น pallet จริง ไม่ใช่ล้อ/ช่องโล่ง/เสา
    # =========================
    final_boxes = []

    for (x1, y1, x2, y2) in pallet_boxes:
        w = x2 - x1
        h = y2 - y1

        # ขนาดต้องใกล้ pallet จริง
        if w < rw * 0.08:
            continue

        if h < rh * 0.10:
            continue

        if w > rw * 0.40:
            continue

        if h > rh * 0.65:
            continue

        # ต้องอยู่เหนือพื้นรถ
        if y2 > bottom_cut:
            continue

        sub = green_mask[y1:y2, x1:x2]
        density = np.sum(sub > 0) / (sub.size + 1e-6)

        # สำคัญ: ถ้าไม่มีเขียวพอ แค่ edge → ไม่ใช่ pallet
        if density < 0.03:
            continue

        final_boxes.append((x1, y1, x2, y2))

    # =========================
    # MERGE เฉพาะแนวตั้ง
    # ใช้กับ pallet ที่ซ้อน 2 ชั้นใน column เดียว
    # =========================
    def merge_vertical(boxes, y_dist=35):
        merged = []

        for (x1, y1, x2, y2) in boxes:
            found = False

            for i, (mx1, my1, mx2, my2) in enumerate(merged):
                overlap_x = not (x2 < mx1 or x1 > mx2)
                close_y = abs(y1 - my2) < y_dist or abs(y2 - my1) < y_dist

                if overlap_x and close_y:
                    merged[i] = (
                        min(x1, mx1),
                        min(y1, my1),
                        max(x2, mx2),
                        max(y2, my2)
                    )
                    found = True
                    break

            if not found:
                merged.append((x1, y1, x2, y2))

        return merged

    final_boxes = merge_vertical(final_boxes)

    # =========================
    # FALLBACK กัน 0
    # ถ้าถูกกรองหมด แต่ยังมี x block → ใช้ x block สร้าง box แบบ conservative
    # =========================
    if len(final_boxes) == 0 and len(x_blocks) > 0:
        for (x1, x2) in x_blocks:
            sub = green_mask[:, x1:x2]
            ys, xs = np.where(sub > 0)

            if xs.size == 0 or ys.size == 0:
                continue

            y1 = int(ys.min())
            y2 = int(ys.max())

            if (x2 - x1) < rw * 0.08:
                continue

            if (y2 - y1) < rh * 0.10:
                continue

            if y2 > bottom_cut:
                continue

            final_boxes.append((x1, y1, x2, y2))

    # =========================
    # DRAW RESULT
    # =========================
    pallet_count = 0

    for i, (x1, y1, x2, y2) in enumerate(final_boxes, 1):
        pallet_count += 1

        cv2.rectangle(
            debug_img,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            3
        )

        cv2.putText(
            debug_img,
            f"P{i}",
            (x1 + 5, y1 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    cv2.putText(
        debug_img,
        f"PALLET={pallet_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        3
    )

    print("=" * 50)
    print("FINAL PALLET =", pallet_count)
    print("=" * 50)

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

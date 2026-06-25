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

    # ROI เฉพาะตัวรถ
    roi = img[
        int(H * 0.15):int(H * 0.85),
        int(W * 0.02):int(W * 0.98)
    ]

    rh, rw = roi.shape[:2]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # =========================
    # GREEN MASK
    # =========================
    green_mask = cv2.inRange(
        hsv,
        (30, 30, 30),
        (90, 255, 255)
    )

    # ตัดส่วนหลังคา/ล้อ/คานล่าง
    top_cut = int(rh * 0.06)
    bottom_cut = int(rh * 0.78)

    green_mask[:top_cut, :] = 0
    green_mask[bottom_cut:, :] = 0

    # =========================
    # CLEAN
    # =========================
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    green_mask = cv2.morphologyEx(
        green_mask,
        cv2.MORPH_OPEN,
        k3,
        iterations=1
    )

    green_mask = cv2.morphologyEx(
        green_mask,
        cv2.MORPH_CLOSE,
        k5,
        iterations=1
    )

    # edge ใช้ช่วยหาแนวแบ่ง ไม่ใช้เป็นตัวนับหลัก
    edges = cv2.Canny(
        cv2.GaussianBlur(gray, (5, 5), 0),
        50,
        150
    )
    edges[:top_cut, :] = 0
    edges[bottom_cut:, :] = 0

    split_map = cv2.bitwise_or(green_mask, edges)

    debug_img = roi.copy()
    debug_grid = roi.copy()

    # =========================
    # หา bounding area ของ cargo จริงก่อน
    # =========================
    ys, xs = np.where(green_mask > 0)

    if xs.size == 0 or ys.size == 0:
        if debug:
            save_debug("debug_green_mask.jpg", green_mask)
            save_debug("debug_edges.jpg", edges)
            save_debug("debug_pallet_box.jpg", debug_img)
        return 0

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())

    # padding นิดหน่อย
    pad_x = int((x_max - x_min) * 0.02)
    pad_y = int((y_max - y_min) * 0.03)

    x_min = max(0, x_min - pad_x)
    x_max = min(rw - 1, x_max + pad_x)
    y_min = max(0, y_min - pad_y)
    y_max = min(rh - 1, y_max + pad_y)

    work_mask = split_map[y_min:y_max, x_min:x_max]
    work_green = green_mask[y_min:y_max, x_min:x_max]

    wh, ww = work_mask.shape[:2]

    # =========================
    # HELPER: หา segment จาก projection
    # =========================
    def find_segments(proj, th_ratio=0.25, min_len=20):
        segs = []
        if proj.size == 0 or np.max(proj) <= 0:
            return segs

        th = float(np.max(proj)) * th_ratio
        inside = False
        start = 0

        for i, v in enumerate(proj):
            if v > th and not inside:
                start = i
                inside = True
            elif v <= th and inside:
                end = i
                if (end - start) >= min_len:
                    segs.append((start, end))
                inside = False

        if inside:
            end = len(proj) - 1
            if (end - start) >= min_len:
                segs.append((start, end))

        return segs

    # =========================
    # 1) หา column หลัก
    # =========================
    col_proj = np.sum(work_mask > 0, axis=0).astype(np.float32)

    # smooth
    col_proj = cv2.GaussianBlur(col_proj.reshape(1, -1), (31, 1), 0).ravel()

    col_segments = find_segments(
        col_proj,
        th_ratio=0.22,
        min_len=max(20, int(ww * 0.06))
    )

    # fallback ถ้า segmentation ได้น้อยเกิน
    if len(col_segments) == 0:
        col_segments = [(0, ww - 1)]

    # ถ้ากว้างมาก → split เพิ่มแบบจิ๊กซอว์
    final_cols = []

    for (s, e) in col_segments:
        width = e - s

        # ประเมินความกว้าง pallet 1 ช่องแบบหยาบ
        est_col_w = max(60, int(ww * 0.16))

        if width > est_col_w * 1.8:
            ncols = int(round(width / float(est_col_w)))
            ncols = max(2, ncols)

            for i in range(ncols):
                cs = s + int(i * width / ncols)
                ce = s + int((i + 1) * width / ncols)
                final_cols.append((cs, ce))
        else:
            final_cols.append((s, e))

    col_segments = final_cols

    # =========================
    # 2) หา row หลัก (ชั้น)
    # =========================
    row_proj = np.sum(work_mask > 0, axis=1).astype(np.float32)
    row_proj = cv2.GaussianBlur(row_proj.reshape(-1, 1), (1, 21), 0).ravel()

    row_segments = find_segments(
        row_proj,
        th_ratio=0.22,
        min_len=max(18, int(wh * 0.08))
    )

    # ถ้าไม่เจอ ให้ใช้ 1 ชั้น
    if len(row_segments) == 0:
        row_segments = [(0, wh - 1)]

    # ถ้าก้อนสูงมาก → split เป็น 2 ชั้น
    final_rows = []

    for (s, e) in row_segments:
        height = e - s
        est_row_h = max(70, int(wh * 0.30))

        if height > est_row_h * 1.6:
            nrows = int(round(height / float(est_row_h)))
            nrows = max(2, nrows)

            for i in range(nrows):
                rs = s + int(i * height / nrows)
                re = s + int((i + 1) * height / nrows)
                final_rows.append((rs, re))
        else:
            final_rows.append((s, e))

    row_segments = final_rows

    # =========================
    # 3) สร้าง cell แบบจิ๊กซอว์
    # =========================
    pallet_count = 0
    cell_boxes = []

    for (rs, re) in row_segments:
        for (cs, ce) in col_segments:
            x1 = x_min + cs
            x2 = x_min + ce
            y1 = y_min + rs
            y2 = y_min + re

            w = x2 - x1
            h = y2 - y1

            # filter ขนาด
            if w < rw * 0.06:
                continue
            if h < rh * 0.10:
                continue
            if y2 > bottom_cut:
                continue

            cell_green = green_mask[y1:y2, x1:x2]
            density = np.sum(cell_green > 0) / (cell_green.size + 1e-6)

            # ต้องมีเขียวพอสมควรถึงนับ
            if density < 0.03:
                continue

            pallet_count += 1
            cell_boxes.append((x1, y1, x2, y2))

    # =========================
    # 4) fallback กันนับขาด
    # ถ้าได้น้อยเกิน 2 ให้ลองใช้ contour บน green_mask โดยตรง
    # =========================
    if pallet_count < 2:
        contour_mask = green_mask.copy()

        contours, _ = cv2.findContours(
            contour_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        fallback_boxes = []

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)

            if w < rw * 0.08:
                continue
            if h < rh * 0.10:
                continue
            if y + h > bottom_cut:
                continue

            fallback_boxes.append((x, y, x + w, y + h))

        if len(fallback_boxes) > pallet_count:
            cell_boxes = fallback_boxes
            pallet_count = len(fallback_boxes)

    # =========================
    # DRAW RESULT
    # =========================
    # วาดกรอบรวมของงาน
    cv2.rectangle(
        debug_grid,
        (x_min, y_min),
        (x_max, y_max),
        (255, 0, 255),
        2
    )

    # วาดแนว column
    for (cs, ce) in col_segments:
        x1 = x_min + cs
        x2 = x_min + ce
        cv2.rectangle(
            debug_grid,
            (x1, y_min),
            (x2, y_max),
            (255, 255, 0),
            1
        )

    # วาดแนว row
    for (rs, re) in row_segments:
        y1 = y_min + rs
        y2 = y_min + re
        cv2.rectangle(
            debug_grid,
            (x_min, y1),
            (x_max, y2),
            (0, 255, 255),
            1
        )

    for i, (x1, y1, x2, y2) in enumerate(cell_boxes, 1):
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
            (x1 + 5, y1 + 28),
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
    print("JIGSAW PALLET =", pallet_count)
    print("COL SEGMENTS =", len(col_segments))
    print("ROW SEGMENTS =", len(row_segments))
    print("=" * 50)

    if debug:
        save_debug("debug_green_mask.jpg", green_mask)
        save_debug("debug_edges.jpg", edges)
        save_debug("debug_grid.jpg", debug_grid)
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

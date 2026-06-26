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

    if img is None or img.size == 0:
        return 0

    # =========================
    # HELPERS
    # =========================
    def save_dbg(name, im):
        if debug and im is not None:
            save_debug(name, im)

    def clean(mask, k=5, close_it=1, open_it=1):
        ker = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))

        if close_it > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker, iterations=close_it)

        if open_it > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker, iterations=open_it)

        return mask

    def color_by_type(t):
        if t == "green":
            return (0, 255, 0)
        if t == "cream":
            return (0, 255, 255)
        return (0, 0, 255)

    def box_iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b

        ax2 = ax + aw
        ay2 = ay + ah
        bx2 = bx + bw
        by2 = by + bh

        ix1 = max(ax, bx)
        iy1 = max(ay, by)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)

        inter = iw * ih
        union = aw * ah + bw * bh - inter

        if union <= 0:
            return 0.0

        return inter / float(union)

    def nms(boxes, types, iou_th=0.20):
        if not boxes:
            return boxes, types

        items = []

        for i, b in enumerate(boxes):
            x, y, w, h = b
            area = w * h
            items.append((b, types[i], area))

        items = sorted(items, key=lambda v: v[2], reverse=True)

        keep_boxes = []
        keep_types = []

        for b, t, area in items:
            duplicate = False

            for old in keep_boxes:
                if box_iou(b, old) > iou_th:
                    duplicate = True
                    break

            if not duplicate:
                keep_boxes.append(b)
                keep_types.append(t)

        return keep_boxes, keep_types

    # =========================
    # RESIZE + ROI
    # =========================
    img = cv2.resize(img, (1280, 720))
    H, W = img.shape[:2]

    roi = img[
        int(H * 0.13):int(H * 0.85),
        int(W * 0.02):int(W * 0.98)
    ]

    rh, rw = roi.shape[:2]

    # =========================
    # NORMALIZE
    # =========================
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    l = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(l)

    norm = cv2.cvtColor(
        cv2.merge((l, a, b)),
        cv2.COLOR_LAB2BGR
    )

    hsv = cv2.cvtColor(norm, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)

    # =========================
    # COLOR MASKS: FRAME COLORS
    # =========================

    # green rack frame
    green_mask = cv2.inRange(
        hsv,
        (30, 30, 30),
        (95, 255, 255)
    )

    # cream / beige / gray cage frame
    cream_mask_1 = cv2.inRange(
        hsv,
        (5, 10, 70),
        (45, 170, 255)
    )

    gray_mask = cv2.inRange(
        hsv,
        (0, 0, 90),
        (180, 85, 230)
    )

    cream_mask = cv2.bitwise_or(
        cream_mask_1,
        gray_mask
    )

    # =========================
    # CUT ROOF / FLOOR / WHEEL
    # =========================
    top_cut = int(rh * 0.18)
    bottom_cut = int(rh * 0.80)

    for m in [green_mask, cream_mask]:
        m[:top_cut, :] = 0
        m[bottom_cut:, :] = 0
        m[:, :int(rw * 0.01)] = 0
        m[:, int(rw * 0.99):] = 0

    # =========================
    # CLEAN COLOR MASKS
    # =========================
    green_mask = clean(green_mask, 5, 1, 1)
    cream_mask = clean(cream_mask, 5, 1, 1)

    # =========================
    # EXTRACT FRAME / BORDER LINES
    # =========================

    def extract_frame(mask):
        # เส้นตั้ง
        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (3, max(25, int(rh * 0.12)))
        )

        vertical = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            vertical_kernel,
            iterations=1
        )

        # เส้นนอน
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(35, int(rw * 0.06)), 3)
        )

        horizontal = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            horizontal_kernel,
            iterations=1
        )

        frame = cv2.bitwise_or(vertical, horizontal)

        frame = cv2.morphologyEx(
            frame,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (11, 7)),
            iterations=2
        )

        return frame

    green_frame = extract_frame(green_mask)
    cream_frame = extract_frame(cream_mask)

    # รวมเฉพาะโครง
    frame_mask = cv2.bitwise_or(green_frame, cream_frame)

    # ใช้ edge ช่วยเฉพาะบริเวณใกล้ frame เท่านั้น
    edges = cv2.Canny(
        cv2.GaussianBlur(gray, (5, 5), 0),
        50,
        150
    )

    edges[:top_cut, :] = 0
    edges[bottom_cut:, :] = 0

    frame_dilate = cv2.dilate(
        frame_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)),
        iterations=1
    )

    edge_near_frame = cv2.bitwise_and(edges, frame_dilate)

    pallet_mask = cv2.bitwise_or(frame_mask, edge_near_frame)

    pallet_mask = cv2.morphologyEx(
        pallet_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (17, 11)),
        iterations=2
    )

    pallet_mask = cv2.morphologyEx(
        pallet_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1
    )

    # =========================
    # FIND PALLET FRAME BOXES
    # =========================
    contours, _ = cv2.findContours(
        pallet_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []
    candidate_types = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        area = w * h

        # ขนาดขั้นต่ำ
        if area < rw * rh * 0.004:
            continue

        if w < rw * 0.055:
            continue

        if h < rh * 0.080:
            continue

        if y + h > bottom_cut:
            continue

        # กันใหญ่เกินไปจนกินทั้งตู้
        if w > rw * 0.36:
            continue

        if h > rh * 0.48:
            continue

        box_area = float(max(1, w * h))

        green_ratio = cv2.countNonZero(
            green_frame[y:y + h, x:x + w]
        ) / box_area

        cream_ratio = cv2.countNonZero(
            cream_frame[y:y + h, x:x + w]
        ) / box_area

        frame_ratio = cv2.countNonZero(
            frame_mask[y:y + h, x:x + w]
        ) / box_area

        # ต้องมี frame จริง
        if frame_ratio < 0.010:
            continue

        if green_ratio >= cream_ratio:
            t = "green"
        else:
            t = "cream"

        candidates.append((x, y, w, h))
        candidate_types.append(t)

    # =========================
    # JIGSAW NORMALIZE BY SIZE
    # ขยาย/ปรับกรอบให้เป็นขนาด pallet มากขึ้น
    # =========================
    def normalize_boxes(boxes, types):
        if not boxes:
            return boxes, types

        ws = []
        hs = []

        for x, y, w, h in boxes:
            if w > rw * 0.06:
                ws.append(w)
            if h > rh * 0.09:
                hs.append(h)

        ref_w = int(np.median(ws)) if ws else int(rw * 0.13)
        ref_h = int(np.median(hs)) if hs else int(rh * 0.20)

        new_boxes = []
        new_types = []

        for i, (x, y, w, h) in enumerate(boxes):
            t = types[i]

            cx = x + w / 2
            cy = y + h / 2

            nw = w
            nh = h

            if w < ref_w * 0.60:
                nw = ref_w

            if h < ref_h * 0.60:
                nh = ref_h

            nx = int(cx - nw / 2)
            ny = int(cy - nh / 2)

            nx = max(0, nx)
            ny = max(top_cut, ny)

            if nx + nw > rw:
                nx = rw - nw

            if ny + nh > bottom_cut:
                ny = bottom_cut - nh

            nx = max(0, nx)
            ny = max(top_cut, ny)

            x1 = int(nx)
            y1 = int(ny)
            x2 = int(nx + nw)
            y2 = int(ny + nh)

            area = float(max(1, (x2 - x1) * (y2 - y1)))

            fr = cv2.countNonZero(
                frame_mask[y1:y2, x1:x2]
            ) / area

            if fr < 0.006:
                continue

            new_boxes.append((x1, y1, x2 - x1, y2 - y1))
            new_types.append(t)

        return new_boxes, new_types

    candidates, candidate_types = normalize_boxes(
        candidates,
        candidate_types
    )

    candidates, candidate_types = nms(
        candidates,
        candidate_types,
        iou_th=0.18
    )

    # =========================
    # FINAL DRAW
    # =========================
    debug_box = roi.copy()
    debug_frame = roi.copy()

    counts = {
        "green": 0,
        "cream": 0
    }

    for i, (x, y, w, h) in enumerate(candidates, 1):
        t = candidate_types[i - 1]

        if t not in counts:
            counts[t] = 0

        counts[t] += 1

        color = color_by_type(t)

        prefix = "G" if t == "green" else "C"

        cv2.rectangle(
            debug_box,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            3
        )

        cv2.putText(
            debug_box,
            f"{prefix}{counts[t]}",
            (x + 6, y + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

        cv2.rectangle(
            debug_frame,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

    total = len(candidates)

    cv2.putText(
        debug_box,
        f"T={total} G={counts['green']} C={counts['cream']}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        3,
        cv2.LINE_AA
    )

    print("=" * 50)
    print("PALLET FRAME DETECTION")
    print(f"TOTAL : {total}")
    print(f"GREEN : {counts['green']}")
    print(f"CREAM : {counts['cream']}")
    print("=" * 50)

    # =========================
    # SAVE DEBUG
    # =========================
    if debug:
        save_dbg("debug_original.jpg", roi)
        save_dbg("debug_green_mask.jpg", green_mask)
        save_dbg("debug_cream_mask.jpg", cream_mask)
        save_dbg("debug_green_frame.jpg", green_frame)
        save_dbg("debug_cream_frame.jpg", cream_frame)
        save_dbg("debug_edges.jpg", edges)
        save_dbg("debug_pallet_mask.jpg", pallet_mask)
        save_dbg("debug_frame_box.jpg", debug_frame)
        save_dbg("debug_pallet_box.jpg", debug_box)

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

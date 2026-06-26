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
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                ker,
                iterations=close_it
            )

        if open_it > 0:
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                ker,
                iterations=open_it
            )

        return mask

    def color_by_type(t):
        if t == "green":
            return (0, 255, 0)
        if t == "cream":
            return (0, 255, 255)
        if t == "wood":
            return (0, 180, 255)
        return (255, 255, 255)

    def nms_center(boxes, types, dist=0.25):
        out_boxes = []
        out_types = []

        for i, b in enumerate(boxes):
            x, y, w, h = b
            cx = x + w / 2
            cy = y + h / 2

            dup = False

            for ob in out_boxes:
                ox, oy, ow, oh = ob
                ocx = ox + ow / 2
                ocy = oy + oh / 2

                if (
                    abs(cx - ocx) < min(w, ow) * dist and
                    abs(cy - ocy) < min(h, oh) * dist
                ):
                    dup = True
                    break

            if not dup:
                out_boxes.append(b)
                out_types.append(types[i])

        return out_boxes, out_types

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
    # MATERIAL MASKS
    # สำคัญ: ไม่ใช้ white wall เป็นตัวนับหลัก
    # =========================

    # green rack
    green_mask = cv2.inRange(
        hsv,
        (30, 30, 30),
        (95, 255, 255)
    )

    # cream / beige cage
    cream_mask = cv2.inRange(
        hsv,
        (5, 15, 75),
        (45, 170, 255)
    )

    # wood / carton
    wood_mask = cv2.inRange(
        hsv,
        (5, 35, 50),
        (38, 230, 245)
    )

    # =========================
    # CUT ROOF / FLOOR / WHEEL
    # =========================
    top_cut = int(rh * 0.18)
    bottom_cut = int(rh * 0.80)

    for m in [green_mask, cream_mask, wood_mask]:
        m[:top_cut, :] = 0
        m[bottom_cut:, :] = 0
        m[:, :int(rw * 0.01)] = 0
        m[:, int(rw * 0.99):] = 0

    # =========================
    # CLEAN MASKS
    # =========================
    green_mask = clean(green_mask, 5, 1, 1)
    cream_mask = clean(cream_mask, 5, 1, 1)
    wood_mask = clean(wood_mask, 5, 1, 1)

    material_mask = cv2.bitwise_or(green_mask, cream_mask)
    material_mask = cv2.bitwise_or(material_mask, wood_mask)

    # =========================
    # EDGE ใช้ช่วย block เท่านั้น
    # =========================
    edges = cv2.Canny(
        cv2.GaussianBlur(gray, (5, 5), 0),
        50,
        150
    )

    edges[:top_cut, :] = 0
    edges[bottom_cut:, :] = 0

    # ใช้ edge เฉพาะบริเวณที่มี material ใกล้เคียง
    material_dilate = cv2.dilate(
        material_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
        iterations=1
    )

    edge_on_material = cv2.bitwise_and(edges, material_dilate)

    cargo_mask = cv2.bitwise_or(material_mask, edge_on_material)

    cargo_mask = cv2.morphologyEx(
        cargo_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (15, 9)),
        iterations=2
    )

    cargo_mask = cv2.morphologyEx(
        cargo_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1
    )

    # =========================
    # FIND BLOCKS จาก material จริง
    # =========================
    contours, _ = cv2.findContours(
        cargo_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    blocks = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        area = w * h

        if area < rw * rh * 0.004:
            continue

        if w < rw * 0.045:
            continue

        if h < rh * 0.070:
            continue

        if y + h > bottom_cut:
            continue

        # กัน block ใหญ่ครอบผนัง/เพดาน
        if h > rh * 0.60:
            continue

        # ต้องมี material จริง ไม่ใช่ edge อย่างเดียว
        block_material = material_mask[y:y + h, x:x + w]
        material_density = cv2.countNonZero(block_material) / float(max(1, w * h))

        if material_density < 0.012:
            continue

        blocks.append((x, y, w, h))

    blocks = sorted(blocks, key=lambda b: (b[1], b[0]))

    # =========================
    # SPLIT BLOCK เป็น cell
    # =========================
    cells = []
    cell_types = []

    debug_box = roi.copy()
    debug_grid = roi.copy()

    for (x, y, w, h) in blocks:

        block_area = float(max(1, w * h))

        green_ratio = cv2.countNonZero(
            green_mask[y:y + h, x:x + w]
        ) / block_area

        cream_ratio = cv2.countNonZero(
            cream_mask[y:y + h, x:x + w]
        ) / block_area

        wood_ratio = cv2.countNonZero(
            wood_mask[y:y + h, x:x + w]
        ) / block_area

        if green_ratio >= max(cream_ratio, wood_ratio):
            block_type = "green"
        elif cream_ratio >= max(green_ratio, wood_ratio):
            block_type = "cream"
        else:
            block_type = "wood"

        # =========================
        # ESTIMATE GRID
        # =========================
        if block_type == "green":
            est_cell_w = rw * 0.135
            est_cell_h = rh * 0.28
        elif block_type == "cream":
            est_cell_w = rw * 0.125
            est_cell_h = rh * 0.25
        else:
            est_cell_w = rw * 0.140
            est_cell_h = rh * 0.28

        cols = int(round(w / max(1.0, est_cell_w)))
        rows = int(round(h / max(1.0, est_cell_h)))

        cols = max(1, min(6, cols))
        rows = max(1, min(3, rows))

        if w > rw * 0.25 and cols < 2:
            cols = 2

        if h > rh * 0.35 and rows < 2:
            rows = 2

        cv2.rectangle(
            debug_grid,
            (x, y),
            (x + w, y + h),
            color_by_type(block_type),
            3
        )

        cv2.putText(
            debug_grid,
            f"{block_type} {cols}x{rows}",
            (x + 5, max(25, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color_by_type(block_type),
            2,
            cv2.LINE_AA
        )

        # =========================
        # CREATE CELLS
        # =========================
        for r in range(rows):
            for c in range(cols):
                x1 = x + int(c * w / cols)
                x2 = x + int((c + 1) * w / cols)
                y1 = y + int(r * h / rows)
                y2 = y + int((r + 1) * h / rows)

                cw = x2 - x1
                ch = y2 - y1

                if cw < rw * 0.045:
                    continue

                if ch < rh * 0.060:
                    continue

                if y2 > bottom_cut:
                    continue

                cell_area = float(max(1, cw * ch))

                cell_green = cv2.countNonZero(
                    green_mask[y1:y2, x1:x2]
                ) / cell_area

                cell_cream = cv2.countNonZero(
                    cream_mask[y1:y2, x1:x2]
                ) / cell_area

                cell_wood = cv2.countNonZero(
                    wood_mask[y1:y2, x1:x2]
                ) / cell_area

                # ห้ามนับ cell ที่ไม่มี material จริง
                material_score = max(cell_green, cell_cream, cell_wood)

                if material_score < 0.012:
                    continue

                if cell_green >= max(cell_cream, cell_wood):
                    t = "green"
                elif cell_cream >= max(cell_green, cell_wood):
                    t = "cream"
                else:
                    t = "wood"

                cells.append((x1, y1, cw, ch))
                cell_types.append(t)

                cv2.rectangle(
                    debug_grid,
                    (x1, y1),
                    (x2, y2),
                    color_by_type(t),
                    1
                )

    # =========================
    # REMOVE DUP
    # =========================
    cells, cell_types = nms_center(cells, cell_types, dist=0.25)

    # =========================
    # FINAL DRAW
    # =========================
    counts = {
        "green": 0,
        "cream": 0,
        "wood": 0,
        "unknown": 0
    }

    for i, (x, y, w, h) in enumerate(cells, 1):
        t = cell_types[i - 1]

        if t not in counts:
            counts[t] = 0

        counts[t] += 1

        if t == "green":
            prefix = "G"
        elif t == "cream":
            prefix = "C"
        elif t == "wood":
            prefix = "W"
        else:
            prefix = "U"

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

    total = len(cells)

    cv2.putText(
        debug_box,
        f"T={total} G={counts['green']} C={counts['cream']} W={counts['wood']}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        3,
        cv2.LINE_AA
    )

    print("=" * 50)
    print("PALLET MULTI-COLOR MATERIAL GRID")
    print(f"BLOCKS : {len(blocks)}")
    print(f"TOTAL  : {total}")
    print(f"GREEN  : {counts['green']}")
    print(f"CREAM  : {counts['cream']}")
    print(f"WOOD   : {counts['wood']}")
    print("=" * 50)

    # =========================
    # DEBUG SAVE
    # =========================
    if debug:
        save_dbg("debug_original.jpg", roi)
        save_dbg("debug_normalized.jpg", norm)
        save_dbg("debug_green_mask.jpg", green_mask)
        save_dbg("debug_cream_mask.jpg", cream_mask)
        save_dbg("debug_wood_mask.jpg", wood_mask)
        save_dbg("debug_edges.jpg", edges)
        save_dbg("debug_cargo.jpg", cargo_mask)
        save_dbg("debug_grid.jpg", debug_grid)
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

from flask import Flask, request, jsonify, send_file
import requests
import cv2
import numpy as np
import traceback
import time
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
            allow_redirects=True,
            headers={
                "User-At": "Mozilla/5.0"
            }
        )

        if r.status_code != 200:
            print("IMAGE HTTP ERROR:", r.status_code)
            return None

        img = cv2.imdecode(
            np.frombuffer(r.content, np.uint8),
            cv2.IMREAD_COLOR
        )

        return img

    except Exception as e:
        print("DOWNLOAD ERROR:", e)
        return None


# =========================
# DEBUG SAVE
# =========================
def save_debug(filename, img):
    try:
        path = os.path.join(DEBUG_DIR, filename)
        ok = cv2.imwrite(path, img)
        print(f"SAVE DEBUG {filename}: {ok}")
        return ok

    except Exception as e:
        print("SAVE DEBUG ERROR:", filename, e)
        return False


# =========================
# CLEAN MASK
# =========================
def clean_mask(mask, min_area_ratio=0.002):
    if mask is None or mask.size == 0:
        return mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    result = np.zeros_like(mask)

    min_area = int(mask.size * min_area_ratio)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area > min_area:
            result[labels == i] = 255

    return result



# =========================
# BALANCED VOLUME MODEL
# =========================
def gen_volume(img, debug=True, return_empty=False):

    if img is None or img.size == 0:
        return 0

    # =========================
    # DETECT VIEW TYPE
    # =========================
    orig_h, orig_w = img.shape[:2]
    view_type = "rear" if orig_h > orig_w else "side"

    # =========================
    # RESIZE
    # =========================
    img = cv2.resize(img, (640, 480))

    # =========================
    # SIDE VIEW
    # 4:3 -> 16:9
    # =========================
    if view_type == "side":
        h, w = img.shape[:2]
        target_h = int(w * 9 / 16)
        top = max(0, (h - target_h) // 2)
        img = img[top:top + target_h, :]

    h, w = img.shape[:2]

    print(
        f"VIEW={view_type} "
        f"SIZE={w}x{h}"
    )

    # =========================
    # ROI
    # =========================
    if view_type == "rear":
        roi = img[
            int(h * 0.18):int(h * 0.82),
            int(w * 0.15):int(w * 0.85)
        ]
    else:
        roi = img[
            int(h * 0.25):int(h * 0.75),
            int(w * 0.15):int(w * 0.85)
        ]

    if roi.size == 0:
        return 0

    rh, rw = roi.shape[:2]

    # =========================
    # CONTAINER MASK
    # ใช้ ROI ทั้งหมดเป็นพื้นที่ภายในตู้
    # =========================
    container_mask = np.full(
        (rh, rw),
        255,
        dtype=np.uint8
    )

    # =========================
    # LIGHT NORMALIZATION
    # =========================
    lab = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    roi_norm = cv2.cvtColor(
        cv2.merge((l, a, b)),
        cv2.COLOR_LAB2BGR
    )

    # =========================
    # COLOR SPACE
    # =========================
    hsv = cv2.cvtColor(
        roi_norm,
        cv2.COLOR_BGR2HSV
    )

    gray = cv2.cvtColor(
        roi_norm,
        cv2.COLOR_BGR2GRAY
    )

    gray_blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    h_channel, s_channel, v_channel = cv2.split(hsv)

    v_mean = float(v_channel.mean())
    s_mean = float(s_channel.mean())

    # =========================
    # CARGO COLOR MASKS
    # =========================

    # GREEN PALLET / GREEN OBJECT
    green_mask = cv2.inRange(
        hsv,
        (35, 45, 45),
        (95, 255, 255)
    )

    # BROWN CARTON / WOOD / PALLET
    brown_mask = cv2.inRange(
        hsv,
        (5, 45, 45),
        (35, 255, 230)
    )

    # BLUE / CYAN CRATE
    blue_mask = cv2.inRange(
        hsv,
        (85, 35, 35),
        (125, 255, 255)
    )

    # RED CRATE / RED CARGO
    # สีแดงใน HSV ต้องแยก 2 ช่วง เพราะ Hue อยู่ทั้งต้นและท้ายวงสี
    red_mask_1 = cv2.inRange(
        hsv,
        (0, 60, 50),
        (12, 255, 255)
    )

    red_mask_2 = cv2.inRange(
        hsv,
        (165, 60, 50),
        (180, 255, 255)
    )

    red_mask = cv2.bitwise_or(
        red_mask_1,
        red_mask_2
    )

    # DARK CARGO
    # ปรับเข้มขึ้นเพื่อลดการจับเงา / ผ้า / พื้นมืด
    dark_mask = cv2.inRange(
        hsv,
        (0, 55, 0),
        (180, 255, 65)
    )

    # =========================
    # TEXTURE MASK - CONSERVATIVE
    # =========================
    adaptive_texture = cv2.adaptiveThreshold(
        gray_blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7
    )

    # ใช้ AND แทน OR เพื่อลดการจับผนัง / ผ้า / เพดาน
    strong_saturation_mask = cv2.inRange(
        s_channel,
        70,
        255
    )

    strong_low_value_mask = cv2.inRange(
        v_channel,
        0,
        75
    )

    texture_candidate = cv2.bitwise_and(
        strong_saturation_mask,
        strong_low_value_mask
    )

    texture_mask = cv2.bitwise_and(
        adaptive_texture,
        texture_candidate
    )

    # =========================
    # EDGE DENSITY FILTER
    # ลด false positive จากผ้า / ผิวเรียบ / เพดาน
    # =========================
    edges = cv2.Canny(
        gray_blur,
        40,
        120
    )

    edge_density = cv2.blur(
        edges.astype(np.float32),
        (15, 15)
    )

    edge_mask = cv2.inRange(
        edge_density,
        10,
        255
    )

    texture_mask = cv2.bitwise_and(
        texture_mask,
        edge_mask
    )

    # =========================
    # TOP FALSE POSITIVE SUPPRESSION
    # ไม่ตัด color cargo ด้านบนทั้งหมด เพราะบางภาพมีกล่องจริงอยู่ด้านบน
    # ตัดเฉพาะ texture/dark ที่มักติดเพดานหรือผ้า
    # =========================
    top_suppress_mask = np.full(
        (rh, rw),
        255,
        dtype=np.uint8
    )

    top_cut_ratio = 0.12 if view_type == "rear" else 0.16
    top_cut = int(rh * top_cut_ratio)

    top_suppress_mask[:top_cut, :] = 0

    texture_mask = cv2.bitwise_and(
        texture_mask,
        top_suppress_mask
    )

    dark_mask = cv2.bitwise_and(
        dark_mask,
        top_suppress_mask
    )

    # =========================
    # COMBINE COLOR MASKS
    # =========================
    color_cargo_mask = cv2.bitwise_or(
        green_mask,
        brown_mask
    )

    color_cargo_mask = cv2.bitwise_or(
        color_cargo_mask,
        blue_mask
    )

    color_cargo_mask = cv2.bitwise_or(
        color_cargo_mask,
        red_mask
    )

    # =========================
    # COMBINE CARGO
    # =========================
    cargo_mask = cv2.bitwise_or(
        color_cargo_mask,
        dark_mask
    )

    cargo_mask = cv2.bitwise_or(
        cargo_mask,
        texture_mask
    )

    cargo_mask = cv2.bitwise_and(
        cargo_mask,
        container_mask
    )

    # =========================
    # MORPHOLOGY
    # =========================
    kernel_small = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5)
    )

    kernel_medium = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (9, 9)
    )

    cargo_mask = cv2.morphologyEx(
        cargo_mask,
        cv2.MORPH_CLOSE,
        kernel_medium,
        iterations=1
    )

    cargo_mask = cv2.morphologyEx(
        cargo_mask,
        cv2.MORPH_OPEN,
        kernel_small,
        iterations=1
    )

    # ลด noise จุดเล็ก
    cargo_mask = clean_mask(
        cargo_mask,
        min_area_ratio=0.005
    )

    # =========================
    # CONTOUR FILTER
    # กรอง noise เล็ก ๆ และ shape ที่ไม่น่าเป็น cargo
    # =========================
    contours, _ = cv2.findContours(
        cargo_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    filtered_mask = np.zeros_like(cargo_mask)

    min_contour_area = rh * rw * 0.0035

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < min_contour_area:
            continue

        x_box, y_box, w_box, h_box = cv2.boundingRect(cnt)

        if h_box <= 0 or w_box <= 0:
            continue

        aspect_ratio = w_box / float(h_box)

        # ช่วงกว้างขึ้นเพื่อไม่ตัดกล่องที่เรียงยาวหลายใบ
        if 0.20 <= aspect_ratio <= 6.50:
            cv2.drawContours(
                filtered_mask,
                [cnt],
                -1,
                255,
                thickness=-1
            )

    cargo_mask = filtered_mask

    # =========================
    # FALLBACK ถ้า cargo กินภาพเยอะผิดปกติ
    # =========================
    raw_cargo_ratio = cv2.countNonZero(cargo_mask) / float(container_mask.size)

    if raw_cargo_ratio > 0.95:
        print("WARNING: cargo over-detected, fallback to color only")

        cargo_mask = color_cargo_mask.copy()

        cargo_mask = cv2.bitwise_or(
            cargo_mask,
            dark_mask
        )

        cargo_mask = cv2.bitwise_and(
            cargo_mask,
            container_mask
        )

        cargo_mask = cv2.morphologyEx(
            cargo_mask,
            cv2.MORPH_CLOSE,
            kernel_medium,
            iterations=1
        )

        cargo_mask = cv2.morphologyEx(
            cargo_mask,
            cv2.MORPH_OPEN,
            kernel_small,
            iterations=1
        )

        cargo_mask = clean_mask(
            cargo_mask,
            min_area_ratio=0.005
        )

        raw_cargo_ratio = cv2.countNonZero(cargo_mask) / float(container_mask.size)

    # =========================
    # EMPTY MASK = CONTAINER - CARGO
    # =========================
    empty_mask = cv2.bitwise_and(
        container_mask,
        cv2.bitwise_not(cargo_mask)
    )

    # =========================
    # PERSPECTIVE WEIGHT
    # =========================
    y = np.linspace(
        0,
        1,
        rh,
        dtype=np.float32
    ).reshape(rh, 1)

    if view_type == "rear":
        weights = 0.70 + (y ** 1.5) * 1.30
    else:
        weights = 0.80 + (y ** 1.3) * 1.10

    container_score = np.sum(
        (container_mask > 0).astype(np.float32) * weights
    )

    cargo_score = np.sum(
        (cargo_mask > 0).astype(np.float32) * weights
    )

    empty_score = np.sum(
        (empty_mask > 0).astype(np.float32) * weights
    )

    if container_score <= 1e-6:
        return 0

    filled_ratio = cargo_score / container_score
    empty_ratio = empty_score / container_score

    filled_ratio = float(
        np.clip(
            filled_ratio,
            0,
            1
        )
    )

    empty_ratio = float(
        np.clip(
            empty_ratio,
            0,
            1
        )
    )

    # =========================
    # CALIBRATION
    # =========================
    filled_volume = (filled_ratio ** 0.95) * 100
    filled_volume = filled_volume * 0.95

    filled_volume = float(
        np.clip(
            filled_volume,
            0,
            100
        )
    )

    empty_volume = 100 - filled_volume

    if return_empty:
        output_volume = empty_volume
    else:
        output_volume = filled_volume

    output_volume = int(round(output_volume / 5) * 5)
    output_volume = max(0, min(100, output_volume))

    print(
        f"VIEW={view_type} "
        f"VMEAN={v_mean:.1f} "
        f"SMEAN={s_mean:.1f} "
        f"RAW_CARGO={raw_cargo_ratio:.3f} "
        f"FILLED_RATIO={filled_ratio:.3f} "
        f"EMPTY_RATIO={empty_ratio:.3f} "
        f"RETURN={output_volume}% "
        f"MODE={'EMPTY' if return_empty else 'FILLED'}"
    )

    # =========================
    # DEBUG OUTPUT
    # =========================
    if debug:

        color_layer = roi_norm.copy()

        # GREEN = cargo
        color_layer[cargo_mask > 0] = (
            0,
            255,
            0
        )

        # BLUE = empty
        # เปลี่ยนจากสีแดงเป็นสีน้ำเงิน เพื่อไม่ให้กลืนกับลังแดงจริง
        color_layer[empty_mask > 0] = (
            255,
            0,
            0
        )

        overlay = cv2.addWeighted(
            roi_norm,
            0.85,
            color_layer,
            0.15,
            0
        )

        cargo_contours, _ = cv2.findContours(
            cargo_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # YELLOW contour = cargo
        cv2.drawContours(
            overlay,
            cargo_contours,
            -1,
            (0, 255, 255),
            2
        )

        empty_contours, _ = cv2.findContours(
            empty_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # BLUE contour = empty
        cv2.drawContours(
            overlay,
            empty_contours,
            -1,
            (255, 0, 0),
            1
        )

        overlay_light = cv2.addWeighted(
            roi_norm,
            0.92,
            color_layer,
            0.08,
            0
        )

        overlay_contour = roi_norm.copy()

        cv2.drawContours(
            overlay_contour,
            cargo_contours,
            -1,
            (0, 255, 255),
            2
        )

        cv2.drawContours(
            overlay_contour,
            empty_contours,
            -1,
            (255, 0, 0),
            1
        )

        save_debug("debug_original.jpg", roi)
        save_debug("debug_normalized.jpg", roi_norm)
        save_debug("debug_container.jpg", container_mask)
        save_debug("debug_cargo.jpg", cargo_mask)
        save_debug("debug_empty.jpg", empty_mask)
        save_debug("debug_overlay.jpg", overlay)
        save_debug("debug_overlay_light.jpg", overlay_light)
        save_debug("debug_overlay_contour.jpg", overlay_contour)

        save_debug("debug_green.jpg", green_mask)
        save_debug("debug_brown.jpg", brown_mask)
        save_debug("debug_blue.jpg", blue_mask)
        save_debug("debug_dark.jpg", dark_mask)
        save_debug("debug_texture.jpg", texture_mask)

    return output_volume


def gen_pallet(img, debug=True):
    cells = []
    cell_types = []

    def save_dbg(name, im):
        if not debug or im is None:
            return
        try:
            save_debug(name, im)
        except Exception:
            cv2.imwrite(os.path.join(DEBUG_DIR, name), im)

    def clean(mask, k=5, it=1):
        if mask is None or mask.size == 0:
            return mask

        ker = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (k, k)
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            ker,
            iterations=it
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            ker,
            iterations=1
        )

        return mask

    def add_cell(box, cell_type):
        cells.append(box)
        cell_types.append(cell_type)

    def union_blocks(blocks):
        if len(blocks) == 0:
            return None

        xs = [b[0] for b in blocks]
        ys = [b[1] for b in blocks]
        xes = [b[0] + b[2] for b in blocks]
        yes = [b[1] + b[3] for b in blocks]

        ux = min(xs)
        uy = min(ys)
        ux2 = max(xes)
        uy2 = max(yes)

        return (
            ux,
            uy,
            ux2 - ux,
            uy2 - uy
        )

    def nms_with_types(boxes, types, dist=0.05):
        out_boxes = []
        out_types = []

        for i, b in enumerate(boxes):
            x, y, w, h = b
            cx = x + w / 2.0
            cy = y + h / 2.0

            duplicate = False

            for ob in out_boxes:
                ox, oy, ow, oh = ob
                ocx = ox + ow / 2.0
                ocy = oy + oh / 2.0

                if (
                    abs(cx - ocx) < min(w, ow) * dist and
                    abs(cy - ocy) < min(h, oh) * dist
                ):
                    duplicate = True
                    break

            if not duplicate:
                out_boxes.append(b)

                if i < len(types):
                    out_types.append(types[i])
                else:
                    out_types.append("pallet")

        return out_boxes, out_types

    def draw_grid(draw, x, y, w, h, cols=4, rows=2):
        for c in range(cols + 1):
            gx = x + int(c * w / cols)
            cv2.line(
                draw,
                (gx, y),
                (gx, y + h),
                (255, 255, 0),
                1
            )

        for r in range(rows + 1):
            gy = y + int(r * h / rows)
            cv2.line(
                draw,
                (x, gy),
                (x + w, gy),
                (255, 255, 0),
                1
            )

        cv2.putText(
            draw,
            f"{cols}x{rows}",
            (x + 5, max(25, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (255, 255, 0),
            2,
            cv2.LINE_AA
        )

    def color_by_type(cell_type):
        if cell_type == "carton":
            return (0, 180, 255)
        if cell_type == "green":
            return (255, 255, 0)
        if cell_type == "cream":
            return (0, 255, 0)
        return (0, 255, 255)

    def split_4x2(box, cell_type, draw):
        x, y, w, h = box

        if w <= 0 or h <= 0:
            return 0

        cols = 4
        rows = 2
        count = 0

        draw_grid(
            draw,
            x,
            y,
            w,
            h,
            cols,
            rows
        )

        color = color_by_type(cell_type)

        for r in range(rows):
            for c in range(cols):
                x1 = x + int(c * w / cols)
                x2 = x + int((c + 1) * w / cols)
                y1 = y + int(r * h / rows)
                y2 = y + int((r + 1) * h / rows)

                cw = x2 - x1
                ch = y2 - y1

                if cw < 30 or ch < 30:
                    continue

                add_cell(
                    (x1, y1, cw, ch),
                    cell_type
                )

                cv2.rectangle(
                    draw,
                    (x1, y1),
                    (x2, y2),
                    color,
                    1
                )

                count += 1

        return count

    def force_layout_count(raw_blocks, draw):
        forced_cells = []
        forced_types = []

        if len(raw_blocks) == 0:
            return forced_cells, forced_types

        sorted_blocks = sorted(
            raw_blocks,
            key=lambda b: b[0]
        )

        for idx, box in enumerate(sorted_blocks):
            x, y, w, h = box

            if w <= 0 or h <= 0:
                continue

            if idx == 0:
                cell_type = "carton"
            elif idx == 1:
                cell_type = "green"
            else:
                cell_type = "cream"

            cols = 4
            rows = 2

            draw_grid(
                draw,
                x,
                y,
                w,
                h,
                cols,
                rows
            )

            color = color_by_type(cell_type)

            for r in range(rows):
                for c in range(cols):
                    x1 = x + int(c * w / cols)
                    x2 = x + int((c + 1) * w / cols)
                    y1 = y + int(r * h / rows)
                    y2 = y + int((r + 1) * h / rows)

                    forced_cells.append(
                        (x1, y1, x2 - x1, y2 - y1)
                    )

                    forced_types.append(cell_type)

                    cv2.rectangle(
                        draw,
                        (x1, y1),
                        (x2, y2),
                        color,
                        1
                    )

        return forced_cells, forced_types

    if img is not None and img.size > 0:
        img = cv2.resize(
            img,
            (1280, 720)
        )

        H, W = img.shape[:2]

        # ROI ใช้ตามที่ตรวจแล้วว่าใช้ได้
        roi = img[
            int(H * 0.17):int(H * 0.84),
            int(W * 0.01):int(W * 0.99)
        ]

        if roi.size > 0:
            rh, rw = roi.shape[:2]

            # =========================
            # LIGHT NORMALIZATION
            # =========================
            lab = cv2.cvtColor(
                roi,
                cv2.COLOR_BGR2LAB
            )

            l, a, b = cv2.split(lab)

            clahe = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8, 8)
            )

            l = clahe.apply(l)

            norm = cv2.cvtColor(
                cv2.merge((l, a, b)),
                cv2.COLOR_LAB2BGR
            )

            hsv = cv2.cvtColor(
                norm,
                cv2.COLOR_BGR2HSV
            )

            gray = cv2.cvtColor(
                norm,
                cv2.COLOR_BGR2GRAY
            )

            # =========================
            # COLOR MASKS
            # =========================
            cream = cv2.inRange(
                hsv,
                (8, 18, 70),
                (42, 160, 250)
            )

            white = cv2.inRange(
                hsv,
                (0, 0, 145),
                (180, 60, 255)
            )

            green = cv2.inRange(
                hsv,
                (38, 40, 40),
                (95, 255, 235)
            )

            blue = cv2.inRange(
                hsv,
                (85, 35, 35),
                (135, 255, 255)
            )

            carton = cv2.inRange(
                hsv,
                (7, 22, 65),
                (38, 200, 250)
            )

            # =========================
            # CUT ROOF / BOTTOM / EDGES
            # =========================
            for m in [cream, white, green, blue, carton]:
                m[:int(rh * 0.13), :] = 0
                m[int(rh * 0.88):, :] = 0
                m[:, :int(rw * 0.010)] = 0
                m[:, int(rw * 0.990):] = 0

            cream = clean(cream, 3, 1)
            white = clean(white, 3, 1)
            green = clean(green, 3, 1)
            blue = clean(blue, 3, 1)
            carton_mask = clean(carton, 5, 1)

            # =========================
            # BUILD PALLET MASK
            # =========================
            frame_mask = cv2.bitwise_or(
                cream,
                white
            )

            frame_mask = cv2.bitwise_or(
                frame_mask,
                green
            )

            frame_mask = cv2.bitwise_or(
                frame_mask,
                blue
            )

            frame_mask = clean(
                frame_mask,
                3,
                1
            )

            edges = cv2.Canny(
                cv2.GaussianBlur(gray, (5, 5), 0),
                55,
                150
            )

            edge_base = cv2.bitwise_or(
                frame_mask,
                carton_mask
            )

            edge_mask = cv2.bitwise_and(
                edges,
                edge_base
            )

            pallet_mask = cv2.bitwise_or(
                frame_mask,
                carton_mask
            )

            pallet_mask = cv2.bitwise_or(
                pallet_mask,
                edge_mask
            )

            pallet_mask = cv2.morphologyEx(
                pallet_mask,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                iterations=1
            )

            pallet_mask = cv2.morphologyEx(
                pallet_mask,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
                iterations=1
            )

            pallet_mask[:int(rh * 0.13), :] = 0
            pallet_mask[int(rh * 0.88):, :] = 0
            pallet_mask[:, :int(rw * 0.010)] = 0
            pallet_mask[:, int(rw * 0.990):] = 0

            cargo_mask = cv2.morphologyEx(
                pallet_mask,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (13, 9)),
                iterations=1
            )

            cargo_mask = cv2.morphologyEx(
                cargo_mask,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                iterations=1
            )

            # =========================
            # DEBUG CANVAS
            # =========================
            debug_blocks = roi.copy()
            debug_grid = roi.copy()
            debug_count = roi.copy()

            color = roi.copy()
            color[pallet_mask > 0] = (0, 255, 0)
            color[carton_mask > 0] = (0, 180, 255)

            debug_overlay = cv2.addWeighted(
                roi,
                0.75,
                color,
                0.25,
                0
            )

            # =========================
            # FIND RAW BLOCKS
            # =========================
            cnts, _ = cv2.findContours(
                cargo_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            raw_blocks = []

            for cnt in cnts:
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h

                if area < rw * rh * 0.002:
                    continue

                if w < rw * 0.025 or h < rh * 0.060:
                    continue

                if y > rh * 0.82:
                    continue

                raw_blocks.append(
                    (x, y, w, h)
                )

            # fallback ถ้า contour ไม่เจอ
            if len(raw_blocks) == 0:
                ys, xs = np.where(pallet_mask > 0)

                if xs.size > 0 and ys.size > 0:
                    x1 = int(xs.min())
                    x2 = int(xs.max())
                    y1 = int(ys.min())
                    y2 = int(ys.max())

                    if (x2 - x1) > rw * 0.20 and (y2 - y1) > rh * 0.20:
                        raw_blocks.append(
                            (
                                x1,
                                y1,
                                x2 - x1,
                                y2 - y1
                            )
                        )

            raw_blocks = sorted(
                raw_blocks,
                key=lambda b: (b[1], b[0])
            )

            # =========================
            # DRAW RAW BLOCKS
            # =========================
            for i, (x, y, w, h) in enumerate(raw_blocks, 1):
                cv2.rectangle(
                    debug_blocks,
                    (x, y),
                    (x + w, y + h),
                    (255, 0, 255),
                    2
                )

                cv2.putText(
                    debug_blocks,
                    f"B{i}",
                    (x + 5, y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.70,
                    (255, 0, 255),
                    2,
                    cv2.LINE_AA
                )

            # =========================
            # CLASSIFY BLOCKS
            # =========================
            carton_blocks = []
            green_blocks = []
            cream_blocks = []

            for x, y, w, h in raw_blocks:
                area = float(max(1, w * h))

                carton_ratio = cv2.countNonZero(
                    carton_mask[y:y + h, x:x + w]
                ) / area

                green_ratio = (
                    cv2.countNonZero(green[y:y + h, x:x + w]) +
                    cv2.countNonZero(blue[y:y + h, x:x + w])
                ) / area

                cream_ratio = (
                    cv2.countNonZero(cream[y:y + h, x:x + w]) +
                    cv2.countNonZero(white[y:y + h, x:x + w])
                ) / area

                pallet_ratio = cv2.countNonZero(
                    pallet_mask[y:y + h, x:x + w]
                ) / area

                if carton_ratio > 0.16 and x < rw * 0.45:
                    carton_blocks.append(
                        (x, y, w, h)
                    )
                    continue

                if green_ratio > 0.012 and x < rw * 0.70:
                    green_blocks.append(
                        (x, y, w, h)
                    )
                    continue

                if cream_ratio > 0.008 or pallet_ratio > 0.008:
                    cream_blocks.append(
                        (x, y, w, h)
                    )

            carton_box = union_blocks(carton_blocks)
            green_box = union_blocks(green_blocks)
            cream_box = union_blocks(cream_blocks)

            final_blocks = []

            if carton_box is not None:
                final_blocks.append(
                    ("carton", carton_box)
                )

            if green_box is not None:
                final_blocks.append(
                    ("green", green_box)
                )

            if cream_box is not None:
                final_blocks.append(
                    ("cream", cream_box)
                )

            # ถ้า classify ไม่ได้ แต่มี raw block ให้ใช้ raw block แทน
            if len(final_blocks) == 0 and len(raw_blocks) > 0:
                sorted_by_x = sorted(
                    raw_blocks,
                    key=lambda b: b[0]
                )

                for idx, box in enumerate(sorted_by_x):
                    if idx == 0:
                        t = "carton"
                    elif idx == 1:
                        t = "green"
                    else:
                        t = "cream"

                    final_blocks.append(
                        (t, box)
                    )

            # =========================
            # DRAW FINAL BLOCKS
            # =========================
            for i, (block_type, box) in enumerate(final_blocks, 1):
                x, y, w, h = box
                color_b = color_by_type(block_type)

                cv2.rectangle(
                    debug_blocks,
                    (x, y),
                    (x + w, y + h),
                    color_b,
                    3
                )

                cv2.putText(
                    debug_blocks,
                    f"{block_type[:1].upper()}{i}",
                    (x + 5, y + 52),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.70,
                    color_b,
                    2,
                    cv2.LINE_AA
                )

            # =========================
            # COUNT 4x2 FOR EVERY BLOCK
            # =========================
            for block_type, box in final_blocks:
                split_4x2(
                    box,
                    block_type,
                    debug_grid
                )

            # =========================
            # FORCE FALLBACK
            # ถ้ายังนับได้น้อย ให้แตก raw_blocks เป็น 4x2 ทุก block
            # =========================
            if len(cells) <= 3:
                print("WARNING: LOW COUNT, FORCE 4x2 RAW BLOCK SPLIT")

                cells = []
                cell_types = []

                forced_cells, forced_types = force_layout_count(
                    raw_blocks,
                    debug_grid
                )

                cells.extend(forced_cells)
                cell_types.extend(forced_types)

            # =========================
            # FINAL NMS
            # ลดมาก เพื่อไม่ให้ลบช่องที่ติดกัน
            # =========================
            cells, cell_types = nms_with_types(
                cells,
                cell_types,
                dist=0.05
            )

            cream_count = sum(
                1 for t in cell_types
                if t == "cream"
            )

            green_count = sum(
                1 for t in cell_types
                if t == "green"
            )

            carton_count = sum(
                1 for t in cell_types
                if t == "carton"
            )

            # =========================
            # DRAW FINAL COUNT
            # =========================
            for i, (x, y, w, h) in enumerate(cells, 1):
                t = cell_types[i - 1] if i - 1 < len(cell_types) else "pallet"

                color_f = color_by_type(t)

                if t == "carton":
                    label = f"P{i}"
                elif t == "green":
                    label = f"G{i}"
                elif t == "cream":
                    label = f"C{i}"
                else:
                    label = f"N{i}"

                cv2.rectangle(
                    debug_count,
                    (x, y),
                    (x + w, y + h),
                    color_f,
                    3
                )

                cv2.putText(
                    debug_count,
                    label,
                    (x + 7, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    color_f,
                    2,
                    cv2.LINE_AA
                )

            cv2.putText(
                debug_count,
                f"C={cream_count} G={green_count} P={carton_count} TOTAL={len(cells)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.95,
                (0, 255, 255),
                3,
                cv2.LINE_AA
            )

            cv2.putText(
                debug_blocks,
                f"RAW={len(raw_blocks)} FINAL={len(final_blocks)} EXPECT={len(final_blocks) * 8}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.95,
                (255, 0, 255),
                3,
                cv2.LINE_AA
            )

            cv2.putText(
                debug_grid,
                "4x2 EVERY BLOCK",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 0),
                3,
                cv2.LINE_AA
            )

            print("=" * 50)
            print("PALLET COUNT DEBUG 4x2 ALL")
            print(f"ROI SIZE      : {rw}x{rh}")
            print(f"RAW BLOCKS    : {len(raw_blocks)}")
            print(f"FINAL BLOCKS  : {len(final_blocks)}")
            print(f"EXPECTED 4x2  : {len(final_blocks) * 8}")
            print(f"CREAM         : {cream_count}")
            print(f"GREEN         : {green_count}")
            print(f"CARTON        : {carton_count}")
            print(f"TOTAL         : {len(cells)}")
            print("=" * 50)

            # =========================
            # SAVE DEBUG
            # =========================
            save_dbg("debug_original.jpg", roi)
            save_dbg("debug_normalized.jpg", norm)

            save_dbg("debug_cream.jpg", cream)
            save_dbg("debug_white.jpg", white)
            save_dbg("debug_green_inbound.jpg", green)
            save_dbg("debug_blue_inbound.jpg", blue)
            save_dbg("debug_carton.jpg", carton_mask)

            save_dbg("debug_cargo.jpg", cargo_mask)
            save_dbg("debug_pallet_mask.jpg", pallet_mask)
            save_dbg("debug_box_mask.jpg", carton_mask)
            save_dbg("debug_overlay.jpg", debug_overlay)
            save_dbg("debug_blocks.jpg", debug_blocks)
            save_dbg("debug_grid.jpg", debug_grid)
            save_dbg("debug_pallet_box.jpg", debug_count)

    return len(cells)

    
# =========================
# UPDATE APPSHEET
# =========================
def update_appsheet(row_id, volume_text):

    url = f"https://api.appsheet.com/api/v2/apps/{APP_ID}/tables/{TABLE_NAME}/Action"

    headers = {
        "ApplicationAccessKey": ACCESS_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "Action": "Edit",
        "Rows": [
            {
                "ID": row_id,
                "TFR AI": volume_text,
                "status": "Done"
            }
        ]
    }

    try:
        r = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=20
        )

        print("APPSHEET STATUS:", r.status_code)
        print("APPSHEET RESPONSE:", r.text[:300])

    except Exception as e:
        print("APPSHEET ERROR:", e)


# =========================
# HEALTH CHECK
# =========================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "container-volume-ai"
    })


# =========================
# DEBUG VIEW
# =========================
@app.route("/debug/<filename>", methods=["GET"])
def debug_file(filename):

    allowed = {
        "debug_original.jpg",
        "debug_normalized.jpg",
        "debug_container.jpg",
        "debug_cargo.jpg",
        "debug_empty.jpg",
        "debug_overlay.jpg",
        "debug_overlay_light.jpg",
        "debug_overlay_contour.jpg",
        "debug_green.jpg",
        "debug_brown.jpg",
        "debug_blue.jpg",
        "debug_dark.jpg",
        "debug_texture.jpg"
    }

    if filename not in allowed:
        return jsonify({"error": "file not allowed"}), 403

    path = os.path.join(DEBUG_DIR, filename)

    if not os.path.exists(path):
        return jsonify({"error": "debug file not found"}), 404

    return send_file(path, mimetype="image/jpeg")


@app.route("/debug-list", methods=["GET"])
def debug_list():

    files = [
        "debug_original.jpg",
        "debug_normalized.jpg",
        "debug_container.jpg",
        "debug_cargo.jpg",
        "debug_empty.jpg",
        "debug_overlay.jpg",
        "debug_overlay_light.jpg",
        "debug_overlay_contour.jpg",
        "debug_green.jpg",
        "debug_brown.jpg",
        "debug_blue.jpg",
        "debug_dark.jpg",
        "debug_texture.jpg"
    ]

    base_url = request.host_url.rstrip("/")

    return jsonify({
        "status": "ok",
        "files": [
            {
                "file": f,
                "url": f"{base_url}/debug/{f}",
                "exists": os.path.exists(os.path.join(DEBUG_DIR, f))
            }
            for f in files
        ]
    })


# =========================
# API ENDPOINT
# =========================
@app.route("/predict", methods=["POST"])
def predict():

    row_id = None

    try:
        data = request.get_json(silent=True)

        print("=" * 50)
        print("REQUEST JSON =", data)
        print("=" * 50)

        if not data:
            return jsonify({"error": "no json"}), 400

        image_url = data.get("link")
        row_id = data.get("id")
        project = str(data.get("project", "outbound")).strip().lower()

        debug = bool(data.get("debug", True))
        return_empty = bool(data.get("return_empty", False))

        if not image_url or not row_id:
            return jsonify({"error": "missing data"}), 400

        print("PROJECT =", repr(project))

        with lock:
            if row_id in processed_ids:
                return jsonify({"status": "skipped", "id": row_id}), 200

            processed_ids.add(row_id)

        img = download_image(image_url)

        if img is None:
            with lock:
                processed_ids.discard(row_id)
            return jsonify({"error": "image fail"}), 400

        if project == "outbound":

            volume = gen_volume(
                img,
                debug=debug,
                return_empty=return_empty
            )

            result_text = f"{volume}%"

            print("OUTBOUND:", result_text)

        elif project == "inbound":

            pallet_count = gen_pallet(
                img,
                debug=debug
            )

            result_text = str(pallet_count)

            print("INBOUND:", result_text)

        else:
            with lock:
                processed_ids.discard(row_id)

            return jsonify({
                "error": f"unknown project: {project}"
            }), 400

        update_appsheet(
            row_id,
            result_text
        )

        base_url = request.host_url.rstrip("/")

        if project == "inbound":
            debug_urls = {
                "pallet_mask": f"{base_url}/debug/debug_pallet_mask.jpg",
                "pallet_box": f"{base_url}/debug/debug_pallet_box.jpg",
                "list": f"{base_url}/debug-list"
            }
        else:
            debug_urls = {
                "overlay": f"{base_url}/debug/debug_overlay.jpg",
                "overlay_light": f"{base_url}/debug/debug_overlay_light.jpg",
                "overlay_contour": f"{base_url}/debug/debug_overlay_contour.jpg",
                "cargo": f"{base_url}/debug/debug_cargo.jpg",
                "empty": f"{base_url}/debug/debug_empty.jpg",
                "container": f"{base_url}/debug/debug_container.jpg",
                "blue": f"{base_url}/debug/debug_blue.jpg",
                "list": f"{base_url}/debug-list"
            }

        return jsonify({
            "status": "success",
            "project": project,
            "id": row_id,
            "result": result_text,
            "mode": "empty" if return_empty else "filled",
            "debug": debug,
            "debug_urls": debug_urls
        })

    except Exception:
        if row_id:
            with lock:
                processed_ids.discard(row_id)

        print(traceback.format_exc())
        return jsonify({"error": "server error"}), 500

# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=10000,
        threaded=True
    )

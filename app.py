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

    def save_debug_local(name, im):
        if not debug or im is None:
            return
        try:
            save_debug(name, im)
        except Exception:
            cv2.imwrite(os.path.join(DEBUG_DIR, name), im)

    def clean(mask, k=5, it=1):
        if mask is None or mask.size == 0:
            return mask
        ker = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker, iterations=it)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker, iterations=1)
        return mask

    def smooth(arr, k=21):
        if arr.size == 0:
            return arr
        k = max(3, min(k, arr.size))
        if k % 2 == 0:
            k -= 1
        kernel = np.ones(k, dtype=np.float32) / float(k)
        return np.convolve(arr.astype(np.float32), kernel, mode="same")

    def merge_lines(pos, gap):
        if len(pos) == 0:
            return []
        pos = sorted(pos)
        out = [pos[0]]
        for p in pos[1:]:
            if p - out[-1] > gap:
                out.append(p)
        return out

    def get_lines(mask, axis=0, thr=0.20, gap=18):
        if mask is None or mask.size == 0:
            return []
        proj = np.sum(mask > 0, axis=axis).astype(np.float32)
        if proj.size == 0 or np.max(proj) <= 0:
            return []
        proj = smooth(proj, 21)
        idx = np.where(proj > np.max(proj) * thr)[0]
        return merge_lines(idx.tolist(), gap)

    def nms(boxes, dist=0.45):
        out = []
        for b in boxes:
            x, y, w, h = b
            cx, cy = x + w / 2, y + h / 2
            dup = False
            for ob in out:
                ox, oy, ow, oh = ob
                ocx, ocy = ox + ow / 2, oy + oh / 2
                if abs(cx - ocx) < min(w, ow) * dist and abs(cy - ocy) < min(h, oh) * dist:
                    dup = True
                    break
            if not dup:
                out.append(b)
        return out

    def count_grid(mask, box, draw_img=None, min_fill=0.012):
        x, y, w, h = box
        crop = mask[y:y + h, x:x + w]
        if crop.size == 0:
            return []

        v = get_lines(crop, axis=0, thr=0.18, gap=max(14, w // 25))
        hln = get_lines(crop, axis=1, thr=0.18, gap=max(12, h // 18))

        if len(v) < 2:
            cols = max(1, min(8, round(w / 170)))
            v = [int(i * w / cols) for i in range(cols + 1)]
        else:
            v = [0] + v + [w - 1]

        if len(hln) < 2:
            rows = max(1, min(4, round(h / 150)))
            hln = [int(i * h / rows) for i in range(rows + 1)]
        else:
            hln = [0] + hln + [h - 1]

        if draw_img is not None:
            for vx in v:
                cv2.line(draw_img, (x + vx, y), (x + vx, y + h), (255, 255, 0), 1)
            for hy in hln:
                cv2.line(draw_img, (x, y + hy), (x + w, y + hy), (255, 255, 0), 1)

        local = []
        for r in range(len(hln) - 1):
            for c in range(len(v) - 1):
                x1, x2 = v[c], v[c + 1]
                y1, y2 = hln[r], hln[r + 1]
                cw, ch = x2 - x1, y2 - y1

                if cw < w * 0.08 or ch < h * 0.16:
                    continue

                cell = crop[y1:y2, x1:x2]
                fill = cv2.countNonZero(cell) / float(max(1, cw * ch))

                if fill >= min_fill:
                    local.append((x + x1, y + y1, cw, ch))
                    if draw_img is not None:
                        cv2.rectangle(draw_img, (x + x1, y + y1), (x + x2, y + y2), (0, 180, 255), 1)

        return local

    if img is not None and img.size > 0:
        img = cv2.resize(img, (1280, 720))
        H, W = img.shape[:2]

        roi = img[int(H * 0.20):int(H * 0.82), int(W * 0.02):int(W * 0.98)]

        if roi.size > 0:
            rh, rw = roi.shape[:2]

            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
            norm = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

            hsv = cv2.cvtColor(norm, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)

            cream = cv2.inRange(hsv, (5, 10, 65), (45, 150, 255))
            white = cv2.inRange(hsv, (0, 0, 120), (180, 70, 255))
            blue = cv2.inRange(hsv, (90, 35, 35), (130, 255, 255))
            box_mask = cv2.inRange(hsv, (8, 25, 70), (35, 180, 245))

            for m in [cream, white, blue, box_mask]:
                m[:int(rh * 0.04), :] = 0
                m[int(rh * 0.88):, :] = 0

            frame_mask = cv2.bitwise_or(cv2.bitwise_or(cream, white), blue)
            frame_mask = clean(frame_mask, 5, 1)
            box_mask = clean(box_mask, 9, 2)

            edges = cv2.Canny(gray, 50, 140)
            rack_mask = cv2.bitwise_or(frame_mask, cv2.bitwise_and(edges, frame_mask))
            rack_mask = clean(rack_mask, 3, 1)

            cargo_mask = cv2.bitwise_or(rack_mask, box_mask)
            cargo_mask = clean(cargo_mask, 15, 2)

            debug_blocks = roi.copy()
            debug_grid = roi.copy()
            debug_count = roi.copy()
            mask_color = roi.copy()

            mask_color[rack_mask > 0] = (0, 255, 0)
            mask_color[box_mask > 0] = (0, 180, 255)
            debug_overlay = cv2.addWeighted(roi, 0.75, mask_color, 0.25, 0)

            cnts, _ = cv2.findContours(cargo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            blocks = []

            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                area = w * h

                if area < rw * rh * 0.015:
                    continue
                if h < rh * 0.22 or w < rw * 0.08:
                    continue
                if y > rh * 0.72:
                    continue

                blocks.append((x, y, w, h))

            blocks = sorted(blocks, key=lambda b: (b[1], b[0]))

            for i, (x, y, w, h) in enumerate(blocks, 1):
                cv2.rectangle(debug_blocks, (x, y), (x + w, y + h), (255, 0, 255), 3)
                cv2.putText(debug_blocks, f"B{i}", (x + 5, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

            for x, y, w, h in blocks:
                area = float(max(1, w * h))

                blue_ratio = cv2.countNonZero(blue[y:y + h, x:x + w]) / area
                box_ratio = cv2.countNonZero(box_mask[y:y + h, x:x + w]) / area
                rack_ratio = cv2.countNonZero(rack_mask[y:y + h, x:x + w]) / area

                if box_ratio > 0.28 and rack_ratio < 0.20:
                    est_row = max(1, min(3, round(h / 190)))

                    for i in range(est_row):
                        cy1 = y + int(i * h / est_row)
                        cy2 = y + int((i + 1) * h / est_row)
                        cells.append((x, cy1, w, cy2 - cy1))
                        cv2.rectangle(debug_grid, (x, cy1), (x + w, cy2), (0, 180, 255), 2)

                    continue

                if blue_ratio > 0.015 or rack_ratio > 0.035:
                    cells.extend(count_grid(rack_mask, (x, y, w, h), draw_img=debug_grid, min_fill=0.012))

            cells = nms(cells)

            for i, (x, y, w, h) in enumerate(cells, 1):
                cv2.rectangle(debug_count, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(debug_count, str(i), (x + 7, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            cv2.putText(debug_count, f"PALLET={len(cells)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
            cv2.putText(debug_blocks, f"BLOCKS={len(blocks)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 255), 3)
            cv2.putText(debug_grid, "GRID / CANDIDATE CELLS", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)

            save_debug_local("debug_original.jpg", roi)
            save_debug_local("debug_normalized.jpg", norm)
            save_debug_local("debug_cargo.jpg", cargo_mask)
            save_debug_local("debug_pallet_mask.jpg", rack_mask)
            save_debug_local("debug_box_mask.jpg", box_mask)
            save_debug_local("debug_overlay.jpg", debug_overlay)
            save_debug_local("debug_blocks.jpg", debug_blocks)
            save_debug_local("debug_grid.jpg", debug_grid)
            save_debug_local("debug_pallet_box.jpg", debug_count)

            print("=" * 50)
            print("PALLET DEBUG")
            print(f"ROI SIZE     : {rw}x{rh}")
            print(f"BLOCKS       : {len(blocks)}")
            print(f"PALLET COUNT : {len(cells)}")
            print("=" * 50)

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

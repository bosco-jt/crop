from flask import Flask, request, Response
import cv2
import numpy as np
import requests
import os

app = Flask(__name__)


def download_image(url):
    """Download image from URL and return as numpy array."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    img_array = np.frombuffer(response.content, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image from URL")
    return img


# ─── QUALITY SCORING ───────────────────────────────────────────────

def score_sharpness(gray):
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return round(min(100, (laplacian_var / 300) * 100))


def score_lighting(gray):
    mean_brightness = np.mean(gray)
    if mean_brightness < 40:
        bs = 10
    elif mean_brightness < 80:
        bs = 30 + (mean_brightness - 40)
    elif mean_brightness < 120:
        bs = 70 + (mean_brightness - 80) * 0.75
    elif mean_brightness <= 160:
        bs = 100
    elif mean_brightness <= 200:
        bs = 100 - (mean_brightness - 160) * 0.75
    elif mean_brightness <= 230:
        bs = 70 - (mean_brightness - 200) * 1.33
    else:
        bs = 15

    overexposed = np.sum(gray > 245) / gray.size
    underexposed = np.sum(gray < 10) / gray.size
    penalty = 0
    if overexposed > 0.15:
        penalty += min(30, overexposed * 100)
    if underexposed > 0.15:
        penalty += min(30, underexposed * 100)

    return round(max(0, min(100, bs - penalty)))


def score_resolution(h, w):
    pixels = h * w
    if pixels < 100000:
        return round(max(5, (pixels / 100000) * 40))
    elif pixels < 400000:
        return round(40 + ((pixels - 100000) / 300000) * 30)
    elif pixels < 960000:
        return round(70 + ((pixels - 400000) / 560000) * 30)
    return 100


def score_contrast(gray):
    std = np.std(gray)
    cs = min(100, (std / 60) * 100)
    p5, p95 = np.percentile(gray, [5, 95])
    rs = min(100, ((p95 - p5) / 150) * 100)
    return round(cs * 0.6 + rs * 0.4)


def evaluate_quality(img, detected=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    sharpness = score_sharpness(gray)
    lighting = score_lighting(gray)
    resolution = score_resolution(h, w)
    contrast = score_contrast(gray)
    overall = round(sharpness * 0.35 + contrast * 0.30 + lighting * 0.20 + resolution * 0.15)
    if not detected:
        overall = min(overall, 25)
    return {
        "overall": overall,
        "sharpness": sharpness,
        "lighting": lighting,
        "resolution": resolution,
        "contrast": contrast
    }


# ─── HELPERS ───────────────────────────────────────────────────────

def is_likely_mask_image(img):
    """
    Decide whether the Gemini image looks like a usable white-background mask.
    We expect:
    - a good amount of white background
    - at least one meaningful non-white connected component
    - not an almost totally white or almost unchanged noisy image
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    white_ratio = np.sum(gray > 245) / gray.size
    if white_ratio < 0.35 or white_ratio > 0.995:
        return False

    _, binary = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    img_area = img.shape[0] * img.shape[1]

    large_components = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > img_area * 0.03:
            large_components += 1

    return large_components >= 1


def detect_coordinates_from_mask(img):
    """
    Detect main object coordinates from a likely white-background mask image.
    Returns (x_pct, y_pct, w_pct, h_pct, strategy) or None.
    """
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    img_area = width * height
    candidates = []

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        bbox_area = w * h
        fill_ratio = area / max(bbox_area, 1)
        aspect = max(w, h) / max(min(w, h), 1)

        # Condiciones amplias para DNI, carnet, capturas, etc.
        if bbox_area > img_area * 0.05 and 1.0 < aspect < 3.8 and fill_ratio > 0.20:
            candidates.append((bbox_area, area, x, y, w, h, aspect, fill_ratio))

    if not candidates:
        return None

    # Preferimos el bbox grande y razonablemente compacto
    candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
    _, _, x, y, w, h, _, _ = candidates[0]

    return (x / width, y / height, w / width, h / height, "white_bg")


# ─── FACE CROP ─────────────────────────────────────────────────────

def crop_face_portrait(img):
    """Detect face and crop to passport/ID photo format (35x45mm ratio)."""
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(30, 30))
    if len(faces) == 0:
        return img, False

    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    fx, fy, fw, fh = faces[0]

    target_ratio = 7.0 / 9.0
    crop_h = int(fh / 0.70)
    crop_w = int(crop_h * target_ratio)
    crop_x = max(0, fx + fw // 2 - crop_w // 2)
    crop_y = max(0, fy - int(crop_h * 0.20))

    if crop_x + crop_w > width:
        crop_x = max(0, width - crop_w)
        crop_w = min(crop_w, width - crop_x)
    if crop_y + crop_h > height:
        crop_y = max(0, height - crop_h)
        crop_h = min(crop_h, height - crop_y)

    cropped = img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
    if cropped.shape[0] > 0 and cropped.shape[1] > 0:
        cropped = cv2.resize(cropped, (350, 450), interpolation=cv2.INTER_LANCZOS4)

    return cropped, True


# ─── DOCUMENT CROP ─────────────────────────────────────────────────

def crop_document(img, margin_pct=0.02):
    """
    Crop document from image directly.
    Used as fallback when Gemini mask is not usable.
    Returns (cropped_image, detected, strategy).
    """
    original = img.copy()
    height, width = img.shape[:2]

    scale = 1.0
    max_dim = 1000
    if max(height, width) > max_dim:
        scale = max_dim / max(height, width)
        img_resized = cv2.resize(img, None, fx=scale, fy=scale)
    else:
        img_resized = img.copy()

    img_area = img_resized.shape[0] * img_resized.shape[1]
    gray_r = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    gray_filtered = cv2.bilateralFilter(gray_r, 11, 17, 17)
    blurred = cv2.GaussianBlur(gray_filtered, (5, 5), 0)
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    sat = cv2.GaussianBlur(hsv[:, :, 1], (5, 5), 0)

    strategies = [
        cv2.Canny(blurred, 30, 200),
        cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        ),
        cv2.Canny(blurred, 10, 80),
        cv2.Canny(sat, 20, 100),
        cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
    ]

    best_contour = None
    best_area = 0

    for edged in strategies:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edged, kernel, iterations=2)
        closed = cv2.morphologyEx(
            dilated,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
            iterations=2
        )
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                contour_area = cv2.contourArea(approx)
                if img_area * 0.10 < contour_area < img_area * 0.75:
                    bx, by, bw, bh = cv2.boundingRect(approx)
                    aspect = max(bw, bh) / max(min(bw, bh), 1)
                    if 1.1 < aspect < 3.2 and contour_area > best_area:
                        best_area = contour_area
                        best_contour = approx

    if best_contour is not None:
        best_contour = (best_contour / scale).astype(int)
        pts = best_contour.reshape(4, 2)

        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        d = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(d)]
        rect[3] = pts[np.argmax(d)]

        mx = int(width * margin_pct)
        my = int(height * margin_pct)
        rect[0] = [max(0, rect[0][0] - mx), max(0, rect[0][1] - my)]
        rect[1] = [min(width, rect[1][0] + mx), max(0, rect[1][1] - my)]
        rect[2] = [min(width, rect[2][0] + mx), min(height, rect[2][1] + my)]
        rect[3] = [max(0, rect[3][0] - mx), min(height, rect[3][1] + my)]

        (tl, tr, br, bl) = rect
        max_w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        max_h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

        if max_w > 10 and max_h > 10:
            dst = np.array(
                [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
                dtype="float32"
            )
            matrix = cv2.getPerspectiveTransform(rect.astype("float32"), dst)
            warped = cv2.warpPerspective(original, matrix, (max_w, max_h))
            return warped, True, "contour"

    return original, False, "none"


def detect_coordinates(img, margin_pct=0.02):
    """
    Detect document coordinates.
    Priority:
    1) If image looks like usable Gemini mask -> use mask detection
    2) Else -> fallback to contour detection
    Returns (x, y, w, h, strategy) as percentages or None.
    """
    height, width = img.shape[:2]

    if is_likely_mask_image(img):
        result = detect_coordinates_from_mask(img)
        if result is not None:
            return result

    # Fallback contour detection
    scale = 1.0
    max_dim = 1000
    if max(height, width) > max_dim:
        scale = max_dim / max(height, width)
        img_resized = cv2.resize(img, None, fx=scale, fy=scale)
    else:
        img_resized = img.copy()

    img_area = img_resized.shape[0] * img_resized.shape[1]
    gray_r = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    gray_filtered = cv2.bilateralFilter(gray_r, 11, 17, 17)
    blurred = cv2.GaussianBlur(gray_filtered, (5, 5), 0)

    strategies = [
        cv2.Canny(blurred, 30, 200),
        cv2.Canny(blurred, 10, 80),
        cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
    ]

    best_contour = None
    best_area = 0

    for edged in strategies:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edged, kernel, iterations=2)
        closed = cv2.morphologyEx(
            dilated,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
            iterations=2
        )
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                contour_area = cv2.contourArea(approx)
                if img_area * 0.10 < contour_area < img_area * 0.75 and contour_area > best_area:
                    best_area = contour_area
                    best_contour = approx

    if best_contour is not None:
        x, y, w, h = cv2.boundingRect(best_contour)
        rh, rw = img_resized.shape[:2]
        return (x / rw, y / rh, w / rw, h / rh, "contour")

    return None


def crop_document_dual(detect_img, crop_img, margin_pct=0.02):
    """
    Detect document position on detect_img (Gemini processed if usable),
    then crop crop_img (original) using those coordinates.
    If mask is not usable, fallback to direct crop on original.
    """
    mask_usable = is_likely_mask_image(detect_img)

    if mask_usable:
        result = detect_coordinates(detect_img, margin_pct)
        if result is not None:
            pct_x, pct_y, pct_w, pct_h, strategy = result
            h, w = crop_img.shape[:2]

            x = int(pct_x * w)
            y = int(pct_y * h)
            cw = int(pct_w * w)
            ch = int(pct_h * h)

            mx = int(cw * margin_pct)
            my = int(ch * margin_pct)
            x = max(0, x - mx)
            y = max(0, y - my)
            cw = min(w - x, cw + 2 * mx)
            ch = min(h - y, ch + 2 * my)

            if cw > 10 and ch > 10:
                return crop_img[y:y + ch, x:x + cw], True, strategy

    # Fallback: detect directly on original
    return crop_document(crop_img, margin_pct)


# ─── ROUTES ────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health():
    return {"status": "ok", "service": "document-crop"}


@app.route("/crop", methods=["POST"])
def crop():
    """
    Crop a document or face from an image.

    Mode 1 - Detect on processed image, crop the original:
      Content-Type: multipart/form-data
      - file: binary image (Gemini-processed with white bg, used for detection)
      - original_url: URL of original image (will be cropped)
      - margin: optional (default 0.02)
      - type: optional ("foto" for face crop)

    Mode 2 - Simple crop (detect + crop same image):
      Content-Type: application/json
      { "image_url": "...", "margin": 0.03, "type": "foto" }

    Mode 3 - Binary image:
      Content-Type: image/*
      Binary image data

    Returns: cropped PNG with quality headers.
    """
    try:
        content_type = request.content_type or ""
        img_type = "document"
        margin_pct = 0.02
        detect_img = None
        crop_img = None
        mask_usable = False

        if "multipart/form-data" in content_type:
            file = request.files.get("file")
            if not file:
                return {"error": "file is required in multipart"}, 400

            img_array = np.frombuffer(file.read(), dtype=np.uint8)
            detect_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if detect_img is None:
                return {"error": "Could not decode uploaded image"}, 400

            original_url = request.form.get("original_url", "")
            margin_pct = float(request.form.get("margin", 0.02))
            img_type = request.form.get("type", "document")

            if original_url:
                crop_img = download_image(original_url)
            else:
                crop_img = detect_img

        elif "application/json" in content_type:
            data = request.get_json(silent=True)
            if not data or "image_url" not in data:
                return {"error": "image_url is required"}, 400
            detect_img = download_image(data["image_url"])
            crop_img = detect_img
            margin_pct = float(data.get("margin", 0.02))
            img_type = data.get("type", "document")

        elif "image/" in content_type or "application/octet-stream" in content_type:
            img_array = np.frombuffer(request.data, dtype=np.uint8)
            detect_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if detect_img is None:
                return {"error": "Could not decode image from body"}, 400
            crop_img = detect_img

        else:
            return {
                "error": "Send multipart with file+original_url, JSON with image_url, or binary image"
            }, 400

        mask_usable = is_likely_mask_image(detect_img)

        if img_type == "foto":
            cropped, detected = crop_face_portrait(crop_img)
            strategy = "face"
        else:
            cropped, detected, strategy = crop_document_dual(detect_img, crop_img, margin_pct)

        quality = evaluate_quality(cropped, detected)

        success, buffer = cv2.imencode(".png", cropped)
        if not success:
            return {"error": "Failed to encode cropped image"}, 500

        return Response(
            buffer.tobytes(),
            mimetype="image/png",
            headers={
                "Content-Disposition": "attachment; filename=cropped.png",
                "X-Quality-Overall": str(quality["overall"]),
                "X-Quality-Sharpness": str(quality["sharpness"]),
                "X-Quality-Lighting": str(quality["lighting"]),
                "X-Quality-Resolution": str(quality["resolution"]),
                "X-Quality-Contrast": str(quality["contrast"]),
                "X-Document-Detected": str(detected).lower(),
                "X-Crop-Type": img_type,
                "X-Crop-Strategy": strategy,
                "X-Mask-Usable": str(mask_usable).lower(),
                "Access-Control-Expose-Headers": (
                    "X-Quality-Overall, X-Quality-Sharpness, X-Quality-Lighting, "
                    "X-Quality-Resolution, X-Quality-Contrast, X-Document-Detected, "
                    "X-Crop-Type, X-Crop-Strategy, X-Mask-Usable"
                )
            }
        )

    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to download image: {str(e)}"}, 400
    except Exception as e:
        return {"error": str(e)}, 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

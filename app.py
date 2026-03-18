from flask import Flask, request, Response
import cv2
import numpy as np
import requests
import os
import io
import json
import base64

app = Flask(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


def download_image(url):
    """Download image from URL and return as numpy array."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    img_array = np.frombuffer(response.content, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image from URL")
    return img


def order_points(pts):
    """Order points in: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


# ─── QUALITY SCORING ───────────────────────────────────────────────

def score_sharpness(gray):
    """Measure image sharpness using Laplacian variance."""
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    score = min(100, (laplacian_var / 300) * 100)
    return round(score)


def score_lighting(gray):
    """Evaluate lighting quality based on brightness distribution."""
    mean_brightness = np.mean(gray)

    if mean_brightness < 40:
        brightness_score = 10
    elif mean_brightness < 80:
        brightness_score = 30 + (mean_brightness - 40) * (40 / 40)
    elif mean_brightness < 120:
        brightness_score = 70 + (mean_brightness - 80) * (30 / 40)
    elif mean_brightness <= 160:
        brightness_score = 100
    elif mean_brightness <= 200:
        brightness_score = 100 - (mean_brightness - 160) * (30 / 40)
    elif mean_brightness <= 230:
        brightness_score = 70 - (mean_brightness - 200) * (40 / 30)
    else:
        brightness_score = 15

    overexposed = np.sum(gray > 245) / gray.size
    underexposed = np.sum(gray < 10) / gray.size

    exposure_penalty = 0
    if overexposed > 0.15:
        exposure_penalty += min(30, overexposed * 100)
    if underexposed > 0.15:
        exposure_penalty += min(30, underexposed * 100)

    score = max(0, min(100, brightness_score - exposure_penalty))
    return round(score)


def score_resolution(h, w):
    """Score based on image resolution."""
    pixels = h * w
    min_pixels = 400 * 250
    good_pixels = 800 * 500
    great_pixels = 1200 * 800

    if pixels < min_pixels:
        score = max(5, (pixels / min_pixels) * 40)
    elif pixels < good_pixels:
        score = 40 + ((pixels - min_pixels) / (good_pixels - min_pixels)) * 30
    elif pixels < great_pixels:
        score = 70 + ((pixels - good_pixels) / (great_pixels - good_pixels)) * 30
    else:
        score = 100

    return round(score)


def score_contrast(gray):
    """Measure text legibility via local contrast."""
    std = np.std(gray)
    contrast_score = min(100, (std / 60) * 100)

    p5 = np.percentile(gray, 5)
    p95 = np.percentile(gray, 95)
    dynamic_range = p95 - p5
    range_score = min(100, (dynamic_range / 150) * 100)

    score = (contrast_score * 0.6) + (range_score * 0.4)
    return round(score)


def evaluate_quality(img, detected=True):
    """Evaluate overall document image quality."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]

    sharpness = score_sharpness(gray)
    lighting = score_lighting(gray)
    resolution = score_resolution(h, w)
    contrast = score_contrast(gray)

    overall = round(
        sharpness * 0.35 +
        contrast * 0.30 +
        lighting * 0.20 +
        resolution * 0.15
    )

    # Penalize if document was not detected
    if not detected:
        overall = min(overall, 25)

    return {
        "overall": overall,
        "sharpness": sharpness,
        "lighting": lighting,
        "resolution": resolution,
        "contrast": contrast
    }


# ─── FACE DETECTION & CROP ────────────────────────────────────────

def crop_face_portrait(img, margin_factor=0.4):
    """
    Detect face and crop to passport/ID photo format (35x45mm ratio).
    
    Args:
        img: Input image (BGR)
        margin_factor: How much extra space around the face (0.4 = 40%)
    
    Returns:
        (cropped_image, detected_bool)
    """
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Load Haar Cascade for face detection
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    # Detect faces at multiple scales
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        # Try with more relaxed parameters
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(30, 30)
        )

    if len(faces) == 0:
        return img, False

    # Take the largest face
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    fx, fy, fw, fh = faces[0]

    # Calculate passport photo crop (35x45mm ratio = 7:9)
    target_ratio = 7.0 / 9.0  # width / height

    # Face should be about 70% of the frame height
    crop_h = int(fh / 0.70)
    crop_w = int(crop_h * target_ratio)

    # Center horizontally on face
    face_center_x = fx + fw // 2
    crop_x = face_center_x - crop_w // 2

    # Position vertically: face starts at about 20% from top
    crop_y = fy - int(crop_h * 0.20)

    # Clamp to image bounds
    crop_x = max(0, crop_x)
    crop_y = max(0, crop_y)

    # Adjust if crop goes beyond image
    if crop_x + crop_w > width:
        crop_x = max(0, width - crop_w)
        crop_w = min(crop_w, width - crop_x)

    if crop_y + crop_h > height:
        crop_y = max(0, height - crop_h)
        crop_h = min(crop_h, height - crop_y)

    cropped = img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]

    # Resize to standard passport size (350x450 px)
    if cropped.shape[0] > 0 and cropped.shape[1] > 0:
        cropped = cv2.resize(cropped, (350, 450), interpolation=cv2.INTER_LANCZOS4)

    return cropped, True


# ─── DOCUMENT DETECTION ───────────────────────────────────────────

def find_document_contour(img_resized):
    """Try multiple strategies to find the document contour."""
    h, w = img_resized.shape[:2]
    image_area = h * w
    best_contour = None
    best_area = 0

    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    gray_filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    blurred = cv2.GaussianBlur(gray_filtered, (5, 5), 0)
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    sat = cv2.GaussianBlur(hsv[:, :, 1], (5, 5), 0)

    strategies = [
        cv2.Canny(blurred, 30, 200),
        cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY_INV, 11, 2),
        cv2.Canny(blurred, 10, 80),
        cv2.Canny(sat, 20, 100),
        cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
        cv2.threshold(
            cv2.morphologyEx(gray_filtered, cv2.MORPH_GRADIENT,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))),
            15, 255, cv2.THRESH_BINARY)[1],
    ]

    for edged in strategies:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edged, kernel, iterations=2)
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE,
                                   cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
                                   iterations=2)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours[:10]:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) == 4:
                contour_area = cv2.contourArea(approx)
                if image_area * 0.10 < contour_area < image_area * 0.80:
                    x, y, bw, bh = cv2.boundingRect(approx)
                    aspect = max(bw, bh) / max(min(bw, bh), 1)
                    if 1.1 < aspect < 3.0 and contour_area > best_area:
                        best_area = contour_area
                        best_contour = approx

    return best_contour


def find_document_bounding_rect(img_resized):
    """Fallback: find the largest reasonable bounding rectangle."""
    h, w = img_resized.shape[:2]
    image_area = h * w
    best_rect = None
    best_area = 0

    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    sat = cv2.GaussianBlur(hsv[:, :, 1], (5, 5), 0)

    edges_list = [
        cv2.Canny(blurred, 30, 200),
        cv2.Canny(blurred, 10, 80),
        cv2.Canny(sat, 20, 100),
    ]

    for edged in edges_list:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edged, kernel, iterations=3)
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE,
                                   cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
                                   iterations=3)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            area = cv2.contourArea(contour)
            if image_area * 0.10 < area < image_area * 0.80:
                x, y, bw, bh = cv2.boundingRect(contour)
                aspect = max(bw, bh) / max(min(bw, bh), 1)
                if 1.1 < aspect < 3.0 and area > best_area:
                    best_area = area
                    best_rect = (x, y, bw, bh)

    return best_rect


def find_document_by_text(img_resized, margin_pct=0.05):
    """
    Detect document area by finding text regions using MSER.
    """
    h, w = img_resized.shape[:2]
    image_area = h * w
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    mser = cv2.MSER_create()
    mser.setMinArea(80)
    mser.setMaxArea(int(image_area * 0.005))

    regions, _ = mser.detectRegions(gray)

    if len(regions) < 5:
        return None

    text_boxes = []
    for region in regions:
        x, y, rw, rh = cv2.boundingRect(region)
        if rw < 3 or rh < 3:
            continue
        if rw > w * 0.3 or rh > h * 0.15:
            continue
        region_aspect = max(rw, rh) / max(min(rw, rh), 1)
        if region_aspect > 10:
            continue
        text_boxes.append([x, y, x + rw, y + rh])

    if len(text_boxes) < 10:
        return None

    text_boxes = np.array(text_boxes)

    for axis in [0, 1]:
        vals = text_boxes[:, axis]
        q1, q3 = np.percentile(vals, [15, 85])
        iqr = q3 - q1
        mask = (vals >= q1 - 1.5 * iqr) & (vals <= q3 + 1.5 * iqr)
        text_boxes = text_boxes[mask]

    if len(text_boxes) < 5:
        return None

    min_x = np.min(text_boxes[:, 0])
    min_y = np.min(text_boxes[:, 1])
    max_x = np.max(text_boxes[:, 2])
    max_y = np.max(text_boxes[:, 3])

    text_w = max_x - min_x
    text_h = max_y - min_y
    text_area = text_w * text_h

    if text_area < image_area * 0.08 or text_area > image_area * 0.80:
        return None

    aspect = max(text_w, text_h) / max(min(text_w, text_h), 1)
    if aspect < 1.1 or aspect > 3.0:
        return None

    mx = int(text_w * margin_pct)
    my = int(text_h * margin_pct)
    x = max(0, min_x - mx)
    y = max(0, min_y - my)
    bw = min(w - x, text_w + 2 * mx)
    bh = min(h - y, text_h + 2 * my)

    return (x, y, bw, bh)


def find_document_by_smoothness(img_resized):
    """
    Detect document by finding the largest smooth/uniform region.
    Documents have smooth surfaces vs textured backgrounds (fabric, wood, etc).
    Uses local variance to create a smoothness map, then finds the largest smooth blob.
    """
    h, w = img_resized.shape[:2]
    image_area = h * w
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Calculate local variance using a sliding window
    kernel_size = 15
    mean = cv2.blur(gray, (kernel_size, kernel_size))
    sqr_mean = cv2.blur(gray * gray, (kernel_size, kernel_size))
    variance = sqr_mean - mean * mean
    variance = np.clip(variance, 0, None)

    # Threshold: low variance = smooth (document), high variance = textured (background)
    # Normalize variance to 0-255
    var_norm = cv2.normalize(variance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Smooth regions have LOW variance → invert so smooth = white
    smooth_map = cv2.bitwise_not(var_norm)

    # Threshold to get binary smooth regions
    _, binary = cv2.threshold(smooth_map, 180, 255, cv2.THRESH_BINARY)

    # Clean up with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find contours of smooth regions
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Find the largest smooth region that could be a document
    best_rect = None
    best_area = 0

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        area = cv2.contourArea(contour)
        if area < image_area * 0.08 or area > image_area * 0.80:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        aspect = max(bw, bh) / max(min(bw, bh), 1)

        # Document aspect ratio check
        if 1.1 < aspect < 3.0 and area > best_area:
            best_area = area
            best_rect = (x, y, bw, bh)

    return best_rect


def gemini_detect_document(img):
    """
    Use Gemini 2.5 Flash to detect document position using percentages.
    Returns ((x, y, w, h) in pixels, info_str) or (None, error_str).
    """
    if not GEMINI_API_KEY:
        return None, "no_api_key"

    try:
        success, buffer = cv2.imencode(".jpeg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            return None, "encode_failed"
        img_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
        img_h, img_w = img.shape[:2]

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"

        prompt = 'Find the rectangular document (ID card, credit card, passport or similar) in this photo. Return its position as PERCENTAGES (0 to 100) of the image dimensions. Return ONLY a JSON object: {"left": 0, "top": 0, "right": 100, "bottom": 100}. "left" = percentage from left edge where document starts. "top" = percentage from top where document starts. "right" = percentage from left where document ends. "bottom" = percentage from top where document ends. It is better to include extra background than to cut the document. No markdown, no backticks, ONLY the JSON.'

        payload = {
            "contents": [{
                "parts": [
                    {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}},
                    {"text": prompt}
                ]
            }],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 1000}
        }

        resp = requests.post(url, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        text = data["candidates"][0]["content"]["parts"][0]["text"]
        text = text.replace("```json", "").replace("```", "").strip()
        coords = json.loads(text)

        left = float(coords["left"])
        top = float(coords["top"])
        right = float(coords["right"])
        bottom = float(coords["bottom"])

        # Convert percentages to pixels
        x = int(img_w * left / 100)
        y = int(img_h * top / 100)
        w = int(img_w * (right - left) / 100)
        h = int(img_h * (bottom - top) / 100)

        info = f"pcts=L{left:.1f},T{top:.1f},R{right:.1f},B{bottom:.1f}|px=x{x},y{y},w{w},h{h}"

        if w < 50 or h < 50:
            return None, f"too_small|{info}"

        return (x, y, w, h), info

    except Exception as e:
        return None, str(e)


def detect_and_crop_document(img, margin_pct=0.02):
    """Detect the document contour and crop it. Returns (image, detected, strategy, debug)."""
    original = img.copy()
    height, width = img.shape[:2]
    debug = {"image_size": f"{width}x{height}"}

    scale = 1.0
    max_dim = 1000
    if max(height, width) > max_dim:
        scale = max_dim / max(height, width)
        img_resized = cv2.resize(img, None, fx=scale, fy=scale)
    else:
        img_resized = img.copy()

    img_area = img_resized.shape[0] * img_resized.shape[1]
    debug["scale"] = round(scale, 3)

    # Strategy 1: Contour-based
    doc_contour = find_document_contour(img_resized)

    if doc_contour is not None:
        contour_area = cv2.contourArea(doc_contour)
        area_pct = round(contour_area / img_area * 100, 1)
        pts_scaled = (doc_contour / scale).astype(int).reshape(4, 2).tolist()
        debug["contour"] = f"found|area={area_pct}%|pts={pts_scaled}"

        if contour_area < img_area * 0.70:
            doc_contour_scaled = (doc_contour / scale).astype(int)
            pts = doc_contour_scaled.reshape(4, 2)
            rect = order_points(pts)

            margin_x = int(width * margin_pct)
            margin_y = int(height * margin_pct)
            rect[0] = [max(0, rect[0][0] - margin_x), max(0, rect[0][1] - margin_y)]
            rect[1] = [min(width, rect[1][0] + margin_x), max(0, rect[1][1] - margin_y)]
            rect[2] = [min(width, rect[2][0] + margin_x), min(height, rect[2][1] + margin_y)]
            rect[3] = [max(0, rect[3][0] - margin_x), min(height, rect[3][1] + margin_y)]

            (tl, tr, br, bl) = rect

            max_width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
            max_height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

            dst = np.array([
                [0, 0], [max_width - 1, 0],
                [max_width - 1, max_height - 1], [0, max_height - 1]
            ], dtype="float32")

            matrix = cv2.getPerspectiveTransform(rect.astype("float32"), dst)
            warped = cv2.warpPerspective(original, matrix, (max_width, max_height))
            return warped, True, "contour", debug
        else:
            debug["contour"] += "|REJECTED:too_large"
    else:
        debug["contour"] = "none_found"

    # Strategy 2: Text-based detection
    text_rect = find_document_by_text(img_resized)

    if text_rect is not None:
        tx, ty, tw, th = text_rect
        debug["text"] = f"found|x={int(tx/scale)},y={int(ty/scale)},w={int(tw/scale)},h={int(th/scale)}"

        x = int(tx / scale)
        y = int(ty / scale)
        w = int(tw / scale)
        h = int(th / scale)

        margin_x = int(width * margin_pct)
        margin_y = int(height * margin_pct)
        x = max(0, x - margin_x)
        y = max(0, y - margin_y)
        w = min(width - x, w + 2 * margin_x)
        h = min(height - y, h + 2 * margin_y)

        debug["text_crop"] = f"x={x},y={y},w={w},h={h}"
        return original[y:y+h, x:x+w], True, "text", debug
    else:
        debug["text"] = "none_found"

    # Strategy 3: Smoothness-based (documents are smooth vs textured backgrounds)
    smooth_rect = find_document_by_smoothness(img_resized)

    if smooth_rect is not None:
        sx, sy, sw, sh = smooth_rect
        debug["smooth"] = f"found|x={int(sx/scale)},y={int(sy/scale)},w={int(sw/scale)},h={int(sh/scale)}"

        x = int(sx / scale)
        y = int(sy / scale)
        w = int(sw / scale)
        h = int(sh / scale)

        margin_x = int(width * margin_pct)
        margin_y = int(height * margin_pct)
        x = max(0, x - margin_x)
        y = max(0, y - margin_y)
        w = min(width - x, w + 2 * margin_x)
        h = min(height - y, h + 2 * margin_y)

        return original[y:y+h, x:x+w], True, "smooth", debug
    else:
        debug["smooth"] = "none_found"

    # Strategy 4: Gemini AI detects document position using percentages
    gemini_rect, gemini_info = gemini_detect_document(original)

    if gemini_rect is not None:
        gx, gy, gw, gh = gemini_rect
        debug["gemini"] = f"found|{gemini_info}"

        margin_x = int(width * margin_pct)
        margin_y = int(height * margin_pct)
        x = max(0, gx - margin_x)
        y = max(0, gy - margin_y)
        w = min(width - x, gw + 2 * margin_x)
        h = min(height - y, gh + 2 * margin_y)

        return original[y:y+h, x:x+w], True, "gemini", debug
    else:
        debug["gemini"] = f"failed|{gemini_info}"

    # Strategy 5: Bounding rect fallback
    bounding = find_document_bounding_rect(img_resized)

    if bounding is not None:
        bx, by, bw, bh = bounding
        debug["bounding"] = f"found|x={int(bx/scale)},y={int(by/scale)},w={int(bw/scale)},h={int(bh/scale)}"

        x = int(bx / scale)
        y = int(by / scale)
        w = int(bw / scale)
        h = int(bh / scale)

        margin_x = int(width * margin_pct)
        margin_y = int(height * margin_pct)
        x = max(0, x - margin_x)
        y = max(0, y - margin_y)
        w = min(width - x, w + 2 * margin_x)
        h = min(height - y, h + 2 * margin_y)

        return original[y:y+h, x:x+w], True, "bounding", debug
    else:
        debug["bounding"] = "none_found"

    return original, False, "none", debug


# ─── ROUTES ────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health():
    return {"status": "ok", "service": "document-crop"}


@app.route("/crop", methods=["POST"])
def crop():
    """
    Crop a document or face from an image.

    JSON body:
    {
        "image_url": "https://...",
        "margin": 0.03,
        "type": "foto"  // "foto" = face crop, anything else = document crop
    }

    Or binary image data with Content-Type: image/*

    Returns: cropped image as PNG with quality headers
    """
    try:
        content_type = request.content_type or ""
        img_type = "document"

        if "application/json" in content_type:
            data = request.get_json(silent=True)
            if not data or "image_url" not in data:
                return {"error": "image_url is required"}, 400
            img = download_image(data["image_url"])
            margin_pct = float(data.get("margin", 0.02))
            img_type = data.get("type", "document")

        elif "image/" in content_type or "application/octet-stream" in content_type:
            img_array = np.frombuffer(request.data, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                return {"error": "Could not decode image from body"}, 400
            margin_pct = 0.02
        else:
            return {"error": "Send JSON with image_url or binary image data"}, 400

        # Route based on type
        if img_type == "foto":
            cropped, detected = crop_face_portrait(img)
            strategy = "face"
            debug = {}
        else:
            cropped, detected, strategy, debug = detect_and_crop_document(img, margin_pct)

        # Evaluate quality
        quality = evaluate_quality(cropped, detected)

        # Encode as PNG
        success, buffer = cv2.imencode(".png", cropped)
        if not success:
            return {"error": "Failed to encode cropped image"}, 500

        # Build debug header string
        debug_str = " | ".join(f"{k}={v}" for k, v in debug.items())

        response = Response(
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
                "X-Debug": debug_str,
                "Access-Control-Expose-Headers": "X-Quality-Overall, X-Quality-Sharpness, X-Quality-Lighting, X-Quality-Resolution, X-Quality-Contrast, X-Document-Detected, X-Crop-Type, X-Crop-Strategy, X-Debug"
            }
        )
        return response

    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to download image: {str(e)}"}, 400
    except Exception as e:
        return {"error": str(e)}, 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

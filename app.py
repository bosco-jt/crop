from flask import Flask, request, Response
import cv2
import numpy as np
import requests
import os
import io
import json

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
    """
    Measure image sharpness using Laplacian variance.
    Higher variance = sharper image.
    Returns 0-100 score.
    """
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Typical range: <50 very blurry, 50-200 acceptable, >200 sharp
    score = min(100, (laplacian_var / 300) * 100)
    return round(score)


def score_lighting(gray):
    """
    Evaluate lighting quality based on brightness distribution.
    Penalizes both overexposure and underexposure.
    Returns 0-100 score.
    """
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)

    # Ideal mean brightness: around 120-140
    # Penalize if too dark (<80) or too bright (>200)
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

    # Check for overexposed/underexposed pixels
    overexposed = np.sum(gray > 245) / gray.size
    underexposed = np.sum(gray < 10) / gray.size

    # Penalize if >15% of pixels are blown out or crushed
    exposure_penalty = 0
    if overexposed > 0.15:
        exposure_penalty += min(30, overexposed * 100)
    if underexposed > 0.15:
        exposure_penalty += min(30, underexposed * 100)

    score = max(0, min(100, brightness_score - exposure_penalty))
    return round(score)


def score_resolution(h, w):
    """
    Score based on image resolution.
    For ID documents, minimum useful is ~400x250px.
    Returns 0-100 score.
    """
    pixels = h * w
    min_pixels = 400 * 250  # 100,000 - minimum for readability
    good_pixels = 800 * 500  # 400,000 - good quality
    great_pixels = 1200 * 800  # 960,000 - excellent

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
    """
    Measure text legibility via local contrast.
    Uses standard deviation and histogram spread.
    Returns 0-100 score.
    """
    std = np.std(gray)
    # Good contrast for documents: std > 40
    contrast_score = min(100, (std / 60) * 100)

    # Also check histogram spread (dynamic range)
    p5 = np.percentile(gray, 5)
    p95 = np.percentile(gray, 95)
    dynamic_range = p95 - p5
    # Good range for documents: >100
    range_score = min(100, (dynamic_range / 150) * 100)

    score = (contrast_score * 0.6) + (range_score * 0.4)
    return round(score)


def evaluate_quality(img):
    """
    Evaluate overall document image quality.
    Returns dict with overall score (0-100) and individual scores.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]

    sharpness = score_sharpness(gray)
    lighting = score_lighting(gray)
    resolution = score_resolution(h, w)
    contrast = score_contrast(gray)

    # Weighted average: sharpness and contrast matter most for document legibility
    overall = round(
        sharpness * 0.35 +
        contrast * 0.30 +
        lighting * 0.20 +
        resolution * 0.15
    )

    return {
        "overall": overall,
        "sharpness": sharpness,
        "lighting": lighting,
        "resolution": resolution,
        "contrast": contrast
    }


# ─── DOCUMENT DETECTION ───────────────────────────────────────────

def find_document_contour(img_resized):
    """
    Try multiple strategies to find the document contour.
    Returns the best 4-point contour found, or None.
    """
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
                if image_area * 0.10 < contour_area < image_area * 0.95:
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
            if image_area * 0.10 < area < image_area * 0.95:
                x, y, bw, bh = cv2.boundingRect(contour)
                aspect = max(bw, bh) / max(min(bw, bh), 1)
                if 1.1 < aspect < 3.0 and area > best_area:
                    best_area = area
                    best_rect = (x, y, bw, bh)

    return best_rect


def detect_and_crop_document(img, margin_pct=0.02):
    """
    Detect the document contour in the image and crop it.
    Uses multiple strategies to handle both dark and light backgrounds.
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

    doc_contour = find_document_contour(img_resized)

    if doc_contour is not None:
        doc_contour = (doc_contour / scale).astype(int)
        pts = doc_contour.reshape(4, 2)
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
        return warped, True

    bounding = find_document_bounding_rect(img_resized)

    if bounding is not None:
        x, y, w, h = bounding
        x = int(x / scale)
        y = int(y / scale)
        w = int(w / scale)
        h = int(h / scale)

        margin_x = int(width * margin_pct)
        margin_y = int(height * margin_pct)
        x = max(0, x - margin_x)
        y = max(0, y - margin_y)
        w = min(width - x, w + 2 * margin_x)
        h = min(height - y, h + 2 * margin_y)

        return original[y:y+h, x:x+w], True

    return original, False


# ─── ROUTES ────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health():
    return {"status": "ok", "service": "document-crop"}


@app.route("/crop", methods=["POST"])
def crop():
    """
    Crop a document from an image and return quality score in headers.

    JSON body: { "image_url": "https://...", "margin": 0.03 }
    Or binary image data with Content-Type: image/*

    Returns: cropped image as PNG with quality headers:
      X-Quality-Overall: 0-100
      X-Quality-Sharpness: 0-100
      X-Quality-Lighting: 0-100
      X-Quality-Resolution: 0-100
      X-Quality-Contrast: 0-100
      X-Document-Detected: true/false
    """
    try:
        content_type = request.content_type or ""

        if "application/json" in content_type:
            data = request.get_json(silent=True)
            if not data or "image_url" not in data:
                return {"error": "image_url is required"}, 400
            img = download_image(data["image_url"])
            margin_pct = float(data.get("margin", 0.02))

        elif "image/" in content_type or "application/octet-stream" in content_type:
            img_array = np.frombuffer(request.data, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                return {"error": "Could not decode image from body"}, 400
            margin_pct = 0.02
        else:
            return {"error": "Send JSON with image_url or binary image data"}, 400

        # Crop
        cropped, detected = detect_and_crop_document(img, margin_pct)

        # Evaluate quality on the cropped image
        quality = evaluate_quality(cropped)

        # Encode as PNG
        success, buffer = cv2.imencode(".png", cropped)
        if not success:
            return {"error": "Failed to encode cropped image"}, 500

        # Return image with quality scores in headers
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
                "Access-Control-Expose-Headers": "X-Quality-Overall, X-Quality-Sharpness, X-Quality-Lighting, X-Quality-Resolution, X-Quality-Contrast, X-Document-Detected"
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

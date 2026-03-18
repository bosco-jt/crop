from flask import Flask, request, send_file
import cv2
import numpy as np
import requests
import os
import io

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
        # 1: Standard Canny (works with dark backgrounds)
        cv2.Canny(blurred, 30, 200),
        # 2: Adaptive threshold (works with low contrast / light backgrounds)
        cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY_INV, 11, 2),
        # 3: Low threshold Canny (catches subtle edges)
        cv2.Canny(blurred, 10, 80),
        # 4: Saturation channel (documents have different saturation than background)
        cv2.Canny(sat, 20, 100),
        # 5: Otsu threshold
        cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
        # 6: Morphological gradient
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

    # Resize for faster processing
    scale = 1.0
    max_dim = 1000
    if max(height, width) > max_dim:
        scale = max_dim / max(height, width)
        img_resized = cv2.resize(img, None, fx=scale, fy=scale)
    else:
        img_resized = img.copy()

    # Try to find a 4-corner document contour
    doc_contour = find_document_contour(img_resized)

    if doc_contour is not None:
        doc_contour = (doc_contour / scale).astype(int)
        pts = doc_contour.reshape(4, 2)
        rect = order_points(pts)

        # Expand by margin
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

    # Fallback: bounding rectangle
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


@app.route("/", methods=["GET"])
def health():
    return {"status": "ok", "service": "document-crop"}


@app.route("/crop", methods=["POST"])
def crop():
    """
    Crop a document from an image.
    JSON body: { "image_url": "https://...", "margin": 0.03 }
    Or binary image data with Content-Type: image/*
    Returns: cropped image as PNG
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

        cropped, detected = detect_and_crop_document(img, margin_pct)

        success, buffer = cv2.imencode(".png", cropped)
        if not success:
            return {"error": "Failed to encode cropped image"}, 500

        return send_file(
            io.BytesIO(buffer.tobytes()),
            mimetype="image/png",
            download_name="cropped.png"
        )

    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to download image: {str(e)}"}, 400
    except Exception as e:
        return {"error": str(e)}, 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

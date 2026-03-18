from flask import Flask, request, send_file
import cv2
import numpy as np
import requests
import tempfile
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


def detect_and_crop_document(img, margin_pct=0.02):
    """
    Detect the document contour in the image and crop it.
    margin_pct: percentage of image to add as margin around document (default 2%)
    Returns cropped image or original if no document detected.
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

    # Grayscale + bilateral filter + blur
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edged = cv2.Canny(blurred, 30, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edged = cv2.dilate(edged, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return original, False

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    doc_contour = None
    for contour in contours[:10]:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            contour_area = cv2.contourArea(approx)
            image_area = img_resized.shape[0] * img_resized.shape[1]
            if contour_area > image_area * 0.10:
                doc_contour = approx
                break

    if doc_contour is None:
        # Fallback: bounding rect of largest contour
        largest = contours[0]
        contour_area = cv2.contourArea(largest)
        image_area = img_resized.shape[0] * img_resized.shape[1]

        if contour_area < image_area * 0.10:
            return original, False

        x, y, w, h = cv2.boundingRect(largest)
        x = int(x / scale)
        y = int(y / scale)
        w = int(w / scale)
        h = int(h / scale)

        # Small margin
        margin = 5
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(width - x, w + 2 * margin)
        h = min(height - y, h + 2 * margin)

        return original[y:y+h, x:x+w], True

    # Scale contour back to original dimensions
    doc_contour = (doc_contour / scale).astype(int)
    pts = doc_contour.reshape(4, 2)
    rect = order_points(pts)

    # Expand rectangle outward by margin
    margin_x = int(width * margin_pct)
    margin_y = int(height * margin_pct)
    rect[0] = [max(0, rect[0][0] - margin_x), max(0, rect[0][1] - margin_y)]
    rect[1] = [min(width, rect[1][0] + margin_x), max(0, rect[1][1] - margin_y)]
    rect[2] = [min(width, rect[2][0] + margin_x), min(height, rect[2][1] + margin_y)]
    rect[3] = [max(0, rect[3][0] - margin_x), min(height, rect[3][1] + margin_y)]

    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect.astype("float32"), dst)
    warped = cv2.warpPerspective(original, matrix, (max_width, max_height))

    return warped, True


@app.route("/", methods=["GET"])
def health():
    return {"status": "ok", "service": "document-crop"}


@app.route("/crop", methods=["POST"])
def crop():
    """
    Crop a document from an image.
    
    Accepts JSON body:
    { "image_url": "https://..." }
    
    Or binary image data in the request body with Content-Type: image/*
    
    Returns: cropped image as PNG binary
    """
    try:
        content_type = request.content_type or ""

        if "application/json" in content_type:
            data = request.get_json(silent=True)
            if not data or "image_url" not in data:
                return {"error": "image_url is required"}, 400
            img = download_image(data["image_url"])
            margin_pct = data.get("margin", 0.02)

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

        # Encode as PNG
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

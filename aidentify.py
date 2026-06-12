import os
import io
import base64
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory

from owlv2_detector import detect_sensitive_objects, DEFAULT_SENSITIVE_OBJECTS

app = Flask(__name__)

# =============================================================
# CONFIGURATION FLAGS (Adjust these to easily customize detection & masking)
# =============================================================
DEFAULT_THRESHOLD = 0.15          # Detection confidence threshold (0.0 to 1.0)
DEFAULT_MASK_TYPE = "inpaint"     # Masking method: "inpaint", "mosaic", or "blur"
# =============================================================

# Ensure template and static directories exist
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

def pil_to_cv2(pil_img):
    """Convert PIL image to OpenCV format (BGR)."""
    open_cv_image = np.array(pil_img)
    # Convert RGB to BGR
    if len(open_cv_image.shape) == 3:
        if open_cv_image.shape[2] == 3:
            open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
        elif open_cv_image.shape[2] == 4:
            open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGBA2BGR)
    return open_cv_image

def cv2_to_base64(cv_img):
    """Convert OpenCV BGR image to base64 string."""
    _, buffer = cv2.imencode('.jpg', cv_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"

def apply_masking(cv_img, detections, mask_type="inpaint", mask_face_mosaic=True):
    """
    Applies inpainting, mosaic, or blur onto the detected sensitive regions.
    Faces are forced to be mosaiced if mask_face_mosaic is True.
    """
    h, w = cv_img.shape[:2]
    
    # Separate face detections from other detections depending on face mosaic toggle
    if mask_face_mosaic:
        face_dets = [d for d in detections if d["label"] == "human face"]
        general_dets = [d for d in detections if d["label"] != "human face"]
    else:
        face_dets = []
        general_dets = detections
    
    output_img = cv_img.copy()
    
    # 1. Process general detections
    if len(general_dets) > 0:
        general_mask = np.zeros((h, w), dtype=np.uint8)
        for det in general_dets:
            box = det["box"]
            xmin, ymin, xmax, ymax = box
            xmin = max(0, min(xmin, w))
            xmax = max(0, min(xmax, w))
            ymin = max(0, min(ymin, h))
            ymax = max(0, min(ymax, h))
            if xmax > xmin and ymax > ymin:
                cv2.rectangle(general_mask, (xmin, ymin), (xmax, ymax), 255, -1)
                
        if mask_type == "inpaint":
            output_img = cv2.inpaint(cv_img, general_mask, 3, cv2.INPAINT_TELEA)
        elif mask_type == "mosaic":
            contours, _ = cv2.findContours(general_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, bw, bh = cv2.boundingRect(contour)
                if bw > 0 and bh > 0:
                    roi = output_img[y:y+bh, x:x+bw]
                    factor = max(16, int(max(bw, bh) / 4))
                    temp = cv2.resize(roi, (max(1, bw // factor), max(1, bh // factor)), interpolation=cv2.INTER_LINEAR)
                    pixelated = cv2.resize(temp, (bw, bh), interpolation=cv2.INTER_NEAREST)
                    output_img[y:y+bh, x:x+bw] = pixelated
        elif mask_type == "blur":
            blurred_full = cv2.GaussianBlur(cv_img, (51, 51), 0)
            mask_3d = cv2.merge([general_mask, general_mask, general_mask])
            output_img = np.where(mask_3d == 255, blurred_full, cv_img)

    # 2. Process face detections (always mosaic)
    if len(face_dets) > 0:
        face_mask = np.zeros((h, w), dtype=np.uint8)
        for det in face_dets:
            box = det["box"]
            xmin, ymin, xmax, ymax = box
            xmin = max(0, min(xmin, w))
            xmax = max(0, min(xmax, w))
            ymin = max(0, min(ymin, h))
            ymax = max(0, min(ymax, h))
            if xmax > xmin and ymax > ymin:
                cv2.rectangle(face_mask, (xmin, ymin), (xmax, ymax), 255, -1)
                
        contours, _ = cv2.findContours(face_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            if bw > 0 and bh > 0:
                roi = output_img[y:y+bh, x:x+bw]
                factor = max(16, int(max(bw, bh) / 4))
                temp = cv2.resize(roi, (max(1, bw // factor), max(1, bh // factor)), interpolation=cv2.INTER_LINEAR)
                pixelated = cv2.resize(temp, (bw, bh), interpolation=cv2.INTER_NEAREST)
                output_img[y:y+bh, x:x+bw] = pixelated
                
    return output_img

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/config", methods=["GET"])
def get_config():
    """Get list of supported sensitive object labels."""
    return jsonify({
        "sensitive_objects": DEFAULT_SENSITIVE_OBJECTS
    })

@app.route("/api/process", methods=["POST"])
def process_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400
        
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty file name"}), 400
        
    # Get parameters (fallback to config flags defined at the top of the file)
    threshold_val = request.form.get("threshold")
    threshold = float(threshold_val) if threshold_val is not None else DEFAULT_THRESHOLD
    
    mask_type = request.form.get("mask_type") or DEFAULT_MASK_TYPE
    mask_face_mosaic = request.form.get("mask_face_mosaic", "false") == "true"
    
    # Resolve objects to search - Always include "human face" to ensure detection in all modes
    selected_objects = list(DEFAULT_SENSITIVE_OBJECTS)
    if "human face" not in selected_objects:
        selected_objects.append("human face")

    # Read image
    try:
        img_bytes = file.read()
        pil_img = Image.open(io.BytesIO(img_bytes))
    except Exception as e:
        return jsonify({"error": f"Failed to load image: {str(e)}"}), 400
        
    # 1. Detect objects using OWLv2
    try:
        detections = detect_sensitive_objects(
            image=pil_img, 
            text_queries=selected_objects, 
            score_threshold=threshold
        )
    except Exception as e:
        return jsonify({"error": f"Detection failed: {str(e)}"}), 500
        
    # 2. Process image with mask/inpaint
    try:
        cv_img = pil_to_cv2(pil_img)
        # If mask_face_mosaic is False, discard face detections from masking process
        mask_detections = detections
        if not mask_face_mosaic:
            mask_detections = [d for d in detections if d["label"] != "human face"]
            
        processed_cv_img = apply_masking(
            cv_img=cv_img, 
            detections=mask_detections, 
            mask_type=mask_type,
            mask_face_mosaic=mask_face_mosaic
        )
    except Exception as e:
        return jsonify({"error": f"Masking process failed: {str(e)}"}), 500
        
    # Calculate Danger Score (sum of confidence scores * coefficient)
    danger_coefficient = float(request.form.get("danger_coefficient", 25.0))
    score_sum = sum(det["score"] for det in detections)
    danger_score = round(min(100.0, score_sum * danger_coefficient), 1)

    # Convert results to response format
    original_base64 = cv2_to_base64(cv_img)
    processed_base64 = cv2_to_base64(processed_cv_img)
    
    return jsonify({
        "original_image": original_base64,
        "processed_image": processed_base64,
        "detections": detections,
        "danger_score": danger_score
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
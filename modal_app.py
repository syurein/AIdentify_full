import os
import io
import base64
import numpy as np
import cv2
from PIL import Image
import modal
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# Create the Modal app
app = modal.App("aidentify")

# Define the container image with required python libraries
image = (
    modal.Image.debian_slim()
    .apt_install("libgl1-mesa-glx", "libglib2.0-0") # Required for OpenCV
    .pip_install(
        "torch",
        "torchvision",
        "transformers",
        "scipy",
        "opencv-python-headless",
        "pillow",
        "numpy",
        "fastapi",
        "python-multipart"
    )
)

# We use class-based definition to load the model ONCE when container starts
@app.cls(gpu="T4", image=image, scaledown_window=120)
class AIdentifyAPI:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading OWLv2 model on {self.device}...")
        self.model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(self.device)
        self.processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        print("Model loaded successfully.")
        
        # Default labels to search for
        self.default_sensitive_objects = [
            "credit card", "passport", "driver's license", "license plate",
            "identity card", "social security card", "health insurance card",
            "bank statement", "tax document", "payslip", "utility bill",
            "medical record", "signature", "human face"
        ]

    def pil_to_cv2(self, pil_img):
        open_cv_image = np.array(pil_img)
        if len(open_cv_image.shape) == 3:
            if open_cv_image.shape[2] == 3:
                open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
            elif open_cv_image.shape[2] == 4:
                open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGBA2BGR)
        return open_cv_image

    def cv2_to_base64(self, cv_img):
        _, buffer = cv2.imencode('.jpg', cv_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{img_base64}"

    def apply_masking(self, cv_img, detections, mask_type="inpaint", mask_face_mosaic=True):
        h, w = cv_img.shape[:2]
        
        if mask_face_mosaic:
            face_dets = [d for d in detections if d["label"] == "human face"]
            general_dets = [d for d in detections if d["label"] != "human face"]
        else:
            face_dets = []
            general_dets = detections
        
        output_img = cv_img.copy()
        
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

    @modal.asgi_app()
    def web(self):
        fastapi_app = FastAPI()

        # Add CORS
        fastapi_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @fastapi_app.get("/", response_class=HTMLResponse)
        def index():
            # Modal will automatically include the templates directory when deploying
            template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
            with open(template_path, "r", encoding="utf-8") as f:
                return f.read()

        @fastapi_app.get("/api/config")
        def get_config():
            return {"sensitive_objects": self.default_sensitive_objects}

        @fastapi_app.post("/api/process")
        async def process_image(
            image: UploadFile = File(...),
            threshold: float = Form(0.15),
            mask_type: str = Form("inpaint"),
            mask_face_mosaic: str = Form("false"),
            danger_coefficient: float = Form(25.0)
        ):
            is_face_mosaic = mask_face_mosaic == "true"

            try:
                img_bytes = await image.read()
                pil_img = Image.open(io.BytesIO(img_bytes))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to load image: {str(e)}")

            # 1. Run detection
            try:
                import torch
                # Resolve queries
                text_queries = list(self.default_sensitive_objects)
                if "human face" not in text_queries:
                    text_queries.append("human face")

                # Prep inputs
                size = max(pil_img.size)
                target_sizes = torch.Tensor([[size, size]])
                inputs = self.processor(text=text_queries, images=pil_img, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)

                outputs.logits = outputs.logits.cpu()
                outputs.pred_boxes = outputs.pred_boxes.cpu()
                results = self.processor.post_process_grounded_object_detection(outputs=outputs, target_sizes=target_sizes)
                boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]

                detections = []
                for box, score, label in zip(boxes, scores, labels):
                    if score.item() < threshold:
                        continue
                    detections.append({
                        "box": [int(i) for i in box.tolist()],
                        "score": float(score.item()),
                        "label": text_queries[label.item()]
                    })
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

            # 2. Masking
            try:
                cv_img = self.pil_to_cv2(pil_img)
                mask_detections = detections
                if not is_face_mosaic:
                    mask_detections = [d for d in detections if d["label"] != "human face"]

                processed_cv_img = self.apply_masking(
                    cv_img=cv_img,
                    detections=mask_detections,
                    mask_type=mask_type,
                    mask_face_mosaic=is_face_mosaic
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Masking process failed: {str(e)}")

            # Danger Score
            score_sum = sum(det["score"] for det in detections)
            danger_score = round(min(100.0, score_sum * danger_coefficient), 1)

            # Base64
            original_base64 = self.cv2_to_base64(cv_img)
            processed_base64 = self.cv2_to_base64(processed_cv_img)

            return {
                "original_image": original_base64,
                "processed_image": processed_base64,
                "detections": detections,
                "danger_score": danger_score
            }

        return fastapi_app

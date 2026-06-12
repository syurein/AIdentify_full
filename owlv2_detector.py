import torch
import numpy as np
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection

# Check device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and processor
print(f"Loading OWLv2 model on {device}...")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device)
processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
print("Model loaded successfully.")

# Default array of objects that could leak personal info
DEFAULT_SENSITIVE_OBJECTS = [
    #"human face",
    "license plate",
    "digital screen",
    "document",
    "text",
    "credit card",
    "QR code",
    "barcode",
    "passport",
    "atm screen",
    "keypad",
    "payment terminal",
    "mobile phone screen",
    "signboard",
    "sign"
]

def detect_sensitive_objects(image: Image.Image, text_queries=None, score_threshold=0.1):
    """
    Detects sensitive objects in the given PIL image using OWLv2.
    
    Args:
        image: PIL.Image
        text_queries: List of labels to detect. Defaults to DEFAULT_SENSITIVE_OBJECTS.
        score_threshold: Float, minimum score threshold for detections.
        
    Returns:
        List of dicts: [{'box': [xmin, ymin, xmax, ymax], 'label': label, 'score': score}]
    """
    if text_queries is None or len(text_queries) == 0:
        text_queries = DEFAULT_SENSITIVE_OBJECTS

    # Convert PIL Image to RGB if it isn't
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    width, height = image.size
    
    # Preprocess inputs
    # Note: OWLv2 works on [height, width] target sizes
    target_sizes = torch.Tensor([[height, width]])
    
    inputs = processor(text=[text_queries], images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Move results to CPU
    outputs.logits = outputs.logits.cpu()
    outputs.pred_boxes = outputs.pred_boxes.cpu()
    
    # Post-process detections
    results = processor.post_process_grounded_object_detection(
        outputs=outputs, 
        target_sizes=target_sizes
    )
    
    boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
    
    detections = []
    for box, score, label in zip(boxes, scores, labels):
        score_val = score.item()
        if score_val < score_threshold:
            continue
            
        box_coords = [int(coord) for coord in box.tolist()] # [xmin, ymin, xmax, ymax]
        label_text = text_queries[label.item()]
        
        detections.append({
            "box": box_coords,
            "label": label_text,
            "score": score_val
        })
        
    return detections

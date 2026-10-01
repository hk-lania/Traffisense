import yaml
from ultralytics import YOLO

# Global model instance
_model = None
_config = None

def load_model():
    global _model, _config
    if _model is None:
        with open("config/model_config.yaml", "r") as f:
            _config = yaml.safe_load(f)
        _model = YOLO(_config["model_path"])

def detect(frame):
    """
    Standardized detection interface for ByteTrack.
    Takes an OpenCV frame and returns a list of dictionaries.
    """
    load_model()
    
    results = _model(frame, conf=_config["confidence_threshold"], iou=_config["iou_threshold"], verbose=False)[0]
    
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        class_name = _model.names[class_id]
        
        detections.append({
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "bbox": [int(x1), int(y1), int(x2), int(y2)]
        })
        
    return detections

from pathlib import Path

import yaml
from ultralytics import YOLO

# Paths are resolved relative to this folder, so detect() works no matter
# which directory the program is started from (e.g. run_pipeline.py at the root).
MODULE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = MODULE_DIR / "config" / "model_config.yaml"
FALLBACK_MODEL = MODULE_DIR / "yolov8n.pt"

# Global model instance
_model = None
_config = None


def load_model(model_path=None):
    """Load the YOLO model once.

    Uses `model_path` if given, otherwise `model_path` from the config
    (models/best.pt). If that file is missing, falls back to the pretrained
    yolov8n.pt in this folder so the rest of the pipeline can still run.
    """
    global _model, _config
    if _model is not None:
        return _model
    with open(CONFIG_PATH, "r") as f:
        _config = yaml.safe_load(f)

    path = Path(model_path) if model_path else MODULE_DIR / _config["model_path"]
    if not path.exists():
        print(f"[riddhi_detection] {path} not found - using pretrained {FALLBACK_MODEL.name}. "
              "Put the trained best.pt in riddhi_detection/models/ to use it.")
        path = FALLBACK_MODEL
    _model = YOLO(str(path))
    return _model


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

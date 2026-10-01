<<<<<<< HEAD
# Riddhi's TraffiSense YOLO Module

This module contains the complete local YOLOv8 pipeline for the TraffiSense project. 
It transitions our system from using a cloud API to a fully local model (`models/best.pt`) that Hitarth's ByteTrack pipeline can consume directly.

## Structure
- `dataset/` - Contains the Roboflow dataset (train/val/test and data.yaml)
- `models/` - Contains our trained `best.pt`
- `config/model_config.yaml` - Configuration for training and inference
- `inference.py` - The main entry point for Hitarth (`detect(frame)`)
- `demo.py` - Visual testing on videos
- `train.py` / `evaluate.py` - Scripts for local ML operations

## Integration for ByteTrack
To use this module in the main pipeline:
```python
import cv2
from inference import detect

frame = cv2.imread("traffic.jpg")
detections = detect(frame)

# detections format:
# [
#   {
#       "class_id": 0,
#       "class_name": "car",
#       "confidence": 0.91,
#       "bbox": [x1, y1, x2, y2]
#   }
# ]
```
=======
# IPD
>>>>>>> d61817337fd184d0c4a2aaaf3874c0dbdb103510

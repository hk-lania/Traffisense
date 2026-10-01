# Model Evaluation Report — TraffiSense YOLOv8n

## Training Summary
- **Model**: YOLOv8n (Ultralytics)
- **Epochs**: 20
- **Image Size**: 640×640
- **Batch Size**: 16
- **Hardware**: CPU (Intel Core i5-1235U)
- **Training Time**: ~5 hours

## Dataset
- **Source**: Roboflow — Hornet Engineers / Vehicle Detection (v4)
- **License**: CC BY 4.0
- **Train**: 1,607 images | **Val**: 425 images | **Test**: 344 images
- **Classes**: 9

## Overall Results (Validation Set)

| Metric | Score |
|--------|-------|
| **mAP@0.5** | **83.2%** |
| **mAP@0.5-0.95** | **61.2%** |
| **Precision** | 89.1% |
| **Recall** | 73.0% |

## Per-Class Results

| Class | Images | Instances | Precision | Recall | mAP@0.5 | mAP@0.5-0.95 |
|-------|--------|-----------|-----------|--------|---------|--------------|
| **Car** | 264 | 569 | 96.8% | 89.9% | **98.1%** | 69.1% |
| **Commercial-Vehicle** | 44 | 50 | 91.5% | 86.4% | **93.6%** | 74.1% |
| **Bus** | 33 | 33 | 88.2% | 97.0% | **92.3%** | 79.8% |
| **Pedestrian** | 82 | 88 | 96.8% | 69.6% | **91.2%** | 47.9% |
| **Pickup-Truck** | 81 | 94 | 87.5% | 79.8% | **91.2%** | 68.1% |
| **Bicyclist** | 11 | 11 | 99.6% | 81.8% | **87.1%** | 57.2% |
| **Trailer** | 7 | 7 | 86.0% | 42.9% | **75.7%** | 48.7% |
| **Semi-Truck** | 8 | 8 | 90.4% | 50.0% | **62.0%** | 53.8% |
| **Emergency-Vehicle** | 5 | 5 | 65.2% | 60.0% | **57.6%** | 52.0% |

> **Note**: Emergency-Vehicle, Semi-Truck, and Trailer have low scores due to very few training examples (5–8 images). This is a data limitation, not a model issue.

## Inference Speed (CPU)
- Preprocess: 1.9ms
- Inference: ~135ms per image
- Postprocess: 2.2ms

## Deliverable
The trained model is saved at `models/best.pt` and the standardized inference function is in `inference.py`.

Hitarth's ByteTrack pipeline can call:
```python
from inference import detect
detections = detect(frame)
```

Output format:
```python
[
    {
        "class_id": 2,
        "class_name": "Car",
        "confidence": 0.92,
        "bbox": [x1, y1, x2, y2]
    }
]
```

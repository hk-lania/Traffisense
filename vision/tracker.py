"""
tracker.py
TraffiSense Vision Module

Optimized YOLOv8 + ByteTrack Tracker
"""

import torch
from ultralytics import YOLO

from .config import (
    DEVICE,
    YOLO_MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
)


class Tracker:

    def __init__(self):

        self.device = DEVICE if torch.cuda.is_available() else "cpu"

        print(f"[Tracker] Running on {self.device.upper()}")

        self.model = YOLO(YOLO_MODEL_PATH)

        self.model.to(self.device)

    def track(self, frame):

        results = self.model.track(

            source=frame,

            persist=True,

            tracker="bytetrack.yaml",

            conf=CONFIDENCE_THRESHOLD,

            iou=IOU_THRESHOLD,

            imgsz=960,

            device=self.device,

            verbose=False,

            stream=False,

        )

        return results

    def get_annotated_frame(self, frame):

        results = self.track(frame)

        annotated = results[0].plot(

            labels=True,

            boxes=True,

            conf=True,

            line_width=2,

            font_size=0.7,

        )

        return annotated, results
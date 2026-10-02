"""
config.py
TraffiSense Global Configuration
"""

import os

# =====================================================
# PROJECT PATHS
# =====================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# =====================================================
# HARDWARE
# =====================================================

DEVICE = "cuda"

# =====================================================
# VIDEO
# =====================================================

VIDEO_PATH = os.path.join(
    DATASETS_DIR,
    "AICity22_Track1_MTMC_Tracking",
    "train",
    "S04",
    "c016",
    "vdo.avi",
)

# =====================================================
# YOLO MODEL
# =====================================================

YOLO_MODEL_NAME = "riddhi_yolov8n.pt"

YOLO_MODEL_PATH = os.path.join(
    WEIGHTS_DIR,
    YOLO_MODEL_NAME,
)

# =====================================================
# DETECTION SETTINGS
# =====================================================

CONFIDENCE_THRESHOLD = 0.40
IOU_THRESHOLD = 0.45

# =====================================================
# COCO CLASSES TO DETECT
# =====================================================

# COCO IDs:
# 0 = person
# 1 = bicycle
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck

TARGET_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
]

# =====================================================
# TRACKING
# =====================================================

TRACKER_CONFIG = "bytetrack.yaml"

# =====================================================
# DISPLAY
# =====================================================

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# =====================================================
# DENSITY
# =====================================================

DENSITY_SIGMA = 12.0
MAX_EXPECTED_DENSITY = 50.0

# =====================================================
# QUEUE ESTIMATION
# =====================================================

MOVEMENT_THRESHOLD = 5.0

# =====================================================
# PRESSURE CALCULATION
# =====================================================

DENSITY_WEIGHT = 0.45
QUEUE_WEIGHT = 0.35
WAITING_WEIGHT = 0.20

# =====================================================
# SIGNAL TIMING
# =====================================================

MIN_GREEN_TIME = 15
MAX_GREEN_TIME = 90
DEFAULT_CYCLE_TIME = 120
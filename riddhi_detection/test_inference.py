import cv2
import json
import sys
import os

# Add parent dir to path so we can import inference.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import detect

def test(image_path):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        print("Please place a test image at that path and try again.")
        return

    frame = cv2.imread(image_path)
    if frame is None:
        print("Failed to read image.")
        return

    print(f"Running detection on: {image_path}")
    detections = detect(frame)

    print(f"\nTotal detections: {len(detections)}")
    print("\nStandardized output for ByteTrack:\n")
    print(json.dumps(detections, indent=2))

if __name__ == "__main__":
    # Change this path to any traffic image you have
    image_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_image.png")
    test(image_path)

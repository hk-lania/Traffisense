"""Visual check - run from the project root:  python vision/check_camera_confidence.py <video.mp4>"""

import cv2

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision.camera_confidence import CameraConfidenceEstimator


VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else "traffic.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

estimator = CameraConfidenceEstimator()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    result = estimator.compute(frame)

    cv2.putText(
        frame,
        f"Brightness : {result['brightness']:.2f}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,0),
        2
    )

    cv2.putText(
        frame,
        f"Blur : {result['blur']:.2f}",
        (20,70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,255),
        2
    )

    cv2.putText(
        frame,
        f"Confidence : {result['confidence']:.2f}",
        (20,100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,0,255),
        2
    )

    cv2.imshow(
        "Camera Confidence",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
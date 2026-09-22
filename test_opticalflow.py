import cv2

from optical_flow import OpticalFlowEstimator


# Change this to your traffic video path
VIDEO_PATH = "traffic.mp4"


cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Unable to open video.")
    exit()

flow_estimator = OpticalFlowEstimator()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    result = flow_estimator.compute(frame)

    print(f"Average Motion: {result['average_motion']:.4f}")

    cv2.imshow("Optical Flow Test", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
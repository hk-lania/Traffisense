import cv2
import argparse
import time
from inference import detect

def run_demo(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return
        
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        start_time = time.time()
        detections = detect(frame)
        fps = 1.0 / (time.time() - start_time)
        
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f'{det["class_name"]} {det["confidence"]:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("TraffiSense YOLO Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Path to input video")
    args = parser.parse_args()
    run_demo(args.source)

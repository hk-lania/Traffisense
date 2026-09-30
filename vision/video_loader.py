import cv2
import os

class VideoLoader:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
    def get_info(self):
        if not self.cap.isOpened():
            return None
            
        info = {
            "resolution": (
                int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ),
            "fps": self.cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        info["duration_sec"] = info["frame_count"] / info["fps"] if info["fps"] > 0 else 0
        return info

    def get_frame(self):
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()



import cv2
import numpy as np


class CameraConfidenceEstimator:

    def __init__(self):

        self.max_brightness = 255.0

    def compute(self, frame):

        # Convert image to grayscale
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

       
        brightness = float(np.mean(gray))

        brightness_score = brightness / self.max_brightness

    
        blur_score = cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()

        
        normalized_blur = min(
            blur_score / 500.0,
            1.0
        )

     
        confidence = (
            brightness_score +
            normalized_blur
        ) / 2

        return {

            "brightness": brightness,

            "brightness_score": brightness_score,

            "blur": blur_score,

            "blur_score": normalized_blur,

            "confidence": confidence

        }
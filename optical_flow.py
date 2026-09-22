
import cv2
import numpy as np


class OpticalFlowEstimator:

    def __init__(self):

        self.previous_gray = None

    def compute(self, frame):

    
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # First frame has nothing to compare with
        if self.previous_gray is None:

            self.previous_gray = gray

            return {
                "average_motion": 0.0,
                "flow": None
            }

        
        flow = cv2.calcOpticalFlowFarneback(
            self.previous_gray,
            gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0
        )

        
        magnitude, angle = cv2.cartToPolar(
            flow[..., 0],
            flow[..., 1]
        )

       
        average_motion = float(np.mean(magnitude))

        
        self.previous_gray = gray

        return {
            "average_motion": average_motion,
            "flow": flow
        }
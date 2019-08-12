import cv2
import numpy as np

class SubtractorYYZ():
    def __init__(self, threshold = 3):
        self.prev_frame = None
        self.threshold = threshold

    def apply(self, frame):
        frame = frame.astype(float)
        
        if self.prev_frame is None:
            self.prev_frame = frame
            return np.zeros(frame.shape[:2])

        diff = np.abs(frame - self.prev_frame).sum(-1)
        difff = diff.ravel()
        mask = (diff - difff.mean())/difff.std() > self.threshold
        self.prev_frame = frame

        

        return (mask * 255).astype(np.uint8)

def createBackgroundSubtractorYYZ(threshold = 3):
    return SubtractorYYZ(threshold = threshold)
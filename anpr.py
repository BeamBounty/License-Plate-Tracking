from skimage.segmentation import clear_border
import pytesseract
import numpy as np
import imutils
import cv2

class PyImageSearchANPR:
    # we will have to change minAR and maxAR (aspect ratios) since this implementation
    # focuses on European/international license plates
    def __init__(self, minAR=4, maxAR=5, debug=False):
        self.minAR = minAR
        self.maxAR = maxAR
        self.debug = debug
    
    def debug_imshow(self, title, image, wait_key = False):
        if self.debug:
            cv2.imshow(title, image)

            if wait_key:
                cv2.waitKey(0)
        

    def locate_license_plate_candidates(self, gray, keep=5):
        rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13,5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)
        self.debug_imshow("Blackhat", blackhat)

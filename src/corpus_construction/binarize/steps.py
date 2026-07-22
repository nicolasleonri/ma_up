import cv2
import numpy as np
import math
from skimage.filters import threshold_niblack


def to_grayscale(image):
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


class Binarization:
    @staticmethod
    def none(image):
        return to_grayscale(image)

    @staticmethod
    def basic(image):
        gray = to_grayscale(image)
        _, out = cv2.threshold(
            gray, 127, 255, cv2.THRESH_BINARY
        )
        return out

    @staticmethod
    def otsu(image):
        gray = to_grayscale(image)
        _, out = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return out

    @staticmethod
    def adaptive_mean(image):
        gray = to_grayscale(image)
        gray = cv2.medianBlur(gray, 5)

        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
    
    @staticmethod
    def adaptive_gaussian(image):
        gray = to_grayscale(image)
        gray = cv2.medianBlur(gray, 5)

        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
    
    @staticmethod
    def yannihorne(image):
        gray = to_grayscale(image)
        mean, std = cv2.meanStdDev(gray)

        threshold = mean[0][0] + std[0][0]
        _, output = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        output = output.astype(np.uint8)

        # Morphological opening to remove noise
        kernel = np.ones((3, 3), np.uint8)
        output = cv2.morphologyEx(output, cv2.MORPH_OPEN, kernel, iterations=2)
        return output
    

    @staticmethod
    def niblack(image, window_size: int = 25, k: float = -0.2):
        gray = to_grayscale(image)
        thresh_niblack = threshold_niblack(gray, window_size=window_size, k=k)
        binary = gray <= thresh_niblack

        output = binary.astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        output = cv2.morphologyEx(output, cv2.MORPH_OPEN, kernel, iterations=2)
        return output



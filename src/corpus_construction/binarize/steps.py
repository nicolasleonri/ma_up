"""Binarization steps for OCR preprocessing.

Converts grayscale or color article crop images to grayscale or
high-contrast black-and-white representations optimized for text
recognition.

Available methods (7):
    - None / grayscale baseline
    - Basic Thresholding
    - Otsu Thresholding
    - Adaptive Mean Thresholding
    - Adaptive Gaussian Thresholding
    - Yanni-Horne
    - Niblack
"""

import cv2
import numpy as np
from skimage.filters import threshold_niblack


def _to_grayscale(
    image: np.ndarray,
) -> np.ndarray:
    """Convert a BGR color image to grayscale.

    If the image is already grayscale, return it unchanged.
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

    return image


class Binarization:
    """Collection of image binarization techniques."""

    # @staticmethod
    # def none(
    #     image: np.ndarray,
    # ) -> np.ndarray:
    #     """Return a grayscale version of the input.

    #     This is the non-binarized grayscale baseline and is included
    #     as a reference configuration for OCR evaluation.
    #     """
    #     return _to_grayscale(image)

    @staticmethod
    def none_color(
        image: np.ndarray,
    ) -> np.ndarray:
        return image  # pass through unchanged

    @staticmethod
    def none_grayscale(
        image: np.ndarray,
    ) -> np.ndarray:
        return _to_grayscale(image)

    @staticmethod
    def basic(
        image: np.ndarray,
    ) -> np.ndarray:
        """Apply fixed-threshold binarization at intensity 127."""
        gray = _to_grayscale(image)

        _, output = cv2.threshold(
            gray,
            127,
            255,
            cv2.THRESH_BINARY,
        )

        return output

    @staticmethod
    def otsu(
        image: np.ndarray,
    ) -> np.ndarray:
        """Apply Otsu's global thresholding."""
        gray = _to_grayscale(image)

        _, output = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY
            + cv2.THRESH_OTSU,
        )

        return output

    @staticmethod
    def adaptive_mean(
        image: np.ndarray,
    ) -> np.ndarray:
        """Apply adaptive mean thresholding.

        A median blur is applied first to reduce sensitivity to noise.
        """
        gray = _to_grayscale(image)

        gray = cv2.medianBlur(
            gray,
            5,
        )

        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )

    @staticmethod
    def adaptive_gaussian(
        image: np.ndarray,
    ) -> np.ndarray:
        """Apply adaptive Gaussian thresholding.

        A median blur is applied first to reduce sensitivity to noise.
        """
        gray = _to_grayscale(image)

        gray = cv2.medianBlur(
            gray,
            5,
        )

        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )

    @staticmethod
    def yannihorne(
        image: np.ndarray,
    ) -> np.ndarray:
        """Apply Yanni-Horne thresholding.

        The threshold is calculated as:

            mean + standard deviation

        Morphological opening is then applied to reduce small noise
        artifacts.
        """
        gray = _to_grayscale(image)

        mean, std = cv2.meanStdDev(
            gray
        )

        threshold = (
            mean[0][0]
            + std[0][0]
        )

        _, output = cv2.threshold(
            gray,
            threshold,
            255,
            cv2.THRESH_BINARY,
        )

        output = output.astype(
            np.uint8
        )

        kernel = np.ones(
            (3, 3),
            dtype=np.uint8,
        )

        output = cv2.morphologyEx(
            output,
            cv2.MORPH_OPEN,
            kernel,
            iterations=2,
        )

        return output

    @staticmethod
    def niblack(
        image: np.ndarray,
        window_size: int = 25,
        k: float = -0.2,
    ) -> np.ndarray:
        """Apply Niblack local thresholding.

        Args:
            image: Input grayscale or color image.
            window_size: Local thresholding window size.
            k: Niblack weighting parameter.

        Returns:
            Binarized image.
        """
        if window_size <= 0:
            raise ValueError(
                "window_size must be greater than zero."
            )

        if window_size % 2 == 0:
            raise ValueError(
                "window_size must be an odd number."
            )

        gray = _to_grayscale(image)

        threshold = threshold_niblack(
            gray,
            window_size=window_size,
            k=k,
        )

        output = (
            gray <= threshold
        ).astype(
            np.uint8
        ) * 255

        kernel = np.ones(
            (3, 3),
            dtype=np.uint8,
        )

        output = cv2.morphologyEx(
            output,
            cv2.MORPH_OPEN,
            kernel,
            iterations=2,
        )

        return output
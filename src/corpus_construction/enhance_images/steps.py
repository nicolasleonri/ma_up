"""Enhancement steps for newspaper page images.

Applies non-destructive classical improvements — contrast normalization,
denoising, and sharpening — without hallucinating new pixel information.
Intended to run on raw downloaded images before layout detection.
"""

import cv2
import numpy as np


class Enhancement:

    @staticmethod
    def clahe(image, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
        """Contrast Limited Adaptive Histogram Equalization.

        Fixes uneven lighting and low contrast across the page without
        touching pixels that are already well-exposed. Operates per-channel
        in LAB color space to avoid shifting hue.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_enhanced = clahe.apply(l)

        enhanced_lab = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def denoise(image, h: float = 10, template_window: int = 7, search_window: int = 21):
        """Non-local means denoising for color images.

        Removes JPEG compression artifacts and scanner noise without
        blurring text edges. ``h`` controls filter strength — higher values
        remove more noise but may soften fine strokes.
        """
        return cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h=h,
            hColor=h,
            templateWindowSize=template_window,
            searchWindowSize=search_window,
        )

    @staticmethod
    def unsharp_mask(image, kernel_size: int = 5, sigma: float = 1.0, amount: float = 1.5):
        """Unsharp masking to sharpen text edges.

        Unlike super-resolution, this only enhances existing edges — it
        cannot invent detail that isn't already in the image, making it
        safe for OCR input.
        """
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
        return sharpened

    @staticmethod
    def full(image, clip_limit: float = 2.0, denoise_h: float = 10, sharpen_amount: float = 1.5):
        """Apply the full enhancement chain: CLAHE → denoise → unsharp mask."""
        image = Enhancement.clahe(image, clip_limit=clip_limit)
        image = Enhancement.denoise(image, h=denoise_h)
        image = Enhancement.unsharp_mask(image, amount=sharpen_amount)
        return image
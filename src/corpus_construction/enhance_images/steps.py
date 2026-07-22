"""Image preprocessing steps for OCR.

The preprocessing pipeline is divided into three sequential stages:

1. Contrast normalization
2. Denoising
3. Sharpening

The available techniques are:

Contrast (2):
    - None
    - CLAHE

Denoising (9):
    - None
    - Mean Filter
    - Gaussian Filter
    - Median Filter
    - Conservative Filter
    - Laplacian Filter
    - Frequency Filtering
    - Crimmins Speckle Removal
    - Unsharp Filter

Sharpening (3):
    - None
    - Unsharp Masking
    - Stroke-Width Enhancement
"""

import cv2
import numpy as np


class Contrast:
    """Contrast normalization techniques."""

    @staticmethod
    def none(image):
        """Do not apply contrast enhancement."""
        return image.copy()

    @staticmethod
    def clahe(
        image,
        clip_limit: float = 2.0,
        tile_grid_size: tuple = (8, 8),
    ):
        """Apply Contrast Limited Adaptive Histogram Equalization.

        CLAHE is applied to the L channel in LAB color space to improve
        local contrast while preserving the original color information.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size,
        )

        l_enhanced = clahe.apply(l)

        enhanced_lab = cv2.merge(
            [l_enhanced, a, b]
        )

        return cv2.cvtColor(
            enhanced_lab,
            cv2.COLOR_LAB2BGR,
        )


class Denoising:
    """Denoising and noise-removal techniques."""

    @staticmethod
    def none(image):
        """Do not apply denoising."""
        return image.copy()

    @staticmethod
    def mean_filter(image):
        """Apply a mean filter."""
        return cv2.blur(image, (3, 3))

    @staticmethod
    def gaussian_filter(image):
        """Apply a Gaussian filter."""
        return cv2.GaussianBlur(
            image,
            (3, 3),
            0,
        )

    @staticmethod
    def median_filter(image):
        """Apply a median filter."""
        return cv2.medianBlur(
            image,
            3,
        )

    @staticmethod
    def conservative_filter(
        image,
        kernel_size: int = 3,
    ):
        """Apply a conservative noise reduction filter.

        The center pixel is compared against the minimum and maximum
        intensity values of its neighbors. The center pixel itself is
        excluded from the min/max calculation.

        If the center pixel is outside the range defined by its neighbors,
        it is replaced by the nearest valid boundary value. Otherwise,
        it remains unchanged.

        This is particularly useful for removing isolated impulse noise
        while preserving edges.
        """
        if kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size must be an odd number."
            )

        pad_size = kernel_size // 2

        # Handle grayscale and color images.
        if len(image.shape) == 2:
            padded = np.pad(
                image,
                pad_size,
                mode="edge",
            )
        else:
            padded = np.pad(
                image,
                (
                    (pad_size, pad_size),
                    (pad_size, pad_size),
                    (0, 0),
                ),
                mode="edge",
            )

        result = image.copy()

        height, width = image.shape[:2]

        for i in range(height):
            for j in range(width):

                # Extract neighborhood.
                region = padded[
                    i : i + kernel_size,
                    j : j + kernel_size,
                ]

                center = region[
                    pad_size,
                    pad_size,
                ]

                # Remove center pixel from neighborhood.
                if len(image.shape) == 2:

                    neighbors = np.delete(
                        region.flatten(),
                        pad_size * kernel_size + pad_size,
                    )

                else:

                    neighbors = np.delete(
                        region.reshape(
                            -1,
                            image.shape[2],
                        ),
                        pad_size * kernel_size + pad_size,
                        axis=0,
                    )

                min_val = np.min(
                    neighbors,
                    axis=0,
                )

                max_val = np.max(
                    neighbors,
                    axis=0,
                )

                # Conservative filtering:
                # replace only pixels outside the
                # range defined by their neighbors.
                result[i, j] = np.clip(
                    center,
                    min_val,
                    max_val,
                )

        return result

    @staticmethod
    def laplacian_filter(image):
        """Apply Laplacian filtering."""
        laplacian = cv2.Laplacian(
            image,
            cv2.CV_8U,
        )

        return 255 - laplacian

    @staticmethod
    def frequency_filter(
        image,
        radius: int = 30,
    ):
        """Apply frequency-domain filtering.

        A circular low-pass mask is applied in the frequency domain.
        """
        # Frequency filtering expects a single-channel image.
        if len(image.shape) == 3:
            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )
        else:
            gray = image

        dft = cv2.dft(
            np.float32(gray),
            flags=cv2.DFT_COMPLEX_OUTPUT,
        )

        dft_shift = np.fft.fftshift(dft)

        rows, cols = gray.shape
        crow = rows // 2
        ccol = cols // 2

        mask = np.zeros(
            (rows, cols, 2),
            np.uint8,
        )

        y, x = np.ogrid[
            :rows,
            :cols,
        ]

        mask_area = (
            (x - ccol) ** 2
            + (y - crow) ** 2
            <= radius ** 2
        )

        mask[mask_area] = 1

        filtered = dft_shift * mask

        filtered = np.fft.ifftshift(
            filtered
        )

        image_back = cv2.idft(
            filtered
        )

        image_back = cv2.magnitude(
            image_back[:, :, 0],
            image_back[:, :, 1],
        )

        image_back = cv2.normalize(
            image_back,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        )

        return image_back.astype(
            np.uint8
        )

    @staticmethod
    def crimmins_speckle_removal(
        image,
    ):
        """Apply Crimmins speckle removal."""
        if len(image.shape) == 3:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )

        output = image.copy().astype(
            np.int32
        )

        for _ in range(2):
            for i in range(
                1,
                image.shape[0] - 1,
            ):
                for j in range(
                    1,
                    image.shape[1] - 1,
                ):
                    current_pixel = output[
                        i,
                        j,
                    ]

                    neighbors = [
                        output[i - 1, j],
                        output[i + 1, j],
                        output[i, j - 1],
                        output[i, j + 1],
                    ]

                    median = np.median(
                        neighbors
                    )

                    mean = np.mean(
                        neighbors
                    )

                    if (
                        abs(
                            current_pixel
                            - median
                        )
                        >
                        abs(
                            current_pixel
                            - mean
                        )
                    ):
                        output[
                            i,
                            j
                        ] = median

        return output.astype(
            np.uint8
        )

    @staticmethod
    def unsharp_filter(
        image,
        kernel_size: int = 5,
        sigma: float = 1.0,
        amount: float = 1.5,
    ):
        """Apply unsharp filtering as a denoising-stage technique."""
        blurred = cv2.GaussianBlur(
            image,
            (kernel_size, kernel_size),
            sigma,
        )

        sharpened = (
            float(amount + 1) * image
            - float(amount) * blurred
        )

        sharpened = np.clip(
            sharpened,
            0,
            255,
        )

        return sharpened.astype(
            np.uint8
        )


class Sharpening:
    """Sharpening techniques."""

    @staticmethod
    def none(image):
        """Do not apply sharpening."""
        return image.copy()

    @staticmethod
    def unsharp_mask(
        image,
        kernel_size: int = 5,
        sigma: float = 1.0,
        amount: float = 1.5,
    ):
        """Apply Unsharp Masking.

        This is intentionally separate from Denoising.unsharp_filter
        because the experiment defines them as different techniques
        in different sequential stages.
        """
        blurred = cv2.GaussianBlur(
            image,
            (kernel_size, kernel_size),
            sigma,
        )

        sharpened = cv2.addWeighted(
            image,
            1 + amount,
            blurred,
            -amount,
            0,
        )

        return sharpened

    @staticmethod
    def stroke_width_enhancement(
        image,
        kernel_sizes=(3, 5, 7),
    ):
        """Enhance text strokes using multi-scale morphological processing.

        The image is converted to grayscale and inverted so that dark text
        appears as bright foreground. Morphological opening is then applied
        at multiple scales to detect and enhance structures corresponding
        to different stroke widths.

        The maximum response across scales is normalized and combined with
        the original grayscale image.

        This is intended to enhance existing text strokes before OCR without
        introducing artificial high-frequency detail.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )
        else:
            gray = image.copy()

        # Ensure uint8 input.
        gray = np.clip(
            gray,
            0,
            255,
        ).astype(np.uint8)

        # Invert so dark newspaper text becomes
        # bright foreground.
        inverted = 255 - gray

        responses = []

        for kernel_size in kernel_sizes:

            if kernel_size % 2 == 0:
                raise ValueError(
                    "All kernel sizes must be odd."
                )

            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (
                    kernel_size,
                    kernel_size,
                ),
            )

            # Morphological opening preserves
            # structures compatible with the
            # selected scale.
            opened = cv2.morphologyEx(
                inverted,
                cv2.MORPH_OPEN,
                kernel,
            )

            responses.append(
                opened
            )

        # Combine responses from all scales.
        stroke_response = np.maximum.reduce(
            responses
        )

        # Normalize the stroke response.
        stroke_response = cv2.normalize(
            stroke_response,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        )

        stroke_response = stroke_response.astype(
            np.uint8
        )

        # Combine the detected stroke structures
        # with the original grayscale image.
        enhanced = cv2.addWeighted(
            gray,
            0.7,
            255 - stroke_response,
            0.3,
            0,
        )

        return enhanced
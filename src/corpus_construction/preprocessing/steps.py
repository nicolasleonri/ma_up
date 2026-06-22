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


class NoiseRemoval:

    @staticmethod
    def none(image):
        return image

    @staticmethod
    def mean_filter(image):
        return cv2.blur(image, (3,3))

    @staticmethod
    def gaussian_filter(image):
        return cv2.GaussianBlur(image,(3,3),0)

    @staticmethod
    def median_filter(image):
        return cv2.medianBlur(image,3)
    
    @staticmethod
    def conservative_filter(image, kernel_size: int = 3):
        pad_size = kernel_size // 2
        padded = np.pad(image, pad_size, mode='edge')
        result = np.zeros_like(image)

        for i in range(pad_size, padded.shape[0] - pad_size):
            for j in range(pad_size, padded.shape[1] - pad_size):
                region = padded[i - pad_size:i + pad_size +
                                1, j - pad_size:j + pad_size + 1]
                min_val = np.min(region)
                max_val = np.max(region)
                if image[i - pad_size, j - pad_size] < min_val:
                    result[i - pad_size, j - pad_size] = min_val
                elif image[i - pad_size, j - pad_size] > max_val:
                    result[i - pad_size, j - pad_size] = max_val
                else:
                    result[i - pad_size, j - pad_size] = image[i - pad_size, j - pad_size]
        return result
    
    @staticmethod
    def laplacian_filter(image):
        laplacian = cv2.Laplacian(image, cv2.CV_8U)
        inverted_laplacian = 255 - laplacian # Invert to highlight dark edges on light background
        return inverted_laplacian
    
    @staticmethod
    def frequency_filter(image):
        dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        rows, cols = image.shape[:2]
        crow, ccol = rows // 2, cols // 2

        # Create circular mask
        mask = np.zeros((rows, cols, 2), np.uint8)
        r = 30
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
        mask[mask_area] = 1

        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        return cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    
    @staticmethod
    def crimmins_speckle_removal(image):
        output = image.copy().astype(np.int32)
        total_iterations = 2 * (image.shape[0] - 2) * (image.shape[1] - 2)

        current_iteration = 0
        for _ in range(2):
            for i in range(1, image.shape[0] - 1):
                for j in range(1, image.shape[1] - 1):
                    current_iteration += 1
                    current_pixel = output[i, j]
                    neighbors = [output[i-1, j], output[i+1, j],
                                 output[i, j-1], output[i, j+1]]
                    med = np.median(neighbors)
                    if abs(current_pixel - med) > abs(current_pixel - np.mean(neighbors)):
                        output[i, j] = med

        return output.astype(np.uint8)

    @staticmethod
    def unsharp_filter(image, kernel_size: int = 5, sigma: float = 1.0, amount: float = 1.5, threshold: int = 0):
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        sharpened = float(amount + 1) * image - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype(np.uint8)

        if threshold > 0:
            low_contrast_mask = np.absolute(image - blurred) < threshold
            np.copyto(sharpened, image, where=low_contrast_mask)
        return sharpened

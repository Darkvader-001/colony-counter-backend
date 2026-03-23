"""
Core segmentation: dual thresholding (adaptive + Otsu),
morphological cleanup and Watershed separation.
"""
import cv2
import numpy as np
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi


def create_binary_mask(l_channel, a_channel):
    bacterial_mask = cv2.adaptiveThreshold(
        a_channel, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )
    _, fungal_mask = cv2.threshold(
        l_channel, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return cv2.bitwise_or(bacterial_mask, fungal_mask)


def clean_mask(binary):
    kernel_open = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=2)
    kernel_close = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    return closed


def apply_watershed(binary):
    distance = ndi.distance_transform_edt(binary)
    coords = peak_local_max(
        distance,
        footprint=np.ones((5, 5)),
        labels=binary,
        min_distance=8
    )
    marker_mask = np.zeros(distance.shape, dtype=bool)
    marker_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(marker_mask)
    labels = watershed(-distance, markers, mask=binary)
    return labels


def segment_image(l_channel, a_channel, original_image=None):
    binary = create_binary_mask(l_channel, a_channel)
    cleaned = clean_mask(binary)
    labels = apply_watershed(cleaned)
    return labels, cleaned
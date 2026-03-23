"""
Image preprocessing: LAB color conversion, CLAHE enhancement,
and illumination correction via background subtraction.
"""
import cv2
import numpy as np


def correct_illumination(channel):
    background = cv2.GaussianBlur(channel, (101, 101), 0)
    corrected = cv2.subtract(background, channel)
    corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
    return corrected


def preprocess_image(image):
    image = cv2.resize(image, (800, 800))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, _ = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    l_corrected = correct_illumination(l_enhanced)
    a_corrected = correct_illumination(a_channel)

    return image, l_corrected, a_corrected
"""
Petri dish detection and circular crop module.
Uses Hough Circle Transform to isolate dish from background.
"""
import cv2
import numpy as np


def detect_and_crop_dish(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=image.shape[0] // 2,
        param1=50,
        param2=30,
        minRadius=int(min(image.shape[:2]) * 0.3),
        maxRadius=int(min(image.shape[:2]) * 0.55)
    )

    if circles is None:
        print("[WARN] No dish circle detected — using full image.")
        return image, None

    x, y, r = np.round(circles[0][0]).astype(int)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (x, y), r - 5, 255, -1)
    result = cv2.bitwise_and(image, image, mask=mask)

    x1, y1 = max(x - r, 0), max(y - r, 0)
    x2, y2 = min(x + r, image.shape[1]), min(y + r, image.shape[0])
    cropped = result[y1:y2, x1:x2]

    dish_info = {"center": (x, y), "radius": r, "crop_offset": (x1, y1)}
    return cropped, dish_info
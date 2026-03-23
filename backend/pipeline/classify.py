"""
Colony contour extraction, size/shape filtering,
and bacterial vs fungal classification by circularity.
"""
import cv2
import numpy as np

MIN_COLONY_AREA = 250
MAX_COLONY_AREA = 5500
MIN_CIRCULARITY = 0.45


def classify_colonies(labels, original_image):
    bacterial, fungal, rejected = [], [], []

    for label_id in np.unique(labels):
        if label_id == 0:
            continue

        colony_mask = np.uint8(labels == label_id) * 255
        contours, _ = cv2.findContours(
            colony_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        cnt = contours[0]
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        if area < MIN_COLONY_AREA or area > MAX_COLONY_AREA:
            rejected.append({"reason": "size", "area": area})
            continue

        if perimeter == 0:
            continue

        circularity = (4 * np.pi * area) / (perimeter ** 2)

        if circularity < MIN_CIRCULARITY:
            rejected.append({"reason": "circularity", "area": area})
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        colony_data = {
            "label_id": int(label_id),
            "area": float(area),
            "circularity": float(round(circularity, 3)),
            "bbox": [int(x), int(y), int(w), int(h)],
            "contour": cnt
        }

        if circularity >= 0.70:
            bacterial.append(colony_data)
        else:
            fungal.append(colony_data)

    return {
        "bacterial": bacterial,
        "fungal": fungal,
        "total": len(bacterial) + len(fungal),
        "rejected": len(rejected)
    }
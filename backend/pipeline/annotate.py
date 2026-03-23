"""
Visual annotation: draws coloured bounding circles and
count labels onto the processed dish image.
"""
import cv2
import numpy as np


def draw_annotations(image, results):
    annotated = image.copy()

    for i, colony in enumerate(results["bacterial"]):
        x, y, w, h = colony["bbox"]
        cx, cy = x + w // 2, y + h // 2
        cv2.circle(annotated, (cx, cy), max(w, h) // 2, (0, 200, 0), 2)
        cv2.putText(annotated, str(i + 1), (cx - 5, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 0), 1)

    offset = len(results["bacterial"])
    for i, colony in enumerate(results["fungal"]):
        x, y, w, h = colony["bbox"]
        cx, cy = x + w // 2, y + h // 2
        cv2.circle(annotated, (cx, cy), max(w, h) // 2, (0, 140, 255), 2)
        cv2.putText(annotated, str(offset + i + 1), (cx - 5, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 140, 255), 1)

    cv2.rectangle(annotated, (5, 5), (220, 65), (0, 0, 0), -1)
    cv2.putText(annotated, f"Total: {results['total']}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(annotated, f"Bacterial: {len(results['bacterial'])}", (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    cv2.putText(annotated, f"Fungal: {len(results['fungal'])}", (120, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1)

    return annotated
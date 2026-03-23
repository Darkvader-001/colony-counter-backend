from inference_sdk import InferenceHTTPClient
import os
import sys
import cv2

sys.path.insert(0, "C:/Projects/colony-counter/backend")

from pipeline.crop import detect_and_crop_dish
from pipeline.preprocess import preprocess_image
from pipeline.segment import segment_image
from pipeline.classify import classify_colonies
from pipeline.roboflow_detect import detect_colonies_roboflow, classify_roboflow_results

API_KEY = "uBjV671qXscY7fmuy81m"
DATASET_PATH = "C:/Projects/colony-counter/dataset/CEMTimages"

images = [f for f in os.listdir(DATASET_PATH)
          if '300' in f and f.endswith('.JPG')][:10]

print(f"Testing 3-layer pipeline on {len(images)} images")
print("="*60)

total_error = 0
for filename in images:
    path = os.path.join(DATASET_PATH, filename)
    image = cv2.imread(path)

    # Layer 1 - crop and preprocess
    cropped, _ = detect_and_crop_dish(image)
    resized, l_ch, a_ch = preprocess_image(cropped)

    # Layer 2 - Roboflow
    rf_colonies, rf_success, method = detect_colonies_roboflow(
        resized, API_KEY
    )

    if rf_success:
        results = classify_roboflow_results(rf_colonies)
        layer = "Roboflow"
    else:
        # Layer 3 - classical fallback
        labels, _ = segment_image(l_ch, a_ch, original_image=resized)
        results = classify_colonies(labels, resized)
        layer = "Classical"

    count = results["total"]
    error = abs(count - 300) / 300 * 100
    total_error += error
    print(f"{filename[:35]:35} | {count:4} detected | "
          f"{error:6.1f}% error | {layer}")

print("="*60)
print(f"Mean error: {total_error/len(images):.1f}%")
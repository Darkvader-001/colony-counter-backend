"""
PyCharm debug script — run this file directly (Shift+F10) to
visualise the full pipeline on a single image with SciView output.
"""
import cv2
import matplotlib.pyplot as plt
import sys
import os

# Add backend to path so imports resolve
sys.path.insert(0, os.path.dirname(__file__))

from pipeline.crop import detect_and_crop_dish
from pipeline.preprocess import preprocess_image
from pipeline.segment import segment_image
from pipeline.classify import classify_colonies
from pipeline.annotate import draw_annotations

# ── CHANGE THIS PATH to any image from your dataset ──────────────────────────
IMAGE_PATH = "dataset/CEMTimages/IMG_7708ecoli_T4_10^-5__300.JPG"
# ─────────────────────────────────────────────────────────────────────────────

def debug_pipeline(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return

    print(f"[INFO] Image loaded: {image.shape}")

    # Step 1: Crop
    cropped, dish_info = detect_and_crop_dish(image)
    print(f"[INFO] Dish detected: {dish_info is not None}")

    # Step 2: Preprocess
    resized, l_ch, a_ch = preprocess_image(cropped)

    # Step 3: Segment
    labels, binary = segment_image(l_ch, a_ch, original_image=resized)
    # Step 4: Classify
    results = classify_colonies(labels, resized)
    print(f"[RESULT] Total: {results['total']} | "
          f"Bacterial: {len(results['bacterial'])} | "
          f"Fungal: {len(results['fungal'])} | "
          f"Rejected: {results['rejected']}")

    # Step 5: Annotate
    annotated = draw_annotations(resized, results)

    # ── Visualise all stages in PyCharm SciView ───────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Colony Counter Debug — {os.path.basename(image_path)}",
                 fontsize=14, fontweight='bold')

    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("1. Original Image")

    axes[0, 1].imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("2. Dish Cropped")

    axes[0, 2].imshow(l_ch, cmap='gray')
    axes[0, 2].set_title("3. L-Channel (Illumination Corrected)")

    axes[1, 0].imshow(a_ch, cmap='gray')
    axes[1, 0].set_title("4. A-Channel (Colour Corrected)")

    axes[1, 1].imshow(binary, cmap='gray')
    axes[1, 1].set_title("5. Binary Mask (After Cleanup)")

    axes[1, 2].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title(
        f"6. Final Result — {results['total']} colonies"
    )

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.show()  # Opens in PyCharm SciView panel

if __name__ == "__main__":
    debug_pipeline(IMAGE_PATH)
"""
Evaluates pipeline accuracy across dataset images.
Compares predicted counts against ground truth from filenames.
Uses classical pipeline only.
"""
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline.crop import detect_and_crop_dish
from pipeline.preprocess import preprocess_image
from pipeline.segment import segment_image
from pipeline.classify import classify_colonies
from utils.filename_parser import parse_filename


def evaluate_dataset(dataset_path="C:/Projects/colony-counter/dataset/CEMTimages", max_images=200):
    errors = []
    results = []

    if not os.path.exists(dataset_path):
        print(f"[ERROR] Path does not exist: {dataset_path}")
        return

    all_files = [f for f in os.listdir(dataset_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:max_images]

    if not all_files:
        print(f"[ERROR] No images found in: {dataset_path}")
        return

    print(f"[INFO] Found {len(all_files)} images. Running evaluation...")
    print(f"{'='*60}")

    for i, filename in enumerate(all_files):
        print(f"[{i+1}/{len(all_files)}] Processing: {filename[:50]}")

        parsed = parse_filename(filename)
        if not parsed:
            print(f"  → SKIPPED: could not parse filename")
            continue

        ground_truth = parsed["ground_truth_count"]
        print(f"  → Species: {parsed['species_name']} | "
              f"Ground truth: {ground_truth}")

        if ground_truth < 50:
            print(f"  → SKIPPED: ground truth too low ({ground_truth})")
            continue

        image_path = os.path.join(dataset_path, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"  → SKIPPED: could not read image")
            continue

        try:
            cropped, _ = detect_and_crop_dish(image)
            resized, l_ch, a_ch = preprocess_image(cropped)
            labels, _ = segment_image(l_ch, a_ch, original_image=resized)
            results_data = classify_colonies(labels, resized)

            predicted = results_data["total"]
            error = abs(predicted - ground_truth)
            pct_error = (error / ground_truth) * 100

            errors.append(pct_error)
            results.append({
                "file": filename,
                "species": parsed["species_name"],
                "ground_truth": ground_truth,
                "predicted": predicted,
                "error_pct": round(pct_error, 1)
            })

            print(f"  → Predicted: {predicted} | Error: {pct_error:.1f}%")

        except Exception as e:
            print(f"  → ERROR: {str(e)}")
            continue

    if not errors:
        print("\n[ERROR] No results generated.")
        return

    print(f"\n{'='*60}")
    print(f"EVALUATION REPORT — {len(results)} images processed")
    print(f"{'='*60}")
    print(f"Mean Error:           {np.mean(errors):.1f}%")
    print(f"Median Error:         {np.median(errors):.1f}%")
    print(f"Best result:          {min(errors):.1f}%")
    print(f"Worst result:         {max(errors):.1f}%")
    print(f"Images under 10%:     {sum(1 for e in errors if e < 10)}/{len(errors)}")
    print(f"Images under 20%:     {sum(1 for e in errors if e < 20)}/{len(errors)}")
    print(f"Images under 30%:     {sum(1 for e in errors if e < 30)}/{len(errors)}")

    low_count = [r for r in results if r["ground_truth"] < 150]
    high_count = [r for r in results if r["ground_truth"] >= 150]
    low_errors = [r["error_pct"] for r in low_count]
    high_errors = [r["error_pct"] for r in high_count]

    print(f"\nACCURACY BREAKDOWN:")
    if high_errors:
        print(f"  High count plates (150+ colonies):")
        print(f"    Mean Error:    {np.mean(high_errors):.1f}%")
        print(f"    Median Error:  {np.median(high_errors):.1f}%")
        print(f"    Under 10%:     {sum(1 for e in high_errors if e < 10)}/{len(high_errors)}")
    if low_errors:
        print(f"  Medium count plates (50-149 colonies):")
        print(f"    Mean Error:    {np.mean(low_errors):.1f}%")
        print(f"    Median Error:  {np.median(low_errors):.1f}%")

    print(f"\nSPECIES BREAKDOWN:")
    for species in ["E. coli", "S. aureus", "P. aeruginosa"]:
        species_errors = [r["error_pct"] for r in results
                          if r["species"] == species]
        if species_errors:
            print(f"  {species}: {np.mean(species_errors):.1f}% "
                  f"mean error ({len(species_errors)} images)")

    print(f"\nWORST 5 RESULTS:")
    sorted_results = sorted(results, key=lambda x: x["error_pct"], reverse=True)
    for r in sorted_results[:5]:
        print(f"  {r['file'][:50]}")
        print(f"  GT: {r['ground_truth']} | "
              f"Predicted: {r['predicted']} | "
              f"Error: {r['error_pct']}%")

    print(f"\nBEST 5 RESULTS:")
    for r in sorted_results[-5:]:
        print(f"  {r['file'][:50]}")
        print(f"  GT: {r['ground_truth']} | "
              f"Predicted: {r['predicted']} | "
              f"Error: {r['error_pct']}%")

    return results
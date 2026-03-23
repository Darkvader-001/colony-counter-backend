"""
Roboflow colony detection with classical pipeline fallback.
Uses confidence-based switching between detection methods.
"""
import cv2


def detect_colonies_roboflow(image, api_key, fallback_threshold=50):
    """
    Detect colonies using Roboflow API.
    Returns (colonies, success, method_used)
    """
    try:
        from inference_sdk import InferenceHTTPClient

        client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key
        )

        # Save temp image for API
        temp_path = "temp_dish.jpg"
        cv2.imwrite(temp_path, image)

        result = client.infer(temp_path, model_id="petri-dishes-mjefq/1")
        predictions = result.get("predictions", [])

        colonies = []
        for pred in predictions:
            x = int(pred["x"] - pred["width"] / 2)
            y = int(pred["y"] - pred["height"] / 2)
            w = int(pred["width"])
            h = int(pred["height"])

            colonies.append({
                "bbox": [x, y, w, h],
                "confidence": round(pred["confidence"], 3),
                "area": float(w * h),
                "circularity": 1.0,
                "label_id": len(colonies) + 1
            })

        # If Roboflow returns too few detections
        # it likely failed on this image style
        if len(colonies) < fallback_threshold:
            print(f"  → Roboflow returned {len(colonies)} "
                  f"(below threshold {fallback_threshold}) "
                  f"— flagging for classical fallback")
            return colonies, False, "roboflow_low"

        print(f"  → Roboflow detected {len(colonies)} colonies")
        return colonies, True, "roboflow"

    except Exception as e:
        print(f"[WARN] Roboflow failed: {e}")
        return [], False, "error"


def classify_roboflow_results(colonies):
    """
    Classify detections into bacterial/fungal by area.
    """
    bacterial = []
    fungal = []

    for colony in colonies:
        if colony["area"] > 2000:
            fungal.append(colony)
        else:
            bacterial.append(colony)

    return {
        "bacterial": bacterial,
        "fungal": fungal,
        "total": len(bacterial) + len(fungal),
        "rejected": 0
    }
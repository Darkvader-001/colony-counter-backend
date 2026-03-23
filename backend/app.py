from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import time
import threading

from pipeline.crop import detect_and_crop_dish
from pipeline.preprocess import preprocess_image
from pipeline.segment import segment_image
from pipeline.classify import classify_colonies
from pipeline.annotate import draw_annotations

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
CORS(app)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "pipeline": "classical", "version": "1.0"})

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    start_time = time.time()

    file = request.files['image']
    img_array = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Invalid image file"}), 400

    # Resize large images immediately to speed up processing
    h, w = image.shape[:2]
    if w > 1000:
        scale = 1000 / w
        image = cv2.resize(image, (1000, int(h * scale)))

    try:
        cropped, dish_info = detect_and_crop_dish(image)
        resized, l_ch, a_ch = preprocess_image(cropped)
        labels, binary = segment_image(l_ch, a_ch, original_image=resized)
        results = classify_colonies(labels, resized)
        annotated = draw_annotations(resized, results)

        _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        img_b64 = base64.b64encode(buffer).decode('utf-8')

        processing_ms = int((time.time() - start_time) * 1000)
        print(f"[INFO] Done in {processing_ms}ms — Total: {results['total']}")

        return jsonify({
            "total_count": results["total"],
            "bacterial_count": len(results["bacterial"]),
            "fungal_count": len(results["fungal"]),
            "rejected_blobs": results["rejected"],
            "annotated_image": img_b64,
            "processing_time_ms": processing_ms,
            "dish_detected": dish_info is not None
        })

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
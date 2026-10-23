# download_models.py
import os, urllib.request

os.makedirs("models", exist_ok=True)
MODELS = {
    "models/face_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "models/efficientdet_lite0.tflite":
        "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite",
}
for path, url in MODELS.items():
    if not os.path.exists(path):
        print("lade", path)
        urllib.request.urlretrieve(url, path)
        print("  ", round(os.path.getsize(path) / 1e6, 1), "MB")
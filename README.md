# Gesellschaft

Webcam-Demo zur Erkennung von Handynutzung: Das Gesicht wird mit dem MediaPipe
FaceLandmarker verfolgt, daraus werden Kopflage (Pitch/Yaw/Roll) und FPS berechnet.
Parallel sucht ein YOLO-Modell nach dem Objekt `cell phone` (COCO-Klasse 67) und
zeichnet die Boxen ins Bild.

## Voraussetzungen

- Windows, da die Kamera mit `cv2.CAP_DSHOW` geöffnet wird
- Python 3.14
- Pakete: `opencv-python`, `mediapipe`, `ultralytics`, `numpy`
  (getestet mit OpenCV 5.0.0, MediaPipe 1.1.1, Ultralytics 8.4.189, NumPy 2.5.4)

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install opencv-python mediapipe ultralytics numpy
python models.py
```

`models.py` lädt `models/face_landmarker.task` und `models/efficientdet_lite0.tflite`
herunter und überspringt bereits vorhandene Dateien.

## Modelle, die nicht im Repo liegen

Diese Dateien sind per `.gitignore` ausgeschlossen (mehrere MB Binärdaten, teils
unklare Lizenz) und müssen selbst nach `models/` gelegt werden:

- `models/yolo26s.pt` (von `webcam_yolo.py` genutzt) und `models/yolo26n.pt`
  (kleiner und schneller, aber ungenauer), Download aus dem Ultralytics-Release `v8.4.0`:
  `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt`
- `models/phone_detection.xml` (Haar-Cascade, nur für `webcam_cascade.py`):
  `https://raw.githubusercontent.com/raj-shah14/Find-Phone-with-Opencv/master/phonecascade30.xml`

## Skripte

| Datei | Zweck |
| --- | --- |
| `webcam_yolo.py` | Aktuelle Version: FaceLandmarker, Kopflage, FPS und Handy-Erkennung mit YOLO (`classes=[67]`, `conf=0.15`). Beenden mit `q`. |
| `webcam_cascade.py` | Ältere Version: gleiche Pipeline, Handy-Erkennung über die Haar-Cascade `models/phone_detection.xml`. |
| `webcam_prot.py` | Prototyp: öffnet nur die Kamera und zeigt das Bild an. |
| `models.py` | Lädt die MediaPipe-Modelle nach `models/`. |
| `testing.py` | Prüft, ob die Cascade-Datei geladen werden kann. |

## Start

```powershell
python webcam_yolo.py
```

## Lizenz

`webcam_yolo.py` nutzt Ultralytics YOLO, das unter AGPL-3.0 steht. Die Gewichte
werden deshalb nicht mitgeliefert und beim Setup separat geladen.

Offene Punkte stehen in `TODO.md`.

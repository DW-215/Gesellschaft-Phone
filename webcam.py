## Webcam Capture
import time
import cv2
import mediapipe as mp
import math
import numpy as np
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

# Initialize the camera
camera = None
captured_image = None
# Initialize the FPS counter
pTime = 0
t0 = time.monotonic()
# Number of faces to detect
NUM_FACE = 1

# check for webcam devices
for index in range(5):
    candidate = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if candidate.isOpened():
        response, candidate_frame = candidate.read()
        if response and candidate_frame is not None and candidate_frame.size:
            camera, captured_image = candidate, candidate_frame
            print(f'Using camera index {index}')
            break
    candidate.release()
if camera is None:
    raise IOError('Could not open webcam')

# Get the frame width and height
frame_height, frame_width = captured_image.shape[:2]

# Initialize the drawing utils
mpDraw = vision.drawing_utils
mpFaceMesh = vision.FaceLandmarksConnections  # holds the connection lists
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

landmarker = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path="models/face_landmarker.task"),
        running_mode=vision.RunningMode.VIDEO,   # has to be video
        num_faces=NUM_FACE,   # same value as used above
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,                  # for eyes
        output_facial_transformation_matrixes=True,     # headpose
    )
)

# Initialize the FPS counter
# main loop for capturing the video
while True:
    response, captured_image = camera.read()
    if not response or captured_image is None:
        break

    # Process the captured frame
    img_rgb = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(captured_image, cv2.COLOR_BGR2GRAY)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    # Run the landmarker on the frame
    ts_ms = int((time.monotonic() - t0) * 1000)
    result = landmarker.detect_for_video(mp_img, ts_ms)

    # Draw the landmarks of the result we already got from the landmarker
    if result.face_landmarks:
        # one list of 478 landmarks per detected face
        for faceLms in result.face_landmarks:
            mpDraw.draw_landmarks(captured_image, faceLms, mpFaceMesh.FACE_LANDMARKS_TESSELATION, drawSpec, drawSpec)

            # convert the normalized (0..1) coordinates to pixel coordinates
            ih, iw, ic = captured_image.shape
            for ID, lm in enumerate(faceLms):
                x, y = int(lm.x * iw), int(lm.y * ih)
                # uncomment the below lines to inspect the 478 facial landmarks
                cv2.putText(captured_image, str(ID), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                print(ID, x, y)

    # Calculate the head pose
    if len(result.facial_transformation_matrixes) > 0:
        M = result.facial_transformation_matrixes[0]
    else: # fallback
        M = np.eye(4)

    R = M[:3, :3]
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -R[2, 0]))))  # nicken (hoch/runter)
    yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))  # drehen (links/rechts)
    roll = math.degrees(math.atan2(R[2, 1], R[2, 2]))  # schief halten

    # TODO add either object detection and or processing for the positioning of the head
    # Object Detection Logic
    phone_cascade = cv2.CascadeClassifier("") # TODO find model
    found = phone_cascade.detectMultiScale(img_gray, minSize=(20, 20))

    # Display the head pose
    cv2.putText(captured_image, f'Pitch: {pitch:.1f}°', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(captured_image, f'Yaw: {yaw:.1f}°', (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(captured_image, f'Roll: {roll:.1f}°', (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Add fps counter
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(captured_image, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the detected objects
    for (x, y, w, h) in found:
        cv2.rectangle(captured_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the captured frame
    cv2.imshow('Camera', captured_image)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and landmarker objects
camera.release()
landmarker.close()
cv2.destroyAllWindows()

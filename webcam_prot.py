## Webcam Capture

import cv2

camera = None
captured_image = None
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

# Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

while True:
    response, captured_image = camera.read()
    if not response or captured_image is None:
        break

    # Write the frame to the output file
    # out.write(captured_image)

    # Display the captured frame
    cv2.imshow('Camera', captured_image)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
camera.release()
# out.release()
cv2.destroyAllWindows()

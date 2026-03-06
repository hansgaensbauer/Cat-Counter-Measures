from ultralytics import YOLO
import numpy as np
model = YOLO('yolov8n.pt')  # or yolov8s.pt, yolov8m.pt

import cv2

# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

last_frame = cam.read()
while True:
    ret, frame = cam.read()
    # print(np.sum(frame-last_frame))

    # Write the frame to the output file

    results = model.track(frame, classes = [15], persist=True)
    cv2.imshow('Camera', results[0].plot())

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
cv2.destroyAllWindows()
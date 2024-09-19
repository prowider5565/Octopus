from imutils.video import VideoStream
import numpy as np
import insightface
import pickle
import faiss
import json
import time
import uuid
import cv2
import os
import time

from redis_connection import connection


# Load FaceAnalysis model from Redis
face_encoding = connection.get("face_encoding")
redis_index = connection.get("index")
model = pickle.loads(face_encoding)
index = pickle.loads(redis_index)


# Start video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
if not os.path.exists("results"):
    os.mkdir("results")

while True:
    # Capture frame-by-frame
    frame = vs.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    faces = model.get(rgb_frame)

    for face in faces:
        embedding = face.embedding
        D, I = index.search(np.array([embedding]).astype("float32"), 1)
        print(D[0][0])
        # Threshold to determine a match (lower value means more similar)
        if D[0][0] < 500:  # Adjust the threshold as per your requirement
            # Get bounding box coordinates
            bbox = face.bbox.astype(int)  # bbox contains [x1, y1, x2, y2]

            # Draw a green rectangle around the detected face
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

            # Save the screenshot
            screenshot_path = os.path.join("results", str(uuid.uuid4()) + ".jpg")
            cv2.imwrite(screenshot_path, frame)
            print(f"[INFO] Screenshot saved to {screenshot_path}")

            # Release video stream and close window
            vs.stop()
            cv2.destroyAllWindows()
            exit(0)

    # Display the resulting frame for debugging purposes
    cv2.imshow("Frame", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
vs.stop()
cv2.destroyAllWindows()

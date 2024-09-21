from imutils.video import VideoStream
import numpy as np
import insightface
import pickle
import faiss
import uuid
import cv2
import os
from concurrent.futures import ThreadPoolExecutor
from redis_connection import connection

# Load FaceAnalysis model and index from Redis
face_encoding = connection.get("face_encoding")
redis_index = connection.get("index")
reference_names = connection.get("reference_names")

model = pickle.loads(face_encoding)
index = pickle.loads(redis_index)
reference_names = pickle.loads(reference_names)

# Start video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

if not os.path.exists("results"):
    os.mkdir("results")

def process_face(face, frame):
    embedding = face.embedding
    D, I = index.search(np.array([embedding]).astype("float32"), 1)

    # Threshold to determine a match (lower value means more similar)
    if D[0][0] < 500:  # Adjust the threshold as per your requirement
        # Get bounding box coordinates
        bbox = face.bbox.astype(int)  # bbox contains [x1, y1, x2, y2]

        # Draw a green rectangle around the detected face
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        # Get the name of the matched image from the dataset
        matched_name = reference_names[I[0][0]]

        # Put the matched name on the image
        cv2.putText(
            frame, matched_name,
            (bbox[0], bbox[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9, (0, 255, 0), 2
        )

        # Save the screenshot of the detected face with bounding box and name
        screenshot_path = os.path.join("results", str(uuid.uuid4()) + ".jpg")
        cv2.imwrite(screenshot_path, frame)
        print(f"[INFO] Screenshot saved to {screenshot_path}")

# Create a thread pool executor to manage threading
with ThreadPoolExecutor(max_workers=7) as executor:
    while True:
        # Capture frame-by-frame
        frame = vs.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        faces = model.get(rgb_frame)

        # Submit each face for processing in a separate thread
        for face in faces:
            executor.submit(process_face, face, frame.copy())

        # Display the resulting frame with detected faces
        cv2.imshow("Frame", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Cleanup
vs.stop()
cv2.destroyAllWindows()

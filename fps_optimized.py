from imutils.video import VideoStream
import numpy as np
import pickle
import faiss
import uuid
import cv2
import os
import threading

from redis_connection import connection

# Load FaceAnalysis model from Redis
face_encoding = connection.get("face_encoding")
redis_index = connection.get("index")
model = pickle.loads(face_encoding)
index = pickle.loads(redis_index)

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

if not os.path.exists("results"):
    os.mkdir("results")

faces = []
processed_frame = None


def detect_faces(rgb_frame, orig_frame):
    global faces, processed_frame
    faces = model.get(rgb_frame)
    processed_frame = orig_frame


detection_thread = None

while True:
    frame = vs.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow("Frame", frame)
    if detection_thread is None or not detection_thread.is_alive():
        detection_thread = threading.Thread(
            target=detect_faces, args=(rgb_frame, frame.copy())
        )
        detection_thread.start()

    if faces and processed_frame is not None:
        for face in faces:
            embedding = face.embedding
            D, I = index.search(np.array([embedding]).astype("float32"), 1)
            print(D[0][0])
            if D[0][0] < 600:
                bbox = face.bbox.astype(int)
                cv2.rectangle(
                    processed_frame,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    (0, 255, 0),
                    2,
                )
                screenshot_path = os.path.join("results", str(uuid.uuid4()) + ".jpg")
                cv2.imwrite(screenshot_path, processed_frame)
                print(f"[INFO] Screenshot saved to {screenshot_path}")
                vs.stop()
                cv2.destroyAllWindows()
                exit(0)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vs.stop()
cv2.destroyAllWindows()

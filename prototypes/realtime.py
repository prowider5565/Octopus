from imutils.video import VideoStream
import numpy as np
import pickle
import faiss
import cv2
import os
import threading

from redis_connection import connection

face_encoding = connection.get("face_encoding")
redis_index = connection.get("index")
model = pickle.loads(face_encoding)
index = pickle.loads(redis_index)
reference_names = connection.get("reference_names")
reference_names = pickle.loads(reference_names)

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

if not os.path.exists("results"):
    os.mkdir("results")

faces = []
processed_frame = None
lock = threading.Lock()


def process_face(face, processed_frame):
    embedding = face.embedding
    D, I = index.search(np.array([embedding]).astype("float32"), 1)

    if D[0][0] < 600:
        bbox = face.bbox.astype(int)
        cv2.rectangle(
            processed_frame,
            (bbox[0], bbox[1]),
            (bbox[2], bbox[3]),
            (0, 255, 0),
            2,
        )
        image_name = reference_names[I[0][0]]
        cv2.putText(
            processed_frame,
            image_name,
            (bbox[2] - 100, bbox[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def detect_faces(rgb_frame, orig_frame):
    global faces
    faces = model.get(rgb_frame)
    for face in faces:
        process_face(face, orig_frame)


while True:
    frame = vs.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detection_thread = threading.Thread(target=detect_faces, args=(rgb_frame, frame))
    detection_thread.start()
    detection_thread.join()
    cv2.imshow("Processed Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vs.stop()
cv2.destroyAllWindows()

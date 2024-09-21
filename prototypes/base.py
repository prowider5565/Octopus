from imutils.video import VideoStream
import numpy as np
import insightface
import pickle
import faiss
import uuid
import cv2
import sys
import os

from redis_connection import connection


face_encoding = connection.get("face_encoding")
redis_index = connection.get("index")
reference_names = connection.get("reference_names")
model = pickle.loads(face_encoding)
index = pickle.loads(redis_index)
reference_names = pickle.loads(reference_names)

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

if not os.path.exists("results"):
    os.mkdir("results")

while True:
    frame = vs.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = model.get(rgb_frame)
    for face in faces:
        embedding = face.embedding
        D, I = index.search(np.array([embedding]).astype("float32"), 1)
        if D[0][0] < 500:
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            matched_name = reference_names[I[0][0]]
            cv2.putText(
                frame,
                matched_name,
                (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
            screenshot_path = os.path.join("results", str(uuid.uuid4()) + ".jpg")
            cv2.imwrite(screenshot_path, frame)
            print(f"[INFO] Screenshot saved to {screenshot_path}")
            sys.exit(0)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vs.stop()
cv2.destroyAllWindows()

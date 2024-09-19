from imutils.video import VideoStream
import concurrent.futures
import numpy as np
import pickle
import uuid
import time
import cv2
import os

from redis_connection import connection


vs = VideoStream(src=0).start()
# Load FaceAnalysis model from Redis
face_encoding = connection.get("face_encoding")
redis_index = connection.get("index")
model = pickle.loads(face_encoding)
index = pickle.loads(redis_index)


def process_image(frame):
    cv2.setUseOptimized(True)
    cv2.ocl.setUseOpenCL(True)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = model.get(rgb_frame)
    for face in faces:
        embedding = face.embedding
        D, I = index.search(np.array([embedding]).astype("float32"), 1)
        print(D[0][0])
        if D[0][0] < 450:
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            screenshot_path = os.path.join("results", str(uuid.uuid4()) + ".jpg")
            cv2.imwrite(screenshot_path, frame)
            print(f"[INFO] Screenshot saved to {screenshot_path}")
            return D[0][0]


def identify_in_threads(frames):
    # Using ThreadPoolExecutor for concurrent execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        print("Started Processing the face ______________________________________")
        # Submitting tasks with arguments (e.g., range(100) as arguments)
        futures = [executor.submit(process_image, per_frame) for per_frame in frames]
        # Release video stream and close window

        # Retrieving the results
        results = [
            future.result() for future in concurrent.futures.as_completed(futures)
        ]
        return results


def gather_frames(interval=2):
    frames = []
    start_time = time.time()
    print("inside the gather frames function")
    while time.time() - start_time < interval:
        time.sleep(0.01)
        frame = vs.read()
        frames.append(frame)
        cv2.imshow("Frame", frame)

    return frames


def main():
    frames = gather_frames(interval=2)
    result = identify_in_threads(frames)
    print(result)


main()

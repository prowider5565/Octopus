from imutils.video import VideoStream
import numpy as np
import insightface
import pickle
import faiss
import time
import cv2
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from redis_connection import connection

# Load FaceAnalysis model from Redis
redis_data = connection.get("face_encoding")
model = pickle.loads(redis_data)
index = faiss.IndexFlatL2(512)

# Start video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
if not os.path.exists("results"):
    os.mkdir("results")


def process_single_frame(frame):
    """Process a single frame and return the result."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = model.get(rgb_frame)
    embeddings = []

    for face in faces:
        embeddings.append(face.embedding)

        # Get bounding box coordinates for drawing
        bbox = face.bbox.astype(int)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

    # If embeddings exist, perform Faiss search on the frame
    if embeddings:
        D, I = index.search(np.array(embeddings).astype("float32"), 1)
        avg_D = np.mean(D)
        return avg_D, frame
    return None, frame


# Main loop to gather and process frames
executor = ThreadPoolExecutor(max_workers=60)
frames_to_process = []
start_time = time.time()

while True:
    # Capture frame-by-frame
    frame = vs.read()
    frames_to_process.append(frame)

    # Process frames every 2 seconds
    if time.time() - start_time >= 2:
        print(f"[INFO] Processing {len(frames_to_process)} frames in parallel")

        # Submit each frame to be processed in its own thread
        futures = [
            executor.submit(process_single_frame, frame) for frame in frames_to_process
        ]
        results = []

        # Wait for all threads to finish and gather results
        for future in as_completed(futures):
            result, processed_frame = future.result()
            results.append(result)

            # Display processed frame
            cv2.imshow("Frame", processed_frame)

        # Calculate the average arithmetic of the results
        valid_results = [r for r in results if r is not None]
        if valid_results:
            avg_D = np.mean(valid_results)
            print(f"[INFO] Average similarity across frames: {avg_D}")

            # Save a screenshot if average similarity is below threshold
            if avg_D < 450:  # Adjust threshold as needed
                screenshot_path = os.path.join("results", str(uuid.uuid4()) + ".jpg")
                cv2.imwrite(screenshot_path, frames_to_process[-1])
                print(f"[INFO] Screenshot saved to {screenshot_path}")

        # Reset for the next batch
        frames_to_process = []
        start_time = time.time()
        break       

    # Display the current frame
    cv2.imshow("Frame", frame)
    time.sleep(1)
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
vs.stop()
cv2.destroyAllWindows()
executor.shutdown()

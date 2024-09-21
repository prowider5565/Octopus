from imutils.video import VideoStream
import numpy as np
import insightface
import pickle
import faiss
import uuid
import cv2
import sys
import os


model = insightface.app.FaceAnalysis(name="buffalo_l")
model.prepare(ctx_id=-1)
assets_path = os.path.join("assets")
images = os.listdir(assets_path)
index = faiss.IndexFlatL2(512)
reference_names = []

for image in images:
    reference_img = cv2.imread(os.path.join(assets_path, image))
    reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)
    faces = model.get(reference_img)
    if faces:
        embedding = faces[0].embedding
        index.add(np.array([embedding]).astype("float32"))
        reference_names.append(str(image).split(".")[0])
    else:
        print(f"No face detected in {image}")

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
    
        # Threshold to determine a match (lower value means more similar)
        if D[0][0] < 500:  # Adjust the threshold as per your requirement
            # Get bounding box coordinates
            bbox = face.bbox.astype(int)  # bbox contains [x1, y1, x2, y2]

            # Get the name of the matched image from the dataset
            matched_name = reference_names[I[0][0]]

            # If recognized face matches expected user (modify this as needed)
            if (
                matched_name == "your_name"
            ):  # Replace "your_name" with your actual reference name
                recognized = True
                break

    if recognized:
        break

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
vs.stop()
cv2.destroyAllWindows()

# Exit with appropriate code for PAM
if recognized:
    sys.exit(0)  # Successful authentication
else:
    sys.exit(1)  # Failed authentication

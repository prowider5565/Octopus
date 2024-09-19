import insightface
import numpy as np
import pickle
import redis
import faiss
import cv2
import os

from redis_connection import connection

# Initialize the face recognition model from InsightFace
model = insightface.app.FaceAnalysis(name="buffalo_l")
model.prepare(ctx_id=-1)  # Use CPU (-1), or set GPU id

assets_path = os.path.join("assets")
images = os.listdir(assets_path)

# Initialize FAISS index with the dimension of face embeddings
index = faiss.IndexFlatL2(512)  # L2 distance, assuming embedding size is 512

for image in images:
    reference_img = cv2.imread(os.path.join(assets_path, image))
    reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)
    
    # Get face embeddings from the model
    faces = model.get(reference_img)
    
    if faces:  # If any face is detected
        embedding = faces[0].embedding  # Take the first face's embedding
        index.add(np.array([embedding]).astype("float32"))  # Add the embedding to FAISS index
    else:
        print(f"No face detected in {image}")

# Save the model to Redis
pickled_data = pickle.dumps(model)
connection.set("face_encoding", pickled_data)

print(connection.get("face_encoding"))

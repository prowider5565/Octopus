import insightface
import numpy as np
import pickle
import redis
import faiss
import cv2
import os

from redis_connection import connection


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

pickled_model = pickle.dumps(model)
pickled_index = pickle.dumps(index)
pickled_names = pickle.dumps(reference_names)
connection.set("face_encoding", pickled_model)
connection.set("index", pickled_index)
connection.set("reference_names", pickled_names)

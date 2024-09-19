import face_recognition
import cv2
import numpy as np
from openvino.inference_engine import IECore


# Initialize OpenVINO for GPU processing
def initialize_openvino():
    ie = IECore()
    # Load the model (e.g., face detection model converted to IR format)
    model_path = (
        "face-detection-adas-0001.xml"
    )
    net = ie.read_network(model=model_path)
    exec_net = ie.load_network(network=net, device_name="GPU")
    return exec_net


# Process input image using GPU
def process_with_openvino(exec_net, input_image):
    # Assuming input_image is preprocessed to match the model's input requirements
    output = exec_net.infer({"data": input_image})
    return output


# Function to match face with known faces
def match_faces(known_face_encodings, face_encoding):
    # Compare the input face with known faces
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)

    if matches[best_match_index]:
        return best_match_index
    return None


# Main face recognition function
def recognize_faces(input_image_path, known_images):
    # Load the input image
    input_image = face_recognition.load_image_file(input_image_path)
    input_face_encodings = face_recognition.face_encodings(input_image)

    if not input_face_encodings:
        print("No face detected in the input image.")
        return None

    # Load known face encodings
    known_face_encodings = []
    known_face_names = []

    for img_path, name in known_images:
        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(name)

    # Loop through each face found in the input image
    for face_encoding in input_face_encodings:
        match_index = match_faces(known_face_encodings, face_encoding)
        if match_index is not None:
            print(f"Matched with: {known_face_names[match_index]}")
        else:
            print("No match found.")


if __name__ == "__main__":
    # Initialize OpenVINO
    exec_net = initialize_openvino()

    # Known images and their labels
    known_images = [
        ("person1.jpg", "Person 1"),
        ("person2.jpg", "Person 2"),
        ("person3.jpg", "Person 3"),
    ]

    # Path to the input image
    input_image_path = "input_image.jpg"

    # Run the face recognition
    recognize_faces(input_image_path, known_images)

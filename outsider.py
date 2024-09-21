import face_recognition
import cv2
import sys
import os

# Load known face
KNOWN_FACE_DIR = "assets"
known_faces = []
for file in os.listdir(KNOWN_FACE_DIR):
    image = face_recognition.load_image_file(f"{KNOWN_FACE_DIR}/{file}")
    known_face_encoding = face_recognition.face_encodings(image)[0]
    known_faces.append(known_face_encoding)


def recognize_face():
    # Initialize webcam
    video_capture = cv2.VideoCapture(0)

    # Grab a frame from the webcam
    ret, frame = video_capture.read()

    if not ret:
        print("Failed to capture image.")
        return False

    # Convert the frame to RGB (OpenCV uses BGR by default)
    rgb_frame = frame[:, :, ::-1]

    # Find all faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding in face_encodings:
        # Check if the face matches a known face
        matches = face_recognition.compare_faces(known_faces, face_encoding)

        if True in matches:
            print("Face recognized, login successful.")
            return True

    print("Face not recognized.")
    return False


if __name__ == "__main__":
    if recognize_face():
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure
